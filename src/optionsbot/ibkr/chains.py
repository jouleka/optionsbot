"""Options chain retrieval.

Strategy:
  1. reqSecDefOptParams -> all (expirations, strikes) for the underlying.
  2. Filter expiries to the DTE window and strikes to a near-ATM band.
  3. Build the (expiry, strike, right) leg set and fetch it in CHUNKS of
     ``max_market_data_lines`` legs. For each chunk: qualify + subscribe every
     leg to STREAMING market data (reqMktData), wait ONCE for greeks to
     populate across the whole chunk, read every ticker, then cancel every
     subscription. Greeks (IV/delta) are computed server-side only on the
     streaming feed; an options *snapshot* returns Error 10091 and would be
     billed per request.
  4. Adapt to OptionChainLeg and return as a list.

Pacing: the only IBKR streaming limit we self-enforce is the simultaneous
market-data line count, capped by the chunk size (``max_market_data_lines``,
default 50, under the 100-line account default). ib_async throttles the
outgoing request *rate* on its own (MaxRequests=45 / RequestsInterval=1s), so
no per-leg rate limiter is needed.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Iterator
from datetime import date, datetime
from typing import TYPE_CHECKING

from optionsbot.ibkr._util import clean_float, clean_int
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.types import OptionChainLeg, OptionRight

if TYPE_CHECKING:
    from ib_async import Contract

log = logging.getLogger(__name__)

_DEFAULT_MAX_LINES = 50  # simultaneous streaming market-data lines per chunk
_DEFAULT_SECDEF_RETRIES = 2
_DEFAULT_SECDEF_RETRY_DELAY = 1.0  # seconds
_DEFAULT_GREEK_TIMEOUT = 10.0  # seconds to wait for streaming greeks to populate
_DEFAULT_GREEK_POLL = 0.5  # seconds between greek-readiness polls
_DEFAULT_GREEK_STABLE_POLLS = 3  # consecutive no-improvement polls => coverage plateaued

_LegSpec = tuple[str, float, OptionRight]


def _dte(expiry: str, today: date | None = None) -> int:
    today = today or date.today()
    exp = datetime.strptime(expiry, "%Y%m%d").date()
    return (exp - today).days


def _select_strikes(
    strikes: list[float],
    reference: float,
    band_pct: float,
    max_per_side: int,
) -> list[float]:
    """Bound a strike ladder to a near-ATM window.

    Keep strikes within ``+/-band_pct`` of ``reference``, then -- as a hard
    pacing governor -- the nearest ``max_per_side`` strikes on each side
    (plus an exact-reference strike if one is listed). Returns a sorted list.
    """
    lo = reference * (1.0 - band_pct)
    hi = reference * (1.0 + band_pct)
    in_band = [s for s in strikes if lo <= s <= hi]
    below = sorted((s for s in in_band if s < reference), reverse=True)[:max_per_side]
    at = [s for s in in_band if s == reference]
    above = sorted(s for s in in_band if s > reference)[:max_per_side]
    return sorted(below + at + above)


def _clean_price(value: float | None) -> float | None:
    """Like clean_float, but also nulls IBKR's -1 'no data' sentinel for delayed
    bid/ask (a real option price is never negative)."""
    cleaned = clean_float(value)
    if cleaned is not None and cleaned < 0:
        return None
    return cleaned


def _chunked(seq: list[_LegSpec], size: int) -> Iterator[list[_LegSpec]]:
    """Yield successive ``size``-length chunks of ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _has_greeks(ticker: object) -> bool:
    """True once the streaming feed has populated model greeks for this ticker."""
    g = getattr(ticker, "modelGreeks", None)
    if g is None:
        return False
    return (
        getattr(g, "delta", None) is not None
        or getattr(g, "impliedVol", None) is not None
    )


class ChainClient:
    def __init__(
        self,
        client: IBKRClient,
        resolver: ContractResolver | None = None,
        max_market_data_lines: int = _DEFAULT_MAX_LINES,
        secdef_retries: int = _DEFAULT_SECDEF_RETRIES,
        secdef_retry_delay: float = _DEFAULT_SECDEF_RETRY_DELAY,
        greek_wait_timeout: float = _DEFAULT_GREEK_TIMEOUT,
        greek_poll_interval: float = _DEFAULT_GREEK_POLL,
        greek_stable_polls: int = _DEFAULT_GREEK_STABLE_POLLS,
    ) -> None:
        self._client = client
        self._resolver = resolver if resolver is not None else ContractResolver(client)
        self._max_lines = max_market_data_lines
        self._secdef_retries = secdef_retries
        self._secdef_retry_delay = secdef_retry_delay
        self._greek_timeout = greek_wait_timeout
        self._greek_poll = greek_poll_interval
        self._greek_stable_polls = greek_stable_polls

    async def get_chain(
        self,
        symbol: str,
        dte_window: tuple[int, int] = (25, 55),
        underlying_price: float | None = None,
        strike_band_pct: float = 0.15,
        max_strikes_per_side: int = 40,
        dte_target: int = 45,
        back_dte_gap: int | None = None,
    ) -> list[OptionChainLeg]:
        underlying = await self._resolver.stock(symbol)
        await self._client.ensure_connected()
        # reqSecDefOptParams occasionally returns a PARTIAL result right after
        # connect -- only the degenerate low-expiry SMART rows, missing the full
        # standard listing -- which filters to zero in-window expiries and would
        # silently yield an empty chain. Re-request a few times before giving up.
        lo, hi = dte_window
        expiries: list[str] = []
        strikes: list[float] = []
        for attempt in range(self._secdef_retries + 1):
            params = await self._client.ib.reqSecDefOptParamsAsync(
                underlying.symbol,
                "",  # futFopExchange
                underlying.secType,
                underlying.conId,
            )
            if params:
                # One row per (exchange, tradingClass, multiplier). Pick SMART
                # rows, then the richest by (#expirations, #strikes).
                smart = [p for p in params if getattr(p, "exchange", "") == "SMART"]
                candidates = smart or list(params)
                chosen = max(candidates, key=lambda p: (len(p.expirations), len(p.strikes)))
                all_expiries = sorted(chosen.expirations)
                in_window = [e for e in all_expiries if lo <= _dte(e) <= hi]
                if back_dte_gap is None or not in_window:
                    expiries = in_window
                else:
                    # Front = nearest dte_target in-window; back = nearest expiry
                    # >= front+gap from the FULL list (the back-month lives
                    # outside the window). Keeps Calendar/Diagonal viable.
                    front = min(in_window, key=lambda e: abs(_dte(e) - dte_target))
                    keep = {front}
                    backs = [e for e in all_expiries if _dte(e) >= _dte(front) + back_dte_gap]
                    if backs:
                        keep.add(min(backs, key=_dte))
                    expiries = sorted(keep)
                if expiries:
                    strikes = sorted(chosen.strikes)
                    break
            if attempt < self._secdef_retries:
                log.warning(
                    "reqSecDefOptParams(%s) returned no in-window expiries "
                    "(attempt %d/%d); retrying",
                    symbol,
                    attempt + 1,
                    self._secdef_retries + 1,
                )
                await asyncio.sleep(self._secdef_retry_delay)
        else:
            log.warning(
                "No in-window expiries for %s after %d attempt(s); empty chain",
                symbol,
                self._secdef_retries + 1,
            )
            return []
        if not strikes:
            return []
        # Bound the strike set to a near-ATM window. reqSecDefOptParams returns
        # the UNION of strikes across all expiries (~497 for SPY); fetching the
        # full ladder would be thousands of legs. Keep strikes within
        # +/-strike_band_pct of a reference price (the underlying spot when
        # known, else the median listed strike), capped at max_strikes_per_side.
        reference = (
            underlying_price
            if underlying_price is not None and underlying_price > 0
            else statistics.median(strikes)
        )
        # Select strikes PER EXPIRY from that expiry's REAL listed grid rather
        # than the cross-expiry UNION reqSecDefOptParams returns. A far-dated
        # month lists a sparser grid (e.g. $5 vs the front week's $1), so the
        # union contains strikes that don't exist for it -- qualifying those
        # spams IBKR Error 200 and wastes round trips. ``listed_strikes`` also
        # primes the qualify cache, so _fetch_chunk re-qualifies nothing. Fall
        # back to the union if per-expiry enumeration is empty (no worse than
        # before, never drops the expiry).
        rights: tuple[OptionRight, ...] = ("C", "P")
        specs: list[_LegSpec] = []
        for expiry in expiries:
            listed = await self._resolver.listed_strikes(symbol, expiry)
            expiry_strikes = listed or strikes
            selected = _select_strikes(
                expiry_strikes, reference, strike_band_pct, max_strikes_per_side
            )
            specs.extend(
                (expiry, strike, right) for strike in selected for right in rights
            )
        if not specs:
            return []
        # Fetch the leg set in chunks bounded by the simultaneous market-data
        # line cap. Each chunk subscribes all its legs, waits ONCE, reads, then
        # cancels -- so greeks for a whole chunk compute in parallel server-side
        # instead of one leg at a time.
        legs: list[OptionChainLeg] = []
        for chunk in _chunked(specs, self._max_lines):
            legs.extend(await self._fetch_chunk(symbol, chunk))
        return legs

    async def _fetch_chunk(
        self, symbol: str, specs: list[_LegSpec]
    ) -> list[OptionChainLeg]:
        """Qualify + subscribe every leg in the chunk, wait once for greeks,
        adapt every ticker, then release every streaming line."""
        subscribed: list[tuple[_LegSpec, Contract, object]] = []
        try:
            # Qualify the whole chunk in ONE call (ib_async qualifies concurrently)
            # instead of a round-trip per leg.
            contracts = await self._resolver.qualify_options(symbol, specs)
            for spec in specs:
                contract = contracts.get(spec)
                if contract is None:
                    continue  # unqualifiable -> skip (same as the old per-leg ValueError)
                expiry, strike, right = spec
                # STREAMING reqMktData (NOT a snapshot): IBKR computes option
                # greeks only on the streaming feed. reqMktData returns a live
                # Ticker synchronously; it fills in over the next seconds.
                try:
                    ticker = self._client.ib.reqMktData(contract, "", False, False)
                except Exception:  # noqa: BLE001 -- one bad leg must not kill the chunk
                    log.warning(
                        "reqMktData failed for %s %s %.2f %s; skipping leg",
                        symbol, expiry, strike, right,
                    )
                    continue
                subscribed.append((spec, contract, ticker))
            # ONE wait for the whole chunk. The await yields to the event loop so
            # ib_async can process inbound greek ticks and fill the tickers. Poll
            # until every live ticker has greeks, or the timeout elapses (a
            # permanently-illiquid leg simply adapts with iv/delta=None).
            tickers = [t for _, _, t in subscribed]
            prev = -1
            stable = 0
            for _ in range(max(1, int(self._greek_timeout / self._greek_poll))):
                count = sum(_has_greeks(t) for t in tickers)
                if count == len(tickers):
                    break  # full coverage
                # Coverage is monotonic (modelGreeks, once set, stays set). Once
                # it stops rising for greek_stable_polls polls the remaining legs
                # are illiquid and won't resolve -- stop waiting. The count > 0
                # guard avoids a false break during warmup (before any greeks).
                if count > 0 and count == prev:
                    stable += 1
                    if stable >= self._greek_stable_polls:
                        break
                else:
                    stable = 0
                prev = count
                await asyncio.sleep(self._greek_poll)
            return [
                self._adapt_ticker(symbol, expiry, strike, right, ticker)
                for (expiry, strike, right), _, ticker in subscribed
            ]
        finally:
            for _, contract, _ in subscribed:
                self._client.ib.cancelMktData(contract)

    @staticmethod
    def _adapt_ticker(
        symbol: str,
        expiry: str,
        strike: float,
        right: OptionRight,
        ticker: object,
    ) -> OptionChainLeg:
        g = getattr(ticker, "modelGreeks", None)
        return OptionChainLeg(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            bid=_clean_price(getattr(ticker, "bid", None)),
            ask=_clean_price(getattr(ticker, "ask", None)),
            iv=clean_float(getattr(g, "impliedVol", None)) if g is not None else None,
            delta=clean_float(getattr(g, "delta", None)) if g is not None else None,
            gamma=clean_float(getattr(g, "gamma", None)) if g is not None else None,
            theta=clean_float(getattr(g, "theta", None)) if g is not None else None,
            vega=clean_float(getattr(g, "vega", None)) if g is not None else None,
            open_interest=clean_int(getattr(ticker, "openInterest", None)),
            volume=clean_int(getattr(ticker, "volume", None)),
        )
