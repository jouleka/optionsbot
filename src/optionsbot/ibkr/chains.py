"""Options chain retrieval.

Strategy:
  1. reqSecDefOptParams -> all (expirations, strikes) for the underlying.
  2. Filter expiries to the DTE window.
  3. For each (expiry, strike, right) in the cross-product, qualify the
     option contract and request a snapshot ticker.
  4. Adapt to OptionChainLeg and return as a list.

Concurrency is bounded by ConcurrencyLimiter; the rate at which we
hit the gateway is gated by RateLimiter to stay under IBKR pacing.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import date, datetime

from optionsbot.ibkr._util import clean_float, clean_int
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.pacing import ConcurrencyLimiter, RateLimiter
from optionsbot.ibkr.types import OptionChainLeg, OptionRight

log = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENT = 8
_DEFAULT_RATE_LIMIT = 50  # calls per window
_DEFAULT_RATE_WINDOW = 10.0  # seconds
_DEFAULT_SECDEF_RETRIES = 2
_DEFAULT_SECDEF_RETRY_DELAY = 1.0  # seconds


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


class ChainClient:
    def __init__(
        self,
        client: IBKRClient,
        resolver: ContractResolver | None = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        max_calls_per_window: int = _DEFAULT_RATE_LIMIT,
        window_seconds: float = _DEFAULT_RATE_WINDOW,
        secdef_retries: int = _DEFAULT_SECDEF_RETRIES,
        secdef_retry_delay: float = _DEFAULT_SECDEF_RETRY_DELAY,
    ) -> None:
        self._client = client
        self._resolver = resolver if resolver is not None else ContractResolver(client)
        self._concurrency = ConcurrencyLimiter(max_concurrent)
        self._rate = RateLimiter(max_calls_per_window, window_seconds)
        self._secdef_retries = secdef_retries
        self._secdef_retry_delay = secdef_retry_delay

    async def get_chain(
        self,
        symbol: str,
        dte_window: tuple[int, int] = (25, 55),
        underlying_price: float | None = None,
        strike_band_pct: float = 0.15,
        max_strikes_per_side: int = 40,
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
            # ib_async exposes reqSecDefOptParams (sync wrapper) and the async
            # sibling; use the async form so we don't block the event loop.
            params = await self._client.ib.reqSecDefOptParamsAsync(
                underlying.symbol,
                "",  # futFopExchange
                underlying.secType,
                underlying.conId,
            )
            if params:
                # One row per (exchange, tradingClass, multiplier). A single
                # underlying can expose MULTIPLE SMART rows: e.g. SPY returns
                # both the standard class (~37 expiries / ~497 strikes) and a
                # degenerate 4-expiry / 4-strike class. Pick SMART rows, then
                # the richest by (#expirations, #strikes) for the full listing.
                smart = [p for p in params if getattr(p, "exchange", "") == "SMART"]
                candidates = smart or list(params)
                chosen = max(candidates, key=lambda p: (len(p.expirations), len(p.strikes)))
                expiries = [e for e in sorted(chosen.expirations) if lo <= _dte(e) <= hi]
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
        # the UNION of strikes across all expiries (~497 for SPY) -- fetching the
        # full ladder x every in-window expiry x {C,P} is thousands of legs and
        # minutes under IBKR pacing. Keep strikes within +/-strike_band_pct of a
        # reference price (the underlying spot when known, else the median listed
        # strike as an ATM proxy), capped at max_strikes_per_side on each side.
        reference = (
            underlying_price
            if underlying_price is not None and underlying_price > 0
            else statistics.median(strikes)
        )
        strikes = _select_strikes(strikes, reference, strike_band_pct, max_strikes_per_side)
        if not strikes:
            return []
        # Fetch each (expiry, strike, right) under concurrency + rate limits.
        # ``return_exceptions=True`` so a single transient per-leg failure
        # doesn't cancel the whole chain; we drop failed legs after the fact.
        rights: tuple[OptionRight, ...] = ("C", "P")
        tasks = [
            self._fetch_one(symbol, expiry, strike, right)
            for expiry in expiries
            for strike in strikes
            for right in rights
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, OptionChainLeg)]

    async def _fetch_one(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: OptionRight,
    ) -> OptionChainLeg | None:
        await self._rate.acquire()
        async with self._concurrency:
            try:
                contract = await self._resolver.option(symbol, expiry, strike, right)
            except ValueError:
                return None
            tickers = await self._client.ib.reqTickersAsync(contract)
            if not tickers:
                return None
            t = tickers[0]
            bid = clean_float(getattr(t, "bid", None))
            ask = clean_float(getattr(t, "ask", None))
            g = getattr(t, "modelGreeks", None)
            return OptionChainLeg(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                bid=bid,
                ask=ask,
                iv=clean_float(getattr(g, "impliedVol", None)) if g is not None else None,
                delta=clean_float(getattr(g, "delta", None)) if g is not None else None,
                gamma=clean_float(getattr(g, "gamma", None)) if g is not None else None,
                theta=clean_float(getattr(g, "theta", None)) if g is not None else None,
                vega=clean_float(getattr(g, "vega", None)) if g is not None else None,
                open_interest=clean_int(getattr(t, "openInterest", None)),
                volume=clean_int(getattr(t, "volume", None)),
            )
