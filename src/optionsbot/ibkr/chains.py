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
from datetime import date, datetime

from optionsbot.ibkr._util import clean_float, clean_int
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.pacing import ConcurrencyLimiter, RateLimiter
from optionsbot.ibkr.types import OptionChainLeg, OptionRight

_DEFAULT_MAX_CONCURRENT = 8
_DEFAULT_RATE_LIMIT = 50  # calls per window
_DEFAULT_RATE_WINDOW = 10.0  # seconds


def _dte(expiry: str, today: date | None = None) -> int:
    today = today or date.today()
    exp = datetime.strptime(expiry, "%Y%m%d").date()
    return (exp - today).days


class ChainClient:
    def __init__(
        self,
        client: IBKRClient,
        resolver: ContractResolver | None = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        max_calls_per_window: int = _DEFAULT_RATE_LIMIT,
        window_seconds: float = _DEFAULT_RATE_WINDOW,
    ) -> None:
        self._client = client
        self._resolver = resolver if resolver is not None else ContractResolver(client)
        self._concurrency = ConcurrencyLimiter(max_concurrent)
        self._rate = RateLimiter(max_calls_per_window, window_seconds)

    async def get_chain(
        self,
        symbol: str,
        dte_window: tuple[int, int] = (25, 55),
    ) -> list[OptionChainLeg]:
        underlying = await self._resolver.stock(symbol)
        await self._client.ensure_connected()
        # ib_async.IB has TWO related methods:
        #   - reqSecDefOptParams (synchronous, wraps the async sibling via IB._run)
        #   - reqSecDefOptParamsAsync (the awaitable form)
        # We use the async sibling so we don't block the event loop in _run.
        params = await self._client.ib.reqSecDefOptParamsAsync(
            underlying.symbol,
            "",  # futFopExchange
            underlying.secType,
            underlying.conId,
        )
        if not params:
            return []
        # Pick the SMART entry; fall back to first row.
        chosen = next(
            (p for p in params if getattr(p, "exchange", "") == "SMART"),
            params[0],
        )
        expiries: list[str] = sorted(chosen.expirations)
        strikes: list[float] = sorted(chosen.strikes)
        # Filter to DTE window.
        lo, hi = dte_window
        expiries = [e for e in expiries if lo <= _dte(e) <= hi]
        if not expiries:
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
