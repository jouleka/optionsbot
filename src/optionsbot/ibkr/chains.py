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
import math
from datetime import date, datetime

from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.pacing import ConcurrencyLimiter, RateLimiter
from optionsbot.ibkr.types import OptionChainLeg, OptionRight

_DEFAULT_MAX_CONCURRENT = 8
_DEFAULT_RATE_LIMIT = 50  # calls per window
_DEFAULT_RATE_WINDOW = 10.0  # seconds


def _clean(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        return value
    return value


def _int_or_none(v: float | int | None) -> int | None:
    if v is None:
        return None
    try:
        if math.isnan(v):
            return None
    except TypeError:
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


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
        # reqSecDefOptParams returns OptionChain rows; pick the SMART/STK row.
        # ib_async's type stubs annotate this as sync-blocking via ``_run``,
        # but the underlying implementation returns an awaitable future. We
        # use the documented method name and ignore the spurious mypy error.
        params = await self._client.ib.reqSecDefOptParams(  # type: ignore[misc]
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
        rights: tuple[OptionRight, ...] = ("C", "P")
        tasks = [
            self._fetch_one(symbol, expiry, strike, right)
            for expiry in expiries
            for strike in strikes
            for right in rights
        ]
        results = await asyncio.gather(*tasks)
        # Drop None entries (failed/missing fetches).
        return [r for r in results if r is not None]

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
            bid = _clean(getattr(t, "bid", None))
            ask = _clean(getattr(t, "ask", None))
            g = getattr(t, "modelGreeks", None)
            return OptionChainLeg(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                bid=bid,
                ask=ask,
                iv=_clean(getattr(g, "impliedVol", None)) if g is not None else None,
                delta=_clean(getattr(g, "delta", None)) if g is not None else None,
                gamma=_clean(getattr(g, "gamma", None)) if g is not None else None,
                theta=_clean(getattr(g, "theta", None)) if g is not None else None,
                vega=_clean(getattr(g, "vega", None)) if g is not None else None,
                open_interest=_int_or_none(getattr(t, "openInterest", None)),
                volume=_int_or_none(getattr(t, "volume", None)),
            )
