"""Snapshot market data via reqTickersAsync.

``reqTickersAsync`` is preferred over ``reqMktData`` for one-shot
snapshots because it returns a fully-populated Ticker without leaving
a subscription open.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.types import OptionQuote, OptionRight, StockQuote

if TYPE_CHECKING:
    from ib_async import Contract, Ticker


def _clean(value: float | None) -> float | None:
    """IBKR uses NaN for missing fields; convert to None."""
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        return value
    return value


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2


def _ticker_ts(ticker: Ticker) -> datetime:
    ts = getattr(ticker, "time", None)
    if isinstance(ts, datetime):
        # Preserve tz if present; else treat as UTC (IBKR returns UTC).
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts
    return datetime.now(UTC)


def _clean_int(value: float | int | None) -> int | None:
    """Convert ib_async float-typed counts (open_interest, volume) to int or None.

    Returns None when the value is missing, NaN, or otherwise not convertible.
    """
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        # Not a float-like; fall through to int() and let it raise/succeed.
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class MarketDataClient:
    def __init__(
        self, client: IBKRClient, resolver: ContractResolver | None = None
    ) -> None:
        self._client = client
        self._resolver = resolver if resolver is not None else ContractResolver(client)

    @property
    def client(self) -> IBKRClient:
        return self._client

    async def get_stock_snapshot(self, symbol: str) -> StockQuote:
        contract = await self._resolver.stock(symbol)
        ticker = await self._fetch_ticker(contract)
        bid = _clean(getattr(ticker, "bid", None))
        ask = _clean(getattr(ticker, "ask", None))
        last = _clean(getattr(ticker, "last", None))
        return StockQuote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            mid=_mid(bid, ask),
            ts=_ticker_ts(ticker),
            delayed=self._client.settings.ibkr.paper,
        )

    async def get_option_snapshot(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: OptionRight,
    ) -> OptionQuote:
        contract = await self._resolver.option(symbol, expiry, strike, right)
        ticker = await self._fetch_ticker(contract)
        bid = _clean(getattr(ticker, "bid", None))
        ask = _clean(getattr(ticker, "ask", None))
        last = _clean(getattr(ticker, "last", None))
        greeks = getattr(ticker, "modelGreeks", None)
        iv = _clean(getattr(greeks, "impliedVol", None)) if greeks is not None else None
        delta = _clean(getattr(greeks, "delta", None)) if greeks is not None else None
        gamma = _clean(getattr(greeks, "gamma", None)) if greeks is not None else None
        theta = _clean(getattr(greeks, "theta", None)) if greeks is not None else None
        vega = _clean(getattr(greeks, "vega", None)) if greeks is not None else None
        open_interest = _clean_int(getattr(ticker, "openInterest", None))
        volume = _clean_int(getattr(ticker, "volume", None))
        return OptionQuote(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            bid=bid,
            ask=ask,
            last=last,
            mid=_mid(bid, ask),
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            open_interest=open_interest,
            volume=volume,
            ts=_ticker_ts(ticker),
            delayed=self._client.settings.ibkr.paper,
        )

    async def _fetch_ticker(self, contract: Contract) -> Ticker:
        await self._client.ensure_connected()
        tickers = await self._client.ib.reqTickersAsync(contract)
        if not tickers:
            raise ValueError(f"No ticker returned for {contract!r}")
        return tickers[0]
