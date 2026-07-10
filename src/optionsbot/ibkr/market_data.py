"""Snapshot market data via reqTickersAsync.

``reqTickersAsync`` is preferred over ``reqMktData`` for one-shot
snapshots because it returns a fully-populated Ticker without leaving
a subscription open.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from optionsbot.ibkr._util import clean_float, clean_int
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.types import OptionQuote, OptionRight, StockQuote

if TYPE_CHECKING:
    from ib_async import Contract, Ticker

log = logging.getLogger(__name__)

# A one-shot snapshot (reqTickersAsync) frequently returns an empty ticker on
# a DELAYED feed — the bid/ask only arrive a couple seconds later over a
# streaming subscription. When that happens we briefly stream until the quote
# populates (the same thing the chain client does for scans), then cancel.
_STREAM_TIMEOUT_S = 6.0
_STREAM_POLL_S = 0.4


def _has_quote(ticker: object) -> bool:
    return (
        clean_float(getattr(ticker, "bid", None)) is not None
        and clean_float(getattr(ticker, "ask", None)) is not None
    )


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2


def _greek(greeks: object, name: str) -> float | None:
    """A model-Greek value with IBKR's -2.0 'not computed' sentinel mapped to None.

    ib_async nulls delta/gamma on the -2 sentinel but leaks the literal -2.0 for theta/
    vega, and clean_float keeps it (it's finite). Summing that into portfolio Greeks
    (IBK-115) would corrupt net theta/vega, so guard every model Greek here at the source."""
    v = clean_float(getattr(greeks, name, None))
    return None if v == -2.0 else v


def _ticker_ts(ticker: Ticker) -> datetime:
    ts = getattr(ticker, "time", None)
    if isinstance(ts, datetime):
        # Preserve tz if present; else treat as UTC (IBKR returns UTC).
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts
    return datetime.now(UTC)


def _ticker_delayed(ticker: Ticker) -> bool:
    """Classify from the feed IBKR actually delivered, not requested config.

    IBKR marketDataType 1 is live; 2/3/4 are frozen or delayed. An absent or
    malformed value is conservatively non-live so execution cannot treat an
    unknown quote as fresh real-time data.
    """
    try:
        market_data_type = int(ticker.marketDataType)
    except (AttributeError, TypeError, ValueError):
        return True
    return market_data_type != 1


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
        bid = clean_float(getattr(ticker, "bid", None))
        ask = clean_float(getattr(ticker, "ask", None))
        last = clean_float(getattr(ticker, "last", None))
        return StockQuote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            mid=_mid(bid, ask),
            ts=_ticker_ts(ticker),
            delayed=_ticker_delayed(ticker),
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
        bid = clean_float(getattr(ticker, "bid", None))
        ask = clean_float(getattr(ticker, "ask", None))
        last = clean_float(getattr(ticker, "last", None))
        greeks = getattr(ticker, "modelGreeks", None)
        iv = _greek(greeks, "impliedVol") if greeks is not None else None
        delta = _greek(greeks, "delta") if greeks is not None else None
        gamma = _greek(greeks, "gamma") if greeks is not None else None
        theta = _greek(greeks, "theta") if greeks is not None else None
        vega = _greek(greeks, "vega") if greeks is not None else None
        open_interest = clean_int(getattr(ticker, "openInterest", None))
        volume = clean_int(getattr(ticker, "volume", None))
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
            delayed=_ticker_delayed(ticker),
        )

    async def _fetch_ticker(self, contract: Contract) -> Ticker:
        await self._client.ensure_connected()
        tickers = await self._client.ib.reqTickersAsync(contract)
        ticker = tickers[0] if tickers else None
        if ticker is not None and _has_quote(ticker):
            return ticker  # fast path: snapshot had the quote (always for stocks)
        streamed = await self._stream_until_quote(contract)
        if streamed is not None:
            return streamed
        if ticker is not None:
            return ticker  # empty, but a real Ticker — caller sees None bid/ask
        raise ValueError(f"No ticker returned for {contract!r}")

    async def _stream_until_quote(self, contract: Contract) -> Ticker | None:
        """Open a streaming subscription and wait (bounded) for bid/ask to
        arrive — the reliable path for delayed feeds. Always cancels."""
        ib = self._client.ib
        ticker = ib.reqMktData(contract, "", False, False)
        try:
            waited = 0.0
            while waited < _STREAM_TIMEOUT_S:
                if _has_quote(ticker):
                    return ticker
                await asyncio.sleep(_STREAM_POLL_S)
                waited += _STREAM_POLL_S
        finally:
            try:
                ib.cancelMktData(contract)
            except Exception:  # noqa: BLE001 -- cancel best-effort; never mask the quote
                log.debug("cancelMktData failed for %r", contract, exc_info=True)
        return ticker if _has_quote(ticker) else None
