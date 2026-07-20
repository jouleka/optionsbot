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

from optionsbot.ibkr._util import (
    clean_float,
    option_open_interest,
    option_volume,
)
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


def _ticker_ts(ticker: Ticker) -> datetime | None:
    ts = getattr(ticker, "time", None)
    if isinstance(ts, datetime):
        # Preserve tz if present; else treat as UTC (IBKR returns UTC).
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts
    return None


def _market_data_type_delayed(observed: object) -> bool:
    """Only an explicitly observed, exact IBKR integer ``1`` proves live data."""
    return type(observed) is not int or observed != 1


class MarketDataClient:
    def __init__(
        self, client: IBKRClient, resolver: ContractResolver | None = None
    ) -> None:
        self._client = client
        self._resolver = resolver if resolver is not None else ContractResolver(client)
        self._delivered_market_data_types: dict[int, tuple[int, object]] = {}
        self._market_data_type_wrapper: object | None = None
        self._install_market_data_type_observer()

    def _install_market_data_type_observer(self) -> None:
        """Record IBKR delivery callbacks before ``reqTickersAsync`` returns.

        ``ib_async.Ticker.marketDataType`` defaults to ``1`` even when IBKR has
        not sent the callback. Wrapping the adapter's underlying callback keeps
        provenance separate from that unsafe object default. The wrapper-level
        observation map keeps repeated adapter construction idempotent.
        """
        wrapper = getattr(self._client.ib, "wrapper", None)
        self._market_data_type_wrapper = wrapper
        observations = getattr(wrapper, "_optionsbot_market_data_types", None)
        if isinstance(observations, dict):
            self._delivered_market_data_types = observations
            return
        callback = getattr(wrapper, "marketDataType", None)
        if wrapper is None or not callable(callback):
            return
        observations = {}
        self._delivered_market_data_types = observations
        wrapper._optionsbot_market_data_types = observations

        def _observe(req_id: int, market_data_type: object) -> None:
            callback(req_id, market_data_type)
            req_id_to_ticker = getattr(wrapper, "reqId2Ticker", None)
            if not isinstance(req_id_to_ticker, dict):
                return
            ticker = req_id_to_ticker.get(req_id)
            if ticker is not None:
                previous = getattr(
                    wrapper, "_optionsbot_market_data_type_sequence", 0
                )
                sequence = previous + 1 if type(previous) is int else 1
                wrapper._optionsbot_market_data_type_sequence = sequence
                observations[id(ticker)] = (sequence, market_data_type)

        wrapper.marketDataType = _observe

    def _market_data_type_sequence(self) -> int:
        sequence = getattr(
            self._market_data_type_wrapper,
            "_optionsbot_market_data_type_sequence",
            0,
        )
        return sequence if type(sequence) is int else 0

    def _take_observed_market_data_type(
        self, ticker: Ticker, *, after_sequence: int
    ) -> object | None:
        observation = self._delivered_market_data_types.pop(id(ticker), None)
        if (
            not isinstance(observation, tuple)
            or len(observation) != 2
            or type(observation[0]) is not int
            or observation[0] <= after_sequence
        ):
            return None
        return observation[1]

    def _has_current_market_data_type_observation(
        self, ticker: Ticker, *, after_sequence: int
    ) -> bool:
        """Whether IBKR identified the feed for this subscription request.

        ``reqMktData`` may return an already-populated cached ``Ticker``. Its
        fields prove neither that this request is live nor that its quote is
        fresh, so review snapshots must wait for a new marketDataType callback
        instead of treating the cached fields as a completed request.
        """
        observation = self._delivered_market_data_types.get(id(ticker))
        return (
            isinstance(observation, tuple)
            and len(observation) == 2
            and type(observation[0]) is int
            and observation[0] > after_sequence
        )

    @property
    def client(self) -> IBKRClient:
        return self._client

    async def get_stock_snapshot(self, symbol: str) -> StockQuote:
        contract = await self._resolver.stock(symbol)
        ticker, observed_market_data_type = await self._fetch_ticker(contract)
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
            delayed=_market_data_type_delayed(observed_market_data_type),
        )

    async def get_option_snapshot(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: OptionRight,
    ) -> OptionQuote:
        contract = await self._resolver.option(symbol, expiry, strike, right)
        ticker, observed_market_data_type = await self._fetch_ticker(contract)
        bid = clean_float(getattr(ticker, "bid", None))
        ask = clean_float(getattr(ticker, "ask", None))
        last = clean_float(getattr(ticker, "last", None))
        greeks = getattr(ticker, "modelGreeks", None)
        iv = _greek(greeks, "impliedVol") if greeks is not None else None
        delta = _greek(greeks, "delta") if greeks is not None else None
        gamma = _greek(greeks, "gamma") if greeks is not None else None
        theta = _greek(greeks, "theta") if greeks is not None else None
        vega = _greek(greeks, "vega") if greeks is not None else None
        open_interest = option_open_interest(ticker, right)
        volume = option_volume(ticker, right)
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
            delayed=_market_data_type_delayed(observed_market_data_type),
        )

    async def get_option_review_snapshot(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: OptionRight,
    ) -> OptionQuote:
        """Fetch a review quote including option volume and open interest.

        Execution pricing only needs NBBO/Greeks and uses the faster snapshot
        path above.  Hermes also audits OI/volume, which IBKR exposes through
        generic ticks 100/101, so alerted candidates use this bounded stream.
        """
        contract = await self._resolver.option(symbol, expiry, strike, right)
        await self._client.ensure_connected()
        start_sequence = self._market_data_type_sequence()
        ticker = self._client.ib.reqMktData(contract, "100,101,106", False, False)
        try:
            waited = 0.0
            while waited < _STREAM_TIMEOUT_S:
                greeks = getattr(ticker, "modelGreeks", None)
                complete = (
                    _has_quote(ticker)
                    and self._has_current_market_data_type_observation(
                        ticker, after_sequence=start_sequence
                    )
                    and greeks is not None
                    and all(
                        _greek(greeks, name) is not None
                        for name in ("impliedVol", "delta", "gamma", "theta", "vega")
                    )
                    and option_open_interest(ticker, right) is not None
                    and option_volume(ticker, right) is not None
                )
                if complete:
                    break
                await asyncio.sleep(_STREAM_POLL_S)
                waited += _STREAM_POLL_S
        finally:
            self._client.ib.cancelMktData(contract)
        observed = self._take_observed_market_data_type(
            ticker, after_sequence=start_sequence
        )
        bid = clean_float(getattr(ticker, "bid", None))
        ask = clean_float(getattr(ticker, "ask", None))
        greeks = getattr(ticker, "modelGreeks", None)
        return OptionQuote(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            bid=bid,
            ask=ask,
            last=clean_float(getattr(ticker, "last", None)),
            mid=_mid(bid, ask),
            iv=_greek(greeks, "impliedVol") if greeks is not None else None,
            delta=_greek(greeks, "delta") if greeks is not None else None,
            gamma=_greek(greeks, "gamma") if greeks is not None else None,
            theta=_greek(greeks, "theta") if greeks is not None else None,
            vega=_greek(greeks, "vega") if greeks is not None else None,
            open_interest=option_open_interest(ticker, right),
            volume=option_volume(ticker, right),
            ts=_ticker_ts(ticker),
            delayed=_market_data_type_delayed(observed),
        )

    async def _fetch_ticker(self, contract: Contract) -> tuple[Ticker, object | None]:
        await self._client.ensure_connected()
        start_sequence = self._market_data_type_sequence()
        tickers = await self._client.ib.reqTickersAsync(contract)
        ticker = tickers[0] if tickers else None
        if ticker is not None and _has_quote(ticker):
            observed = self._take_observed_market_data_type(
                ticker, after_sequence=start_sequence
            )
            return ticker, observed  # fast path: snapshot had the quote
        streamed = await self._stream_until_quote(contract)
        if streamed is not None:
            if ticker is not None and streamed is not ticker:
                self._delivered_market_data_types.pop(id(ticker), None)
            observed = self._take_observed_market_data_type(
                streamed, after_sequence=start_sequence
            )
            return streamed, observed
        if ticker is not None:
            observed = self._take_observed_market_data_type(
                ticker, after_sequence=start_sequence
            )
            return ticker, observed  # empty, but real; caller sees None bid/ask
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
