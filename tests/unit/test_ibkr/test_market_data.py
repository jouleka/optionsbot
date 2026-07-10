"""Tests for get_snapshot / quote retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from ib_async import Ticker

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.types import OptionQuote, StockQuote


def _ticker(
    *,
    bid: float = 400.0,
    ask: float = 400.2,
    last: float = 400.1,
    time: datetime | None = None,
    market_data_type: int | None = None,
    **extra: Any,
) -> MagicMock:
    """Mimic ib_async Ticker. Extra kwargs become attrs (modelGreeks etc.)."""
    t = MagicMock()
    t.bid = bid
    t.ask = ask
    t.last = last
    t.time = time if time is not None else datetime(2026, 5, 26, 14, 30)
    t.marketDataType = market_data_type
    for k, v in extra.items():
        setattr(t, k, v)
    return t


@pytest.fixture()
def md(mock_ib: MagicMock) -> MarketDataClient:
    s = Settings()
    s.ibkr.paper = True
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=s, ib=mock_ib, backoff_seconds=())
    return MarketDataClient(client)


async def test_ticker_live_default_without_callback_is_unknown(
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    stock_contract = MagicMock(symbol="SPY", secType="STK")
    mock_ib.qualifyContractsAsync.return_value = [stock_contract]
    ticker = Ticker()
    ticker.bid = 400.0
    ticker.ask = 400.2
    ticker.last = 400.1
    ticker.time = datetime(2026, 5, 26, 14, 30)
    mock_ib.reqTickersAsync.return_value = [ticker]

    quote = await md.get_stock_snapshot("SPY")

    assert quote.delayed is True


async def test_explicit_live_callback_marks_returned_ticker_live(
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    stock_contract = MagicMock(symbol="SPY", secType="STK")
    ticker = _ticker(market_data_type=1)
    mock_ib.qualifyContractsAsync.return_value = [stock_contract]
    mock_ib.wrapper.reqId2Ticker = {41: ticker}

    async def _snapshot_with_callback(*contracts: object) -> list[MagicMock]:
        mock_ib.wrapper.marketDataType(41, 1)
        return [ticker]

    mock_ib.reqTickersAsync.side_effect = _snapshot_with_callback

    quote = await md.get_stock_snapshot("SPY")

    assert quote.delayed is False


async def test_discarded_snapshot_callback_cannot_prove_later_request_live(
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    import math

    stock_contract = MagicMock(symbol="SPY", secType="STK")
    discarded = _ticker(bid=math.nan, ask=math.nan, last=math.nan)
    streamed = _ticker(bid=400.0, ask=400.2, last=400.1)
    mock_ib.qualifyContractsAsync.return_value = [stock_contract]
    mock_ib.wrapper.reqId2Ticker = {46: discarded}
    calls = 0

    async def _snapshots(*contracts: object) -> list[MagicMock]:
        nonlocal calls
        calls += 1
        if calls == 1:
            mock_ib.wrapper.marketDataType(46, 1)
        return [discarded]

    mock_ib.reqTickersAsync.side_effect = _snapshots
    mock_ib.reqMktData.return_value = streamed
    mock_ib.cancelMktData = MagicMock()

    first = await md.get_stock_snapshot("SPY")
    assert first.delayed is True  # streaming ticker received no callback

    discarded.bid = 400.0
    discarded.ask = 400.2
    discarded.last = 400.1
    second = await md.get_stock_snapshot("SPY")

    assert second.delayed is True  # prior callback is stale, not request provenance


@pytest.mark.parametrize("callback_value", [True, 1.0, "1", 2, None])
async def test_only_exact_integer_live_callback_proves_live_delivery(
    md: MarketDataClient, mock_ib: MagicMock, callback_value: object
) -> None:
    stock_contract = MagicMock(symbol="SPY", secType="STK")
    ticker = _ticker(market_data_type=1)
    mock_ib.qualifyContractsAsync.return_value = [stock_contract]
    mock_ib.wrapper.reqId2Ticker = {42: ticker}

    async def _snapshot_with_callback(*contracts: object) -> list[MagicMock]:
        mock_ib.wrapper.marketDataType(42, callback_value)
        return [ticker]

    mock_ib.reqTickersAsync.side_effect = _snapshot_with_callback

    quote = await md.get_stock_snapshot("SPY")

    assert quote.delayed is True


def test_market_data_callback_observer_is_installed_once(
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    first_observer = mock_ib.wrapper.marketDataType

    second = MarketDataClient(md.client)

    assert second is not md
    assert mock_ib.wrapper.marketDataType is first_observer


async def test_get_snapshot_stock(md: MarketDataClient, mock_ib: MagicMock) -> None:
    stock_contract = MagicMock(symbol="SPY", secType="STK")
    mock_ib.qualifyContractsAsync.return_value = [stock_contract]
    mock_ib.reqTickersAsync.return_value = [_ticker(bid=400.0, ask=400.2, last=400.1)]
    quote = await md.get_stock_snapshot("SPY")
    assert isinstance(quote, StockQuote)
    assert quote.symbol == "SPY"
    assert quote.bid == 400.0
    assert quote.ask == 400.2
    assert quote.mid == pytest.approx(400.1)
    assert quote.delayed is True  # paper mode default


async def test_option_snapshot_preserves_missing_timestamp_as_unknown(
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    opt_contract = MagicMock(
        symbol="SPY",
        secType="OPT",
        lastTradeDateOrContractMonth="20260619",
        strike=400.0,
        right="C",
    )
    ticker = _ticker(bid=5.0, ask=5.1)
    ticker.time = None
    mock_ib.qualifyContractsAsync.return_value = [opt_contract]
    mock_ib.reqTickersAsync.return_value = [ticker]

    quote = await md.get_option_snapshot("SPY", "20260619", 400.0, "C")

    assert quote.ts is None


async def test_get_snapshot_option_extracts_greeks(
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    opt_contract = MagicMock(
        symbol="SPY",
        secType="OPT",
        lastTradeDateOrContractMonth="20260619",
        strike=400.0,
        right="C",
    )
    mock_ib.qualifyContractsAsync.return_value = [opt_contract]
    greeks = MagicMock(impliedVol=0.18, delta=0.5, gamma=0.02, theta=-0.04, vega=0.6)
    mock_ib.reqTickersAsync.return_value = [
        _ticker(
            bid=5.0,
            ask=5.1,
            last=5.05,
            modelGreeks=greeks,
            openInterest=1000,
            volume=42,
        )
    ]
    quote = await md.get_option_snapshot("SPY", "20260619", 400.0, "C")
    assert isinstance(quote, OptionQuote)
    assert quote.iv == pytest.approx(0.18)
    assert quote.delta == pytest.approx(0.5)
    assert quote.open_interest == 1000
    assert quote.volume == 42


async def test_get_snapshot_handles_missing_fields(
    md: MarketDataClient, mock_ib: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # IBKR returns NaN for missing fields; our adapter converts NaN -> None.
    # An empty snapshot now triggers the streaming fallback — keep it instant
    # and have the stream also return NaN (no quote anywhere).
    import math

    from optionsbot.ibkr import market_data

    monkeypatch.setattr(market_data, "_STREAM_TIMEOUT_S", 0.02)
    monkeypatch.setattr(market_data, "_STREAM_POLL_S", 0.01)
    nan_ticker = _ticker(bid=math.nan, ask=math.nan, last=math.nan)
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="ABC", secType="STK")]
    mock_ib.reqTickersAsync.return_value = [nan_ticker]
    mock_ib.reqMktData = MagicMock(return_value=nan_ticker)
    mock_ib.cancelMktData = MagicMock()
    quote = await md.get_stock_snapshot("ABC")
    assert quote.bid is None
    assert quote.ask is None
    assert quote.last is None
    assert quote.mid is None


async def test_empty_snapshot_falls_back_to_streaming_quote(
    md: MarketDataClient, mock_ib: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The delayed-data case: the one-shot snapshot is empty, but a streaming
    # subscription fills the quote a moment later. This is the fix for option
    # /execute pricing on a delayed feed.
    import math

    from optionsbot.ibkr import market_data

    monkeypatch.setattr(market_data, "_STREAM_TIMEOUT_S", 1.0)
    monkeypatch.setattr(market_data, "_STREAM_POLL_S", 0.01)
    mock_ib.qualifyContractsAsync.return_value = [
        MagicMock(symbol="SPY", secType="OPT", lastTradeDateOrContractMonth="20260717",
                  strike=755.0, right="C")
    ]
    mock_ib.reqTickersAsync.return_value = [_ticker(bid=math.nan, ask=math.nan, last=math.nan)]
    # Streaming subscription returns a ticker that already carries the quote
    # (simulating the delayed tick that arrives shortly after subscribing).
    mock_ib.reqMktData = MagicMock(return_value=_ticker(bid=13.08, ask=13.12, last=13.10))
    mock_ib.cancelMktData = MagicMock()
    quote = await md.get_option_snapshot("SPY", "20260717", 755.0, "C")
    assert quote.bid == pytest.approx(13.08)
    assert quote.ask == pytest.approx(13.12)
    assert quote.mid == pytest.approx(13.10)
    mock_ib.cancelMktData.assert_called_once()


async def test_get_snapshot_maps_unset_greek_sentinel_to_none(
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    # IBKR sends -2 for a greek it couldn't compute; ib_async leaks the literal -2.0 for
    # theta/vega. The adapter must map it to None so portfolio Greeks (IBK-115) aren't
    # corrupted by a bogus -2.0 (the real delta is preserved).
    mock_ib.qualifyContractsAsync.return_value = [
        MagicMock(symbol="SPY", secType="OPT", lastTradeDateOrContractMonth="20260619",
                  strike=400.0, right="C")
    ]
    greeks = MagicMock(impliedVol=0.18, delta=0.5, gamma=0.02, theta=-2.0, vega=-2.0)
    mock_ib.reqTickersAsync.return_value = [_ticker(modelGreeks=greeks)]
    quote = await md.get_option_snapshot("SPY", "20260619", 400.0, "C")
    assert quote.delta == pytest.approx(0.5)
    assert quote.theta is None and quote.vega is None


async def test_get_snapshot_records_live_when_not_paper(mock_ib: MagicMock) -> None:
    s = Settings()
    s.ibkr.paper = False
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=s, ib=mock_ib, backoff_seconds=())
    md = MarketDataClient(client)
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="SPY", secType="STK")]
    ticker = _ticker(market_data_type=1)
    mock_ib.wrapper.reqId2Ticker = {43: ticker}

    async def _snapshot_with_callback(*contracts: object) -> list[MagicMock]:
        mock_ib.wrapper.marketDataType(43, 1)
        return [ticker]

    mock_ib.reqTickersAsync.side_effect = _snapshot_with_callback
    quote = await md.get_stock_snapshot("SPY")
    assert quote.delayed is False


async def test_actual_delayed_feed_overrides_live_configuration(
    mock_ib: MagicMock,
) -> None:
    s = Settings()
    s.ibkr.paper = True
    s.ibkr.market_data_type = 1
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=s, ib=mock_ib, backoff_seconds=())
    md = MarketDataClient(client)
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="SPY", secType="STK")]
    ticker = _ticker(market_data_type=3)
    mock_ib.wrapper.reqId2Ticker = {44: ticker}

    async def _snapshot_with_callback(*contracts: object) -> list[MagicMock]:
        mock_ib.wrapper.marketDataType(44, 3)
        return [ticker]

    mock_ib.reqTickersAsync.side_effect = _snapshot_with_callback

    quote = await md.get_stock_snapshot("SPY")

    assert quote.delayed is True


async def test_actual_live_feed_overrides_delayed_configuration(
    mock_ib: MagicMock,
) -> None:
    s = Settings()
    s.ibkr.paper = True
    s.ibkr.market_data_type = 3
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=s, ib=mock_ib, backoff_seconds=())
    md = MarketDataClient(client)
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="SPY", secType="STK")]
    ticker = _ticker(market_data_type=1)
    mock_ib.wrapper.reqId2Ticker = {45: ticker}

    async def _snapshot_with_callback(*contracts: object) -> list[MagicMock]:
        mock_ib.wrapper.marketDataType(45, 1)
        return [ticker]

    mock_ib.reqTickersAsync.side_effect = _snapshot_with_callback

    quote = await md.get_stock_snapshot("SPY")

    assert quote.delayed is False
