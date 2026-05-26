"""Tests for get_snapshot / quote retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

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
    **extra: Any,
) -> MagicMock:
    """Mimic ib_async Ticker. Extra kwargs become attrs (modelGreeks etc.)."""
    t = MagicMock()
    t.bid = bid
    t.ask = ask
    t.last = last
    t.time = time if time is not None else datetime(2026, 5, 26, 14, 30)
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
    md: MarketDataClient, mock_ib: MagicMock
) -> None:
    # IBKR returns NaN for missing fields; our adapter converts NaN -> None.
    import math

    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="ABC", secType="STK")]
    mock_ib.reqTickersAsync.return_value = [_ticker(bid=math.nan, ask=math.nan, last=math.nan)]
    quote = await md.get_stock_snapshot("ABC")
    assert quote.bid is None
    assert quote.ask is None
    assert quote.last is None
    assert quote.mid is None


async def test_get_snapshot_records_live_when_not_paper(mock_ib: MagicMock) -> None:
    s = Settings()
    s.ibkr.paper = False
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=s, ib=mock_ib, backoff_seconds=())
    md = MarketDataClient(client)
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="SPY", secType="STK")]
    mock_ib.reqTickersAsync.return_value = [_ticker()]
    quote = await md.get_stock_snapshot("SPY")
    assert quote.delayed is False
