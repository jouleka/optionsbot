"""Tests for historical bars + disk cache."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.history import HistoryClient


def _bar(d: date, *, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000) -> MagicMock:
    b = MagicMock()
    b.date = d
    b.open = open
    b.high = high
    b.low = low
    b.close = close
    b.volume = volume
    return b


@pytest.fixture()
def history(mock_ib, tmp_path) -> HistoryClient:
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    return HistoryClient(client, cache_dir=tmp_path / "history")


async def test_get_history_returns_dataframe(history, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="SPY", secType="STK")]
    today = date.today()
    bars = [_bar(today - timedelta(days=i)) for i in range(5)]
    mock_ib.reqHistoricalDataAsync.return_value = list(reversed(bars))  # ascending order
    df = await history.get_history("SPY", days=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"


async def test_get_history_caches_on_disk(history, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="SPY", secType="STK")]
    today = date.today()
    bars = [_bar(today - timedelta(days=i)) for i in range(3)]
    mock_ib.reqHistoricalDataAsync.return_value = list(reversed(bars))
    df1 = await history.get_history("SPY", days=3)
    df2 = await history.get_history("SPY", days=3)  # cache hit
    pd.testing.assert_frame_equal(df1, df2)
    # Only one fetch from IBKR.
    assert mock_ib.reqHistoricalDataAsync.await_count == 1


async def test_get_history_cache_keyed_by_symbol(history, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.side_effect = [
        [MagicMock(symbol="SPY", secType="STK")],
        [MagicMock(symbol="QQQ", secType="STK")],
    ]
    today = date.today()
    mock_ib.reqHistoricalDataAsync.return_value = [_bar(today)]
    await history.get_history("SPY", days=1)
    await history.get_history("QQQ", days=1)
    assert mock_ib.reqHistoricalDataAsync.await_count == 2


async def test_get_history_writes_parquet_to_cache_dir(history, mock_ib, tmp_path) -> None:
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="SPY", secType="STK")]
    today = date.today()
    mock_ib.reqHistoricalDataAsync.return_value = [_bar(today)]
    await history.get_history("SPY", days=1)
    cached = list((tmp_path / "history").glob("SPY-*.parquet"))
    assert len(cached) == 1


async def test_get_history_raises_when_no_bars(history, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = [MagicMock(symbol="ZZZ", secType="STK")]
    mock_ib.reqHistoricalDataAsync.return_value = []
    with pytest.raises(ValueError, match="No historical bars"):
        await history.get_history("ZZZ", days=5)
