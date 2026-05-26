"""Tests for positions + account-summary read-only access."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.ibkr.types import AccountSummary, PositionRecord


def _ib_position(symbol="SPY", sec_type="STK", position=100.0, avg_cost=399.5) -> MagicMock:
    c = MagicMock(symbol=symbol, secType=sec_type, exchange="SMART", currency="USD")
    p = MagicMock(account="DU1234567", contract=c, position=position, avgCost=avg_cost)
    return p


def _account_value(tag: str, value: str, currency: str = "USD") -> MagicMock:
    v = MagicMock()
    v.tag = tag
    v.value = value
    v.currency = currency
    return v


@pytest.fixture()
def positions_client(mock_ib) -> PositionsClient:
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    return PositionsClient(client, cache_ttl_seconds=60.0)


async def test_get_positions_flattens_contract_fields(positions_client, mock_ib) -> None:
    mock_ib.positions.return_value = [
        _ib_position("SPY", "STK", 100.0, 399.5),
        _ib_position("QQQ", "STK", 50.0, 350.0),
    ]
    out = await positions_client.get_positions()
    assert all(isinstance(p, PositionRecord) for p in out)
    assert {p.symbol for p in out} == {"SPY", "QQQ"}
    spy = next(p for p in out if p.symbol == "SPY")
    assert spy.sec_type == "STK"
    assert spy.position == 100.0
    assert spy.avg_cost == 399.5
    assert spy.account == "DU1234567"


async def test_get_positions_caches_within_ttl(positions_client, mock_ib) -> None:
    mock_ib.positions.return_value = [_ib_position()]
    await positions_client.get_positions()
    await positions_client.get_positions()
    assert mock_ib.positions.call_count == 1


async def test_get_positions_refreshes_after_ttl(mock_ib) -> None:
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    pc = PositionsClient(client, cache_ttl_seconds=0.05)
    mock_ib.positions.return_value = [_ib_position()]
    await pc.get_positions()
    time.sleep(0.1)
    await pc.get_positions()
    assert mock_ib.positions.call_count == 2


async def test_get_account_summary_extracts_tags(positions_client, mock_ib) -> None:
    mock_ib.accountSummary.return_value = [
        _account_value("NetLiquidation", "10000.00"),
        _account_value("BuyingPower", "20000.00"),
        _account_value("AvailableFunds", "9500.00"),
        _account_value("SomeOtherTag", "ignored"),
    ]
    summary = await positions_client.get_account_summary()
    assert isinstance(summary, AccountSummary)
    assert summary.net_liquidation == Decimal("10000.00")
    assert summary.buying_power == Decimal("20000.00")
    assert summary.available_funds == Decimal("9500.00")
    assert summary.currency == "USD"


async def test_get_account_summary_handles_missing_tags(positions_client, mock_ib) -> None:
    mock_ib.accountSummary.return_value = [_account_value("BuyingPower", "5000.00")]
    summary = await positions_client.get_account_summary()
    assert summary.net_liquidation is None
    assert summary.buying_power == Decimal("5000.00")
    assert summary.available_funds is None
