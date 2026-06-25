"""Tests for positions + account-summary read-only access (incl. EUR->USD)."""

from __future__ import annotations

import sys
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.ibkr.types import AccountSummary, PortfolioPosition, PositionRecord


def _ib_position(symbol="SPY", sec_type="STK", position=100.0, avg_cost=399.5) -> MagicMock:
    c = MagicMock(symbol=symbol, secType=sec_type, exchange="SMART", currency="USD")
    p = MagicMock(account="DU1234567", contract=c, position=position, avgCost=avg_cost)
    return p


def _ib_portfolio_item(
    symbol="SPY", sec_type="OPT", expiry="20260717", strike=95.0, right="P",
    multiplier="100", position=-1.0, avg_cost=250.0, market_price=1.1,
    market_value=-110.0, upnl=45.0, rpnl=0.0,
) -> MagicMock:
    c = MagicMock(symbol=symbol, secType=sec_type, lastTradeDateOrContractMonth=expiry,
                  strike=strike, right=right, multiplier=multiplier)
    return MagicMock(account="DU1234567", contract=c, position=position, averageCost=avg_cost,
                     marketPrice=market_price, marketValue=market_value,
                     unrealizedPNL=upnl, realizedPNL=rpnl)


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


async def test_get_portfolio_maps_option_and_stock(positions_client, mock_ib) -> None:
    mock_ib.portfolio.return_value = [
        _ib_portfolio_item(),  # SPY option
        _ib_portfolio_item(symbol="AAPL", sec_type="STK", expiry="", strike=0.0, right="",
                           multiplier="", position=100.0, avg_cost=180.0, market_price=185.0,
                           market_value=18500.0, upnl=500.0),
    ]
    out = await positions_client.get_portfolio()
    assert all(isinstance(p, PortfolioPosition) for p in out)
    opt = next(p for p in out if p.symbol == "SPY")
    assert opt.sec_type == "OPT" and opt.expiry == "20260717" and opt.strike == 95.0
    assert opt.right == "P" and opt.multiplier == 100 and opt.position == -1.0
    assert opt.unrealized_pnl == 45.0 and opt.market_value == -110.0
    assert opt.market_price == 1.1 and opt.realized_pnl == 0.0
    stk = next(p for p in out if p.symbol == "AAPL")
    assert stk.sec_type == "STK" and stk.expiry is None and stk.strike is None
    assert stk.right is None and stk.multiplier == 1 and stk.unrealized_pnl == 500.0


async def test_get_portfolio_caches_within_ttl(positions_client, mock_ib) -> None:
    mock_ib.portfolio.return_value = [_ib_portfolio_item()]
    await positions_client.get_portfolio()
    await positions_client.get_portfolio()
    assert mock_ib.portfolio.call_count == 1


async def test_get_portfolio_treats_unset_double_as_none(positions_client, mock_ib) -> None:
    # IBKR sends sys.float_info.max as the "unset double" sentinel for a not-yet-priced
    # leg; it must surface as None, not a 309-digit number (IBK-112 review).
    big = sys.float_info.max
    mock_ib.portfolio.return_value = [
        _ib_portfolio_item(market_price=big, market_value=big, upnl=big, rpnl=big),
    ]
    (p,) = await positions_client.get_portfolio()
    assert p.market_price is None and p.market_value is None
    assert p.unrealized_pnl is None and p.realized_pnl is None


async def test_get_account_summary_extracts_tags(positions_client, mock_ib) -> None:
    mock_ib.accountSummaryAsync.return_value = [
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
    # Pin the async path: reverting to the blocking sync ib.accountSummary()
    # (which raises "event loop is already running" inside a running loop)
    # must fail this test, not pass by coincidence.
    mock_ib.accountSummaryAsync.assert_awaited_once()
    mock_ib.accountSummary.assert_not_called()


async def test_get_account_summary_handles_missing_tags(positions_client, mock_ib) -> None:
    mock_ib.accountSummaryAsync.return_value = [_account_value("BuyingPower", "5000.00")]
    summary = await positions_client.get_account_summary()
    assert summary.net_liquidation is None
    assert summary.buying_power == Decimal("5000.00")
    assert summary.available_funds is None


# ---------------------------------------------------------------------------
# EUR -> USD conversion tests (Task 3 / IBK-122)
# ---------------------------------------------------------------------------

def _row(tag: str, value: str, currency: str) -> MagicMock:
    r = MagicMock()
    r.tag = tag
    r.value = value
    r.currency = currency
    return r


def test_account_summary_net_liq_usd_identity_for_usd() -> None:
    s = AccountSummary(
        net_liquidation=Decimal("5000"), buying_power=None,
        available_funds=Decimal("5000"), currency="USD",
    )
    assert s.fx_to_usd == Decimal(1)
    assert s.net_liquidation_usd == Decimal("5000")


def test_account_summary_net_liq_usd_converts_eur() -> None:
    # base EUR; ExchangeRate(USD)=0.80 EUR/USD -> usd_per_base = 1/0.80 = 1.25
    s = AccountSummary(
        net_liquidation=Decimal("5000"), buying_power=None,
        available_funds=Decimal("4000"), currency="EUR",
        fx_to_usd=Decimal("1.25"),
    )
    assert s.net_liquidation_usd == Decimal("6250")


def test_account_summary_net_liq_usd_none_when_netliq_none() -> None:
    s = AccountSummary(
        net_liquidation=None, buying_power=None,
        available_funds=None, currency="USD",
    )
    assert s.net_liquidation_usd is None


async def test_get_account_summary_parses_eur_exchange_rate() -> None:
    ib = MagicMock()
    ib.accountSummaryAsync = AsyncMock(return_value=[
        _row("NetLiquidation", "5000", "EUR"),
        _row("AvailableFunds", "4000", "EUR"),
        _row("ExchangeRate", "0.80", "USD"),   # 0.80 EUR per USD
        _row("ExchangeRate", "1", "EUR"),
    ])
    client = MagicMock()
    client.ib = ib
    client.ensure_connected = AsyncMock()
    pc = PositionsClient(client)
    summary = await pc.get_account_summary()
    assert summary.currency == "EUR"
    assert summary.fx_to_usd == Decimal("1.25")          # 1 / 0.80
    assert summary.net_liquidation_usd == Decimal("6250")  # 5000 * 1.25
