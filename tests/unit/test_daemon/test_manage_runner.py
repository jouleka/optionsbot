"""Tests for the position-management tick (IBK-113)."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from optionsbot.daemon.manage_runner import run_manage_tick
from optionsbot.ibkr.types import PortfolioPosition, StockQuote
from optionsbot.storage.schema import position_alerts


def _short_put(strike: float = 95.0, days: int = 5) -> PortfolioPosition:
    # Expiry is computed relative to the real date.today() (run_manage_tick reads it),
    # so the DTE bucket is deterministic regardless of when the suite runs: +5d -> urgent.
    expiry = (date.today() + timedelta(days=days)).strftime("%Y%m%d")
    return PortfolioPosition(
        account="DU1", symbol="SPY", sec_type="OPT", expiry=expiry, strike=strike,
        right="P", multiplier=100, position=-1.0, avg_cost=250.0, market_price=1.0,
        market_value=-100.0, unrealized_pnl=10.0, realized_pnl=0.0,
    )


def _quote(mid: float = 99.0) -> StockQuote:
    return StockQuote(symbol="SPY", bid=mid - 0.1, ask=mid + 0.1, last=mid, mid=mid,
                      ts=datetime.now(UTC), delayed=True)


def _ctx(daemon_engine, daemon_settings) -> MagicMock:
    ctx = MagicMock()
    ctx.engine = daemon_engine
    ctx.settings = daemon_settings
    ctx.ibkr = MagicMock()
    ctx.resolver = MagicMock()
    ctx.ibkr_lock = asyncio.Lock()
    ctx.alerting_paused = False
    ctx.telegram = MagicMock()
    ctx.telegram.send_message = AsyncMock(return_value=1)
    return ctx


async def test_manage_tick_sends_and_dedups(daemon_engine, daemon_settings) -> None:
    ctx = _ctx(daemon_engine, daemon_settings)
    with patch("optionsbot.daemon.manage_runner.PositionsClient") as PC, \
         patch("optionsbot.daemon.manage_runner.MarketDataClient") as MD, \
         patch("optionsbot.daemon.manage_runner.is_market_open", return_value=True):
        PC.return_value.get_portfolio = AsyncMock(return_value=[_short_put()])
        MD.return_value.get_stock_snapshot = AsyncMock(return_value=_quote(99.0))  # OTM put
        summary = await run_manage_tick(ctx)
        assert summary.alerts_sent == 1  # dte_urgent only (OTM -> no assignment)
        ctx.telegram.send_message.assert_awaited_once()
        with daemon_engine.connect() as conn:
            rows = conn.execute(select(position_alerts)).fetchall()
        assert len(rows) == 1 and rows[0].dedup_key.endswith(":dte_urgent")
        # Second tick within cooldown -> deduped, nothing sent.
        ctx.telegram.send_message.reset_mock()
        summary2 = await run_manage_tick(ctx)
        assert summary2.alerts_sent == 0
        ctx.telegram.send_message.assert_not_awaited()


async def test_manage_tick_noop_when_market_closed(daemon_engine, daemon_settings) -> None:
    ctx = _ctx(daemon_engine, daemon_settings)
    with patch("optionsbot.daemon.manage_runner.is_market_open", return_value=False):
        summary = await run_manage_tick(ctx)
    assert summary.alerts_sent == 0
    ctx.telegram.send_message.assert_not_awaited()


async def test_manage_tick_noop_when_paused(daemon_engine, daemon_settings) -> None:
    ctx = _ctx(daemon_engine, daemon_settings)
    ctx.alerting_paused = True
    with patch("optionsbot.daemon.manage_runner.is_market_open", return_value=True):
        summary = await run_manage_tick(ctx)
    assert summary.alerts_sent == 0


async def test_manage_tick_noop_when_disabled(daemon_engine, daemon_settings) -> None:
    daemon_settings.manage.enabled = False
    ctx = _ctx(daemon_engine, daemon_settings)
    with patch("optionsbot.daemon.manage_runner.is_market_open", return_value=True):
        summary = await run_manage_tick(ctx)
    assert summary.alerts_sent == 0


async def test_manage_tick_tolerates_spot_failure(daemon_engine, daemon_settings) -> None:
    ctx = _ctx(daemon_engine, daemon_settings)
    with patch("optionsbot.daemon.manage_runner.PositionsClient") as PC, \
         patch("optionsbot.daemon.manage_runner.MarketDataClient") as MD, \
         patch("optionsbot.daemon.manage_runner.is_market_open", return_value=True):
        PC.return_value.get_portfolio = AsyncMock(return_value=[_short_put()])
        MD.return_value.get_stock_snapshot = AsyncMock(side_effect=RuntimeError("no data"))
        summary = await run_manage_tick(ctx)
    # Spot fetch failed -> assignment skipped, but the 5-DTE urgent alert still sends.
    assert summary.alerts_sent == 1


def _profit_book() -> list[PortfolioPosition]:
    # QQQ short put, far OTM vs the mocked spot (99) and far-dated, so the ONLY possible
    # trigger is profit: avg_cost 250 (net credit), up $200 -> 80% -> take_profit.
    return [PortfolioPosition(
        account="DU1", symbol="QQQ", sec_type="OPT", expiry="20260717", strike=80.0,
        right="P", multiplier=100, position=-1.0, avg_cost=250.0, market_price=0.5,
        market_value=-50.0, unrealized_pnl=200.0, realized_pnl=0.0,
    )]


async def test_manage_tick_sends_profit_alert(daemon_engine, daemon_settings) -> None:
    ctx = _ctx(daemon_engine, daemon_settings)
    with patch("optionsbot.daemon.manage_runner.PositionsClient") as PC, \
         patch("optionsbot.daemon.manage_runner.MarketDataClient") as MD, \
         patch("optionsbot.daemon.manage_runner.is_market_open", return_value=True):
        PC.return_value.get_portfolio = AsyncMock(return_value=_profit_book())
        MD.return_value.get_stock_snapshot = AsyncMock(return_value=_quote(99.0))
        summary = await run_manage_tick(ctx)
        assert summary.alerts_sent == 1
        with daemon_engine.connect() as conn:
            rows = conn.execute(select(position_alerts)).fetchall()
        assert len(rows) == 1 and rows[0].dedup_key == "QQQ:profit:take_profit"
        # second tick within cooldown -> deduped
        ctx.telegram.send_message = AsyncMock(return_value=1)
        assert (await run_manage_tick(ctx)).alerts_sent == 0


async def test_profit_alerts_disabled_suppresses(daemon_engine, daemon_settings) -> None:
    daemon_settings.manage.profit_alerts = False
    ctx = _ctx(daemon_engine, daemon_settings)
    with patch("optionsbot.daemon.manage_runner.PositionsClient") as PC, \
         patch("optionsbot.daemon.manage_runner.MarketDataClient") as MD, \
         patch("optionsbot.daemon.manage_runner.is_market_open", return_value=True):
        PC.return_value.get_portfolio = AsyncMock(return_value=_profit_book())
        MD.return_value.get_stock_snapshot = AsyncMock(return_value=_quote(99.0))
        summary = await run_manage_tick(ctx)
    # OTM + far-dated, profit off -> no trigger at all.
    assert summary.alerts_sent == 0
