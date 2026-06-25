"""Tests for the full-auto entry hook + loss kill-triggers (IBK-130)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert, update

from optionsbot.daemon.auto_executor import auto_execute_candidates
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import nyse_session_date
from optionsbot.daemon.order_watcher import run_orders_tick
from optionsbot.execution.engine import ExecuteOutcome
from optionsbot.execution.orders import record_fill, set_fill_commission
from optionsbot.execution.state import load_state
from optionsbot.storage.schema import orders, snapshots, strategy_scores

NOW = datetime.now(UTC)


def _pick(context: DaemonContext, symbol: str = "SPY") -> tuple[int, int]:
    with context.engine.begin() as conn:
        snap = int(conn.execute(insert(snapshots).values(
            symbol=symbol, ts=NOW, spot=600.0,
        )).inserted_primary_key[0])
        score = int(conn.execute(insert(strategy_scores).values(
            snapshot_id=snap, strategy="bull_put_spread", score=80.0,
            rationale="t", legs_json=[], suggestion_json={},
        )).inserted_primary_key[0])
    return snap, score


async def test_auto_executes_alerted_candidates(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    snap, score = _pick(daemon_context)
    scored = MagicMock()
    scored.strategy_name = "bull_put_spread"
    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(return_value=ExecuteOutcome(ok=True, message="✅ submitted #9", order_id=9)),
    ) as run:
        n = await auto_execute_candidates(daemon_context, [("SPY", scored, snap)])
    assert n == 1
    assert run.await_args.args[1] == score
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("🤖" in m and "submitted #9" in m for m in sent)


async def test_auto_noop_in_confirm_mode(daemon_context: DaemonContext) -> None:
    daemon_context.order_client = MagicMock()
    snap, _ = _pick(daemon_context)
    scored = MagicMock()
    scored.strategy_name = "bull_put_spread"
    with patch(
        "optionsbot.execution.engine.execute_pick", new=AsyncMock()
    ) as run:
        n = await auto_execute_candidates(daemon_context, [("SPY", scored, snap)])
    assert n == 0
    run.assert_not_awaited()


# --- loss kill-triggers (order watcher) -----------------------------------------------


def _closed_pair(context: DaemonContext, *, pnl_credit: float, closed_ts: datetime) -> int:
    engine = context.engine
    with engine.begin() as conn:
        epk = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread", legs_json=[],
            quantity=1, status="filled", staged_ts=closed_ts - timedelta(days=3),
            terminal_ts=closed_ts - timedelta(days=3), reprice_count=0,
        )).inserted_primary_key[0])
        cpk = int(conn.execute(insert(orders).values(
            intent="close", closes_order_id=epk, symbol="SPY",
            strategy="bull_put_spread", legs_json=[], quantity=1, status="filled",
            staged_ts=closed_ts, terminal_ts=closed_ts, reprice_count=0,
        )).inserted_primary_key[0])
        for oid in (epk, cpk):
            conn.execute(update(orders).where(orders.c.id == oid)
                         .values(order_ref=f"obot-{oid}"))
    # entry collects pnl_credit, close costs 0 -> pair pnl = pnl_credit*100 - commissions
    record_fill(engine, epk, exec_id=f"k{epk}", side="SELL",
                price=max(pnl_credit, 0.01) if pnl_credit > 0 else 0.10,
                qty=1, ts=closed_ts - timedelta(days=3))
    close_price = 0.01 if pnl_credit > 0 else 0.10 - pnl_credit
    record_fill(engine, cpk, exec_id=f"k{cpk}", side="BUY",
                price=close_price, qty=1, ts=closed_ts)
    set_fill_commission(engine, f"k{epk}", 0.65)
    set_fill_commission(engine, f"k{cpk}", 0.65)
    return cpk


async def test_consecutive_losses_trip_kill(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.max_consecutive_losses = 2
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=5))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch("optionsbot.daemon.order_watcher._net_liq", new=AsyncMock(return_value=100_000.0)):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is True
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("consecutive" in m.lower() for m in sent)


async def test_daily_loss_trips_kill(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.max_consecutive_losses = 99
    daemon_context.settings.execution.max_daily_loss_pct = 0.02
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    # One big loss today: -25/unit x 100 = -2,501.3 incl. commissions on 100k = -2.5%.
    _closed_pair(daemon_context, pnl_credit=-25.0, closed_ts=NOW - timedelta(minutes=1))
    with patch("optionsbot.daemon.order_watcher._net_liq", new=AsyncMock(return_value=100_000.0)):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is True
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("daily loss" in m.lower() for m in sent)


async def test_winning_closes_do_not_kill(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.max_consecutive_losses = 2
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    _closed_pair(daemon_context, pnl_credit=1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch("optionsbot.daemon.order_watcher._net_liq", new=AsyncMock(return_value=100_000.0)):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is False


async def test_unrealized_drawdown_trips_kill_with_nothing_closed(
    daemon_context: DaemonContext,
) -> None:
    # PHASE 0 B1: no closed pairs at all — purely a mark-to-market decline past
    # the cap must trip the kill on a normal tick.
    from optionsbot.daemon.market_hours import nyse_session_date
    from optionsbot.execution.equity_guard import capture_day_start_net_liq

    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    # Pre-capture the day-start baseline keyed to TODAY's NYSE session so the
    # breaker sees the 100k baseline when the tick runs with mocked 97k net-liq.
    today_session = nyse_session_date(NOW).isoformat()
    capture_day_start_net_liq(daemon_context.engine, 100_000.0, session=today_session)
    daemon_context.day_start_net_liq = 100_000.0
    # 97k = 3% down, cap is 2% -> trip. Nothing closed (close_filled stays False).
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=97_000.0),
    ):
        await run_orders_tick(daemon_context)
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert "net liq" in (state.reason or "").lower()
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("net-liq" in m.lower() or "net liq" in m.lower() for m in sent)


async def test_unrealized_drawdown_under_cap_does_not_trip(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.execution.equity_guard import capture_day_start_net_liq

    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    capture_day_start_net_liq(
        daemon_context.engine, 100_000.0, session=nyse_session_date(NOW).isoformat()
    )
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=99_500.0),  # 0.5% down
    ):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is False


async def test_daily_loss_window_uses_et_session_not_utc(
    daemon_context: DaemonContext,
) -> None:
    # PHASE 0 B2: a loss closed at 11:00 ET (15:00 UTC, 2026-06-15) must count
    # toward the SAME ET session as a 'now' of 20:30 ET (00:30 UTC, 2026-06-16).
    # A UTC-keyed day_start (00:00 UTC the 16th) would EXCLUDE the loss and the
    # kill would not trip; the ET anchor INCLUDES it and it DOES.
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    daemon_context.settings.execution.max_consecutive_losses = 99
    daemon_context.settings.execution.max_daily_loss_pct = 0.02
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()

    now_utc = _dt(2026, 6, 15, 20, 30, tzinfo=et).astimezone(UTC)
    loss_close_utc = _dt(2026, 6, 15, 11, 0, tzinfo=et).astimezone(UTC)
    # Set watermark before the loss so the close shows as newly-terminal (triggers close_filled).
    daemon_context.orders_notified_through = loss_close_utc - timedelta(minutes=1)
    cpk = _closed_pair(daemon_context, pnl_credit=-25.0, closed_ts=loss_close_utc)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(orders).where(orders.c.id == cpk).values(terminal_ts=loss_close_utc)
        )
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=100_000.0),
    ):
        await run_orders_tick(daemon_context, now=now_utc)
    assert load_state(daemon_context.engine).killed is True


async def test_consecutive_loss_ignores_prior_session_losses(
    daemon_context: DaemonContext,
) -> None:
    # PHASE 0 B3: two losses from YESTERDAY's session + one fresh loss today must
    # NOT trip a 3-consecutive-loss kill — only THIS session's losses count.
    daemon_context.settings.execution.max_consecutive_losses = 3
    daemon_context.settings.execution.max_daily_loss_pct = 0.99  # don't let daily-loss trip
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(days=2)
    # Two stale losses ~26-27h ago (prior ET session).
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(hours=27))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(hours=26))
    # One fresh loss this session.
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=100_000.0),
    ):
        await run_orders_tick(daemon_context)
    # Only ONE loss this session < limit of 3 -> NOT killed.
    assert load_state(daemon_context.engine).killed is False


async def test_consecutive_loss_trips_within_session(
    daemon_context: DaemonContext,
) -> None:
    # Three losses THIS session in a row -> trips, proving the session scoping
    # didn't simply disable the check.
    daemon_context.settings.execution.max_consecutive_losses = 3
    daemon_context.settings.execution.max_daily_loss_pct = 0.99
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=2)
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=20))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=10))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=100_000.0),
    ):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is True
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("consecutive" in m.lower() for m in sent)
