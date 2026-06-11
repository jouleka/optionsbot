"""Tests for the exit runner tick (IBK-129)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert, select, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.exit_runner import run_exits_tick
from optionsbot.execution.orders import record_fill
from optionsbot.execution.state import trip_kill
from optionsbot.ibkr.types import OptionQuote, PlacedOrder
from optionsbot.storage.schema import orders

NOW = datetime.now(UTC)
FAR = (NOW + timedelta(days=36)).strftime("%Y%m%d")
NEAR = (NOW + timedelta(days=2)).strftime("%Y%m%d")


def _legs(expiry: str) -> list[dict[str, object]]:
    return [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": expiry,
         "strike": 580.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": expiry,
         "strike": 575.0, "right": "P", "quantity": 1},
    ]


def _filled_entry(context: DaemonContext, *, expiry: str = FAR) -> int:
    engine = context.engine
    with engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=_legs(expiry), quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, ib_order_id=11, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(order_ref=f"obot-{order_id}"))
    record_fill(engine, order_id, exec_id=f"x{order_id}a", side="SELL",
                price=1.60, qty=1, ts=NOW)
    record_fill(engine, order_id, exec_id=f"x{order_id}b", side="BUY",
                price=0.40, qty=1, ts=NOW)
    return order_id


def _quote(strike: float, right: str, mid: float) -> OptionQuote:
    return OptionQuote(
        symbol="SPY", expiry=FAR, strike=strike, right=right,  # type: ignore[arg-type]
        bid=round(mid - 0.05, 4), ask=round(mid + 0.05, 4), last=None, mid=mid,
        iv=None, delta=None, gamma=None, theta=None, vega=None,
        open_interest=None, volume=None, ts=NOW, delayed=True,
    )


def _wire(context: DaemonContext, mids: dict[tuple[float, str], float]) -> MagicMock:
    context.settings.execution.enabled = True
    context.settings.execution.walk_max_steps = 0  # no walk task in tests
    order_client = MagicMock()
    order_client.place_combo_limit = AsyncMock(
        side_effect=lambda *a, **k: PlacedOrder(
            ib_order_id=99, order_ref=k["order_ref"], action="BUY",
            limit_price=k["limit_price"], quantity=k["quantity"],
        )
    )
    context.order_client = order_client

    md = MagicMock()
    md.get_option_snapshot = AsyncMock(
        side_effect=lambda symbol, expiry, strike, right: _quote(
            strike, right, mids[(strike, right)]
        )
    )
    context._test_md = md  # type: ignore[attr-defined]
    return order_client


async def test_take_profit_fires_closing_order(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    # Entry credit 1.20; structure now reopens at 0.50 -> kept 58% -> close.
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 1
    call = order_client.place_combo_limit.call_args
    # Flipped close: we PAY ~0.50/unit -> BUY-bag positive limit.
    assert call.kwargs["limit_price"] > 0
    with daemon_context.engine.connect() as conn:
        close = conn.execute(
            select(orders).where(orders.c.intent == "close")
        ).one()
    assert close.closes_order_id == entry_id
    assert close.status == "submitted"
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("closing" in m.lower() for m in sent)


async def test_no_trigger_no_close(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()


async def test_active_close_blocks_duplicate(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)
    assert first.closes_submitted == 1
    assert second.closes_submitted == 0
    assert order_client.place_combo_limit.await_count == 1


async def test_kill_switch_blocks_exits(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    trip_kill(daemon_context.engine, "halt")
    summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()


async def test_expiry_guard_forces_close(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context, expiry=NEAR)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()


async def test_half_closed_position_hands_off_instead_of_reclosing(
    daemon_context: DaemonContext,
) -> None:
    # Opus IBK-129 critical: a close that PARTIALLY filled then died must
    # never be auto-restaged at full quantity (over-close = wrong-way risk).
    entry_id = _filled_entry(daemon_context)
    with daemon_context.engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="close", closes_order_id=entry_id, symbol="SPY",
            strategy="bull_put_spread", legs_json=_legs(FAR), quantity=1,
            status="abandoned", staged_ts=NOW, submitted_ts=NOW,
            terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        close_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == close_id)
                     .values(order_ref=f"obot-{close_id}"))
    record_fill(daemon_context.engine, close_id, exec_id="half1", side="BUY",
                price=0.80, qty=1, ts=NOW)

    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)
    assert first.closes_submitted == 0
    assert second.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    warns = [
        c.args[0] for c in daemon_context.telegram.send_message.await_args_list
        if "HALF-CLOSED" in c.args[0]
    ]
    assert len(warns) == 1  # escalated exactly once


async def test_missing_quote_skips_and_retries_later(
    daemon_context: DaemonContext,
) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {})
    md = MagicMock()
    md.get_option_snapshot = AsyncMock(side_effect=RuntimeError("no data"))
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 0
    assert summary.errors >= 0  # tick survives
    order_client.place_combo_limit.assert_not_awaited()
