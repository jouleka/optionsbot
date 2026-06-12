"""Tests for the 1-minute order watcher (IBK-126): TTL sweep + notifications."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import Engine, insert, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.order_watcher import run_orders_tick
from optionsbot.execution.orders import get_order, record_fill
from optionsbot.storage.schema import orders

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)


def _insert_order(
    engine: Engine,
    status: str,
    *,
    submitted_ts: datetime | None = None,
    terminal_ts: datetime | None = None,
    ib_order_id: int | None = 11,
    last_error: str | None = None,
) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            insert(orders).values(
                intent="open", symbol="SPY", strategy="bull_put_spread",
                legs_json=[], quantity=1, status=status, staged_ts=NOW,
                submitted_ts=submitted_ts, terminal_ts=terminal_ts,
                ib_order_id=ib_order_id, reprice_count=0, last_error=last_error,
            )
        )
        pk = result.inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(
            update(orders).where(orders.c.id == order_id)
            .values(order_ref=f"obot-{order_id}")
        )
    return order_id


def _wire_exec(daemon_context: DaemonContext) -> MagicMock:
    order_client = MagicMock()
    order_client.cancel = AsyncMock()
    daemon_context.order_client = order_client
    return order_client


async def test_noop_without_order_client(daemon_context: DaemonContext) -> None:
    summary = await run_orders_tick(daemon_context)
    assert summary.expired == 0 and summary.notified == 0


async def test_ttl_expiry_requests_cancel_tracker_confirms(
    daemon_context: DaemonContext,
) -> None:
    from sqlalchemy import update as sa_update

    order_client = _wire_exec(daemon_context)
    stale = _insert_order(
        daemon_context.engine, "submitted",
        submitted_ts=datetime.now(UTC) - timedelta(minutes=30),
    )
    fresh = _insert_order(
        daemon_context.engine, "submitted", submitted_ts=datetime.now(UTC),
    )

    async def confirm(ib_order_id: int) -> None:  # the tracker's job
        with daemon_context.engine.begin() as conn:
            conn.execute(sa_update(orders).where(orders.c.id == stale)
                         .values(status="cancelled", terminal_ts=datetime.now(UTC)))

    order_client.cancel = AsyncMock(side_effect=confirm)
    summary = await run_orders_tick(daemon_context)
    assert summary.expired == 1
    order_client.cancel.assert_awaited_once_with(11)
    record = get_order(daemon_context.engine, stale)
    assert record is not None
    assert record.status == "cancelled"  # tracker confirmed, not the sweep
    assert "TTL" in (record.last_error or "")
    assert get_order(daemon_context.engine, fresh).status == "submitted"  # type: ignore[union-attr]


async def test_ttl_unconfirmed_cancel_retries_next_tick(
    daemon_context: DaemonContext,
) -> None:
    # If the broker never confirms (no Cancelled event), the row stays WORKING
    # and the sweep simply requests the cancel again — it must never mark the
    # row terminal itself (a fill may still be racing the cancel).
    order_client = _wire_exec(daemon_context)
    stale = _insert_order(
        daemon_context.engine, "submitted",
        submitted_ts=datetime.now(UTC) - timedelta(minutes=30),
    )
    first = await run_orders_tick(daemon_context)
    second = await run_orders_tick(daemon_context)
    assert first.expired == 1 and second.expired == 1
    assert order_client.cancel.await_count == 2
    assert get_order(daemon_context.engine, stale).status == "submitted"  # type: ignore[union-attr]


async def test_ttl_registry_miss_warns_once_and_leaves_row(
    daemon_context: DaemonContext,
) -> None:
    order_client = _wire_exec(daemon_context)
    order_client.cancel = AsyncMock(side_effect=ValueError("unknown order id 11"))
    stale = _insert_order(
        daemon_context.engine, "submitted",
        submitted_ts=datetime.now(UTC) - timedelta(minutes=30),
    )
    first = await run_orders_tick(daemon_context)
    second = await run_orders_tick(daemon_context)
    assert first.expired == 0 and second.expired == 0
    # Row stays working (we could NOT cancel it at the broker — abandoning would lie).
    assert get_order(daemon_context.engine, stale).status == "submitted"  # type: ignore[union-attr]
    # Warned exactly once across both ticks.
    warn_texts = [
        c.args[0] for c in daemon_context.telegram.send_message.await_args_list
        if "restart" in c.args[0] or "manually" in c.args[0]
    ]
    assert len(warn_texts) == 1


async def test_notifies_new_terminals_once(daemon_context: DaemonContext) -> None:
    _wire_exec(daemon_context)
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    filled = _insert_order(
        daemon_context.engine, "filled", terminal_ts=datetime.now(UTC),
    )
    record_fill(
        daemon_context.engine, filled, exec_id="e1", side="SELL", price=1.2,
        qty=1, ts=NOW,
    )
    _insert_order(
        daemon_context.engine, "rejected", terminal_ts=datetime.now(UTC),
        last_error="insufficient margin",
    )
    # Old terminal (before the notify watermark) must NOT re-notify.
    _insert_order(
        daemon_context.engine, "cancelled", terminal_ts=NOW - timedelta(hours=2),
    )
    first = await run_orders_tick(daemon_context)
    second = await run_orders_tick(daemon_context)
    assert first.notified == 2
    assert second.notified == 0
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("filled" in m.lower() for m in sent)
    assert any("insufficient margin" in m for m in sent)
