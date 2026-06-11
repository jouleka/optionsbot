"""Tests for broker reconciliation (IBK-128)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import Engine, insert, select, update

from optionsbot.execution.orders import get_order, record_fill
from optionsbot.execution.reconcile import reconcile
from optionsbot.execution.state import load_state
from optionsbot.ibkr.types import ExecutionFill
from optionsbot.storage.schema import fills, orders

NOW = datetime(2026, 6, 11, 16, 0, tzinfo=UTC)

LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
]


OLD = NOW - timedelta(minutes=5)  # past the in-flight grace window


def _insert_order(
    engine: Engine, status: str, *, quantity: int = 1,
    staged_ts: datetime | None = None, submitted_ts: datetime | None = OLD,
    ib_order_id: int | None = 11,
) -> int:
    with engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=LEGS, quantity=quantity, status=status,
            staged_ts=staged_ts or OLD, submitted_ts=submitted_ts,
            ib_order_id=ib_order_id, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(order_ref=f"obot-{order_id}"))
    return order_id


def _client(
    open_orders: list[tuple[int, str | None, str]] | None = None,
    executions: list[ExecutionFill] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.adopt_open_orders = AsyncMock(return_value=open_orders or [])
    client.recent_executions = AsyncMock(return_value=executions or [])
    return client


def _fill(
    ref: str, exec_id: str, *, qty: int = 1, commission: float | None = 0.65,
) -> ExecutionFill:
    return ExecutionFill(
        ib_order_id=11, order_ref=ref, exec_id=exec_id, side="SELL",
        price=1.20, qty=qty, ts=NOW, con_id=1580, sec_type="OPT",
        commission=commission,
    )


def _notify() -> tuple[AsyncMock, list[str]]:
    sent: list[str] = []

    async def send(text: str) -> None:
        sent.append(text)

    return AsyncMock(side_effect=send), sent


async def test_working_row_still_at_broker_stays_working(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    client = _client(open_orders=[(11, f"obot-{order_id}", "Submitted")])
    notify, sent = _notify()
    summary = await reconcile(tmp_db, client, notify=notify, now=NOW)
    assert summary.adopted == 1
    assert get_order(tmp_db, order_id).status == "submitted"  # type: ignore[union-attr]
    assert not sent


async def test_submitting_row_at_broker_becomes_submitted(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitting")
    client = _client(open_orders=[(11, f"obot-{order_id}", "PreSubmitted")])
    notify, _ = _notify()
    await reconcile(tmp_db, client, notify=notify, now=NOW)
    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status == "submitted"
    assert record.ib_order_id == 11


async def test_submitting_row_with_no_broker_record_is_skipped(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitting", ib_order_id=None)
    notify, _ = _notify()
    await reconcile(tmp_db, _client(), notify=notify, now=NOW)
    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status == "skipped"
    assert "never resubmitted" in (record.last_error or "")


async def test_working_row_missing_at_broker_without_fills_is_cancelled(
    tmp_db: Engine,
) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    notify, _ = _notify()
    await reconcile(tmp_db, _client(), notify=notify, now=NOW)
    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status == "cancelled"
    assert "not at broker" in (record.last_error or "")


async def test_working_row_with_complete_fills_becomes_filled(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted", quantity=1)
    ref = f"obot-{order_id}"
    # 2 option legs x quantity 1 -> complete when total fill qty >= 2.
    client = _client(executions=[
        _fill(ref, "x1", qty=1), _fill(ref, "x2", qty=1),
    ])
    notify, _ = _notify()
    summary = await reconcile(tmp_db, client, notify=notify, now=NOW)
    assert summary.fills_replayed == 2
    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status == "filled"
    with tmp_db.connect() as conn:
        rows = conn.execute(select(fills)).fetchall()
    assert {r.ib_exec_id for r in rows} == {"x1", "x2"}
    assert all(r.commission == 0.65 for r in rows)


async def test_foreign_broker_order_warns_and_is_left_alone(tmp_db: Engine) -> None:
    client = _client(open_orders=[(0, None, "Submitted"), (42, "manual-x", "Submitted")])
    notify, sent = _notify()
    summary = await reconcile(tmp_db, client, notify=notify, now=NOW)
    assert summary.foreign == 2
    assert sent and "not placed by the bot" in sent[0]


async def test_fill_for_failed_terminal_row_trips_kill_switch(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    ref = f"obot-{order_id}"
    with tmp_db.begin() as conn:  # simulate an earlier abandoned outcome
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(status="abandoned", terminal_ts=NOW))
    client = _client(executions=[_fill(ref, "k1")])
    notify, sent = _notify()
    summary = await reconcile(tmp_db, client, notify=notify, now=NOW)
    assert summary.mismatches == 1
    assert load_state(tmp_db).killed is True
    assert any("kill" in m.lower() for m in sent)


async def test_fresh_working_row_not_at_broker_is_left_alone(tmp_db: Engine) -> None:
    # Opus C1: an /execute can be suspended between transition(submitting/
    # submitted) and the broker snapshot — a row inside the grace window must
    # NOT be resolved as failed.
    submitted = _insert_order(tmp_db, "submitted", submitted_ts=NOW)
    submitting = _insert_order(
        tmp_db, "submitting", staged_ts=NOW, submitted_ts=None, ib_order_id=None,
    )
    notify, _ = _notify()
    summary = await reconcile(tmp_db, _client(), notify=notify, now=NOW)
    assert summary.resolved == 0
    assert get_order(tmp_db, submitted).status == "submitted"  # type: ignore[union-attr]
    assert get_order(tmp_db, submitting).status == "submitting"  # type: ignore[union-attr]


async def test_partial_at_broker_is_not_downgraded(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "partial")
    client = _client(open_orders=[(11, f"obot-{order_id}", "Submitted")])
    notify, _ = _notify()
    summary = await reconcile(tmp_db, client, notify=notify, now=NOW)
    assert summary.resolved == 0
    assert get_order(tmp_db, order_id).status == "partial"  # type: ignore[union-attr]


async def test_stale_staged_row_is_skipped(tmp_db: Engine) -> None:
    order_id = _insert_order(
        tmp_db, "staged", staged_ts=NOW - timedelta(hours=3), ib_order_id=None,
    )
    notify, _ = _notify()
    await reconcile(tmp_db, _client(), notify=notify, now=NOW)
    assert get_order(tmp_db, order_id).status == "skipped"  # type: ignore[union-attr]


async def test_reconcile_is_idempotent(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted", quantity=1)
    ref = f"obot-{order_id}"
    client = _client(executions=[_fill(ref, "x1"), _fill(ref, "x2")])
    notify, sent = _notify()
    first = await reconcile(tmp_db, client, notify=notify, now=NOW)
    second = await reconcile(tmp_db, client, notify=notify, now=NOW)
    assert first.fills_replayed == 2
    assert second.fills_replayed == 0  # execId dedupe
    assert second.mismatches == 0  # filled row + same fills is NOT a mismatch
    assert get_order(tmp_db, order_id).status == "filled"  # type: ignore[union-attr]


async def test_duplicate_fill_on_already_filled_row_no_kill(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted", quantity=1)
    ref = f"obot-{order_id}"
    record_fill(tmp_db, order_id, exec_id="x1", side="SELL", price=1.2, qty=1, ts=NOW)
    client = _client(executions=[_fill(ref, "x1")])
    notify, _ = _notify()
    summary = await reconcile(tmp_db, client, notify=notify, now=NOW)
    assert summary.mismatches == 0
    assert load_state(tmp_db).killed is False
