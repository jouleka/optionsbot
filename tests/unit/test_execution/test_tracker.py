"""Tests for OrderTracker — OrderClient events → IBK-124 ledger (IBK-125)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Engine, insert, select, update

from optionsbot.execution.orders import get_order, record_fill
from optionsbot.execution.tracker import OrderTracker, map_ib_status, row_id_from_ref
from optionsbot.ibkr.types import CommissionUpdate, ExecutionFill, OrderStatusUpdate
from optionsbot.storage.schema import fills, orders

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)


def _insert_order(engine: Engine, status: str) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            insert(orders).values(
                intent="open", symbol="SPY", strategy="iron_condor",
                legs_json=[], quantity=1, status=status, staged_ts=NOW,
                reprice_count=0,
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


def _status(
    order_id: int, status: str, *, filled: float = 0.0, remaining: float = 1.0
) -> OrderStatusUpdate:
    return OrderStatusUpdate(
        ib_order_id=11, perm_id=4242, order_ref=f"obot-{order_id}",
        status=status, filled=filled, remaining=remaining, avg_fill_price=None,
    )


def _fill(order_id: int, exec_id: str = "e1", sec_type: str = "OPT") -> ExecutionFill:
    return ExecutionFill(
        ib_order_id=11, order_ref=f"obot-{order_id}", exec_id=exec_id,
        side="SELL", price=1.20, qty=1, ts=NOW, con_id=1580, sec_type=sec_type,
    )


# --- pure mapping --------------------------------------------------------------


@pytest.mark.parametrize(
    ("ib_status", "filled", "remaining", "expected"),
    [
        ("PendingSubmit", 0, 2, None),
        ("PendingCancel", 0, 2, None),
        ("ApiPending", 0, 2, None),
        ("ApiUpdate", 0, 2, None),
        ("ValidationError", 0, 2, None),
        ("PreSubmitted", 0, 2, "submitted"),
        ("Submitted", 0, 2, "submitted"),
        ("PreSubmitted", 1, 1, "partial"),
        ("Submitted", 1, 1, "partial"),
        ("Filled", 2, 0, "filled"),
        ("Cancelled", 0, 2, "cancelled"),
        ("ApiCancelled", 1, 1, "cancelled"),
        ("Inactive", 0, 2, "rejected"),
        ("SomethingNew", 0, 2, None),
    ],
)
def test_map_ib_status(
    ib_status: str, filled: float, remaining: float, expected: str | None
) -> None:
    assert map_ib_status(ib_status, filled, remaining) == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("obot-12", 12),
        ("obot-1", 1),
        ("obot-x", None),
        ("obot-", None),
        ("manual-3", None),
        ("", None),
        (None, None),
    ],
)
def test_row_id_from_ref(ref: str | None, expected: int | None) -> None:
    assert row_id_from_ref(ref) == expected


# --- ledger flows ---------------------------------------------------------------


def test_status_submitted_records_ids(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitting")
    OrderTracker(tmp_db).handle_status(_status(order_id, "Submitted"))
    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status == "submitted"
    assert record.ib_order_id == 11
    assert record.ib_perm_id == 4242


def test_status_partial_then_filled(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitting")
    tracker = OrderTracker(tmp_db)
    tracker.handle_status(_status(order_id, "Submitted"))
    tracker.handle_status(_status(order_id, "Submitted", filled=1, remaining=1))
    assert get_order(tmp_db, order_id).status == "partial"  # type: ignore[union-attr]
    tracker.handle_status(_status(order_id, "Filled", filled=2, remaining=0))
    assert get_order(tmp_db, order_id).status == "filled"  # type: ignore[union-attr]


def test_status_inactive_maps_to_rejected_with_error(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitting")
    tracker = OrderTracker(tmp_db)
    tracker.handle_status(_status(order_id, "Submitted"))
    tracker.handle_status(_status(order_id, "Inactive"))
    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status == "rejected"
    assert record.last_error  # some explanation recorded


def test_terminal_redelivery_never_raises(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "filled")
    tracker = OrderTracker(tmp_db)
    tracker.handle_status(_status(order_id, "Filled", filled=2, remaining=0))
    assert get_order(tmp_db, order_id).status == "filled"  # type: ignore[union-attr]


def test_unknown_and_foreign_refs_ignored(tmp_db: Engine) -> None:
    tracker = OrderTracker(tmp_db)
    tracker.handle_status(_status(999_999, "Submitted"))  # row doesn't exist
    update_obj = OrderStatusUpdate(
        ib_order_id=5, perm_id=None, order_ref="manual-trade", status="Submitted",
        filled=0, remaining=1, avg_fill_price=None,
    )
    tracker.handle_status(update_obj)  # not our ref — silently ignored
    none_ref = OrderStatusUpdate(
        ib_order_id=5, perm_id=None, order_ref=None, status="Submitted",
        filled=0, remaining=1, avg_fill_price=None,
    )
    tracker.handle_status(none_ref)


def test_fill_persists_and_skips_bag_rows(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    tracker = OrderTracker(tmp_db)
    tracker.handle_fill(_fill(order_id, exec_id="leg1", sec_type="OPT"))
    tracker.handle_fill(_fill(order_id, exec_id="bag1", sec_type="BAG"))  # skipped
    tracker.handle_fill(_fill(order_id, exec_id="leg1", sec_type="OPT"))  # dupe
    with tmp_db.connect() as conn:
        rows = conn.execute(select(fills)).fetchall()
    assert [r.ib_exec_id for r in rows] == ["leg1"]
    assert rows[0].leg_con_id == 1580


def test_live_fill_on_failed_terminal_row_trips_kill(tmp_db: Engine) -> None:
    from optionsbot.execution.state import load_state

    order_id = _insert_order(tmp_db, "abandoned")
    tracker = OrderTracker(tmp_db)
    tracker.handle_fill(_fill(order_id, exec_id="zombie1"))
    assert load_state(tmp_db).killed is True
    # Replay of the same execId must not re-trip anything (dedupe path).
    tracker.handle_fill(_fill(order_id, exec_id="zombie1"))


def test_commission_attaches_by_exec_id(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    record_fill(
        tmp_db, order_id, exec_id="leg1", side="SELL", price=1.2, qty=1, ts=NOW
    )
    tracker = OrderTracker(tmp_db)
    tracker.handle_commission(CommissionUpdate(exec_id="leg1", commission=0.66))
    tracker.handle_commission(CommissionUpdate(exec_id="ghost", commission=0.1))
    with tmp_db.connect() as conn:
        row = conn.execute(select(fills)).one()
    assert row.commission == pytest.approx(0.66)


def test_attach_subscribes_all_three(tmp_db: Engine) -> None:
    tracker = OrderTracker(tmp_db)
    order_client = MagicMock()
    tracker.attach(order_client)
    order_client.on_status.assert_called_once_with(tracker.handle_status)
    order_client.on_fill.assert_called_once_with(tracker.handle_fill)
    order_client.on_commission.assert_called_once_with(tracker.handle_commission)
