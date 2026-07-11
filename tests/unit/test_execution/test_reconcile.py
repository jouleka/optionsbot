"""Tests for broker reconciliation (IBK-128)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Engine, insert, select, update

from optionsbot.execution.orders import get_order, record_fill
from optionsbot.execution.reconcile import reconcile
from optionsbot.execution.state import load_state
from optionsbot.ibkr.types import ExecutionFill, PortfolioPosition
from optionsbot.storage.schema import fills, orders

NOW = datetime(2026, 6, 11, 16, 0, tzinfo=UTC)

LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1,
     "con_id": 1580, "multiplier": 100, "currency": "USD"},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1,
     "con_id": 1575, "multiplier": 100, "currency": "USD"},
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
    ref: str,
    exec_id: str,
    *,
    qty: int = 1,
    side: str = "SELL",
    con_id: int = 1580,
    commission: float | None = 0.65,
) -> ExecutionFill:
    return ExecutionFill(
        ib_order_id=11, order_ref=ref, exec_id=exec_id, side=side,
        price=1.20, qty=qty, ts=NOW, con_id=con_id, sec_type="OPT",
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
    # Exact conId, side, and per-leg ratio are required for every option leg.
    client = _client(executions=[
        _fill(ref, "x1", qty=1, side="SELL", con_id=1580),
        _fill(ref, "x2", qty=1, side="BUY", con_id=1575),
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


@pytest.mark.parametrize("bad_shape", ["duplicate_contract", "wrong_side", "overfill"])
async def test_fill_count_cannot_fake_exact_leg_completion(
    tmp_db: Engine,
    bad_shape: str,
) -> None:
    order_id = _insert_order(tmp_db, "submitted", quantity=1)
    ref = f"obot-{order_id}"
    if bad_shape == "duplicate_contract":
        executions = [
            _fill(ref, "bad-a", side="SELL", con_id=1580),
            _fill(ref, "bad-b", side="SELL", con_id=1580),
        ]
    elif bad_shape == "wrong_side":
        executions = [
            _fill(ref, "bad-a", side="BUY", con_id=1580),
            _fill(ref, "bad-b", side="SELL", con_id=1575),
        ]
    else:
        executions = [
            _fill(ref, "bad-a", qty=2, side="SELL", con_id=1580),
            _fill(ref, "bad-b", side="BUY", con_id=1575),
        ]

    await reconcile(tmp_db, _client(executions=executions), now=NOW)

    record = get_order(tmp_db, order_id)
    assert record is not None
    assert record.status != "filled"


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
    client = _client(executions=[
        _fill(ref, "x1", side="SELL", con_id=1580),
        _fill(ref, "x2", side="BUY", con_id=1575),
    ])
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


async def test_reconcile_resumes_persisted_walks(tmp_db: Engine, monkeypatch: Any) -> None:
    from optionsbot.execution.orders import upsert_walk_state

    order_id = _insert_order(tmp_db, "submitted")
    upsert_walk_state(
        tmp_db, order_id, ib_order_id=11, symbol="SPY", legs=LEGS,
        decision_mid=1.20, budget=0.09, increment=0.01, step=1,
        prev_target=1.17, ts=NOW,
    )
    client = _client(open_orders=[(11, f"obot-{order_id}", "Submitted")])
    resumed: list[int] = []

    async def fake_resume(**kwargs: Any) -> int:
        resumed.append(kwargs["walk_tasks"] is not None)
        return 1

    monkeypatch.setattr("optionsbot.execution.reconcile.resume_walks", fake_resume)
    notify, sent = _notify()
    summary = await reconcile(
        tmp_db, client, notify=notify, now=NOW,
        walk_resume=fake_resume, walk_md=MagicMock(), walk_tasks=set(),
    )
    assert summary.adopted == 1
    assert resumed == [True]  # resume_walks was invoked with walk_tasks


def _portfolio_pos(
    symbol: str, *, strike: float, right: str, position: float,
    expiry: str = "20260717",
) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU123", symbol=symbol, sec_type="OPT", expiry=expiry,
        strike=strike, right=right, multiplier=100, position=position,  # type: ignore[arg-type]
        avg_cost=120.0, market_price=1.2, market_value=120.0,
        unrealized_pnl=0.0, realized_pnl=0.0,
        con_id={580.0: 1580, 575.0: 1575}.get(strike, int(strike * 100)),
    )


async def test_broker_position_with_no_ledger_row_alerts_and_kills(tmp_db: Engine) -> None:
    # No open or filled ledger rows at all, but the broker holds an SPY put.
    snapshot = [_portfolio_pos("SPY", strike=580.0, right="P", position=-1.0)]

    async def positions_snapshot() -> list[PortfolioPosition]:
        return snapshot

    client = _client(open_orders=[], executions=[])
    notify, sent = _notify()
    summary = await reconcile(
        tmp_db, client, notify=notify, now=NOW,
        positions_snapshot=positions_snapshot,
    )
    assert summary.orphan_positions == 1
    assert load_state(tmp_db).killed
    assert any("KILL SWITCH" in m and "position" in m.lower() for m in sent)


async def test_broker_position_matching_a_filled_ledger_order_is_fine(tmp_db: Engine) -> None:
    # A filled open-intent order in the ledger that matches the broker position:
    # no orphan, no kill.
    order_id = _insert_order(tmp_db, "submitted")
    from optionsbot.execution.orders import transition
    transition(tmp_db, order_id, "filled", now=NOW)
    record_fill(
        tmp_db, order_id, exec_id="position-short", side="SELL",
        price=1.60, qty=1, ts=NOW, leg_con_id=1580,
    )
    record_fill(
        tmp_db, order_id, exec_id="position-long", side="BUY",
        price=0.40, qty=1, ts=NOW, leg_con_id=1575,
    )
    snapshot = [
        _portfolio_pos("SPY", strike=580.0, right="P", position=-1.0),
        _portfolio_pos("SPY", strike=575.0, right="P", position=1.0),
    ]

    async def positions_snapshot() -> list[PortfolioPosition]:
        return snapshot

    client = _client(open_orders=[], executions=[])
    notify, sent = _notify()
    summary = await reconcile(
        tmp_db, client, notify=notify, now=NOW,
        positions_snapshot=positions_snapshot,
    )
    assert summary.orphan_positions == 0
    assert not load_state(tmp_db).killed
    assert not any("KILL SWITCH" in m for m in sent)


@pytest.mark.parametrize(
    "snapshot",
    [
        [
            _portfolio_pos("SPY", strike=580.0, right="P", position=-2.0),
            _portfolio_pos("SPY", strike=575.0, right="P", position=1.0),
        ],
        [],
    ],
    ids=["broker-quantity-mismatch", "ledger-position-missing-at-broker"],
)
async def test_exact_position_mismatch_kills(
    tmp_db: Engine,
    snapshot: list[PortfolioPosition],
) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    from optionsbot.execution.orders import transition

    transition(tmp_db, order_id, "filled", now=NOW)
    record_fill(
        tmp_db, order_id, exec_id="mismatch-short", side="SELL",
        price=1.60, qty=1, ts=NOW, leg_con_id=1580,
    )
    record_fill(
        tmp_db, order_id, exec_id="mismatch-long", side="BUY",
        price=0.40, qty=1, ts=NOW, leg_con_id=1575,
    )

    async def positions_snapshot() -> list[PortfolioPosition]:
        return snapshot

    notify, sent = _notify()
    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        notify=notify,
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.orphan_positions >= 1
    assert load_state(tmp_db).killed
    assert any("KILL SWITCH" in message for message in sent)


async def test_position_snapshot_failure_does_not_crash_reconcile(tmp_db: Engine) -> None:
    async def positions_snapshot() -> list[PortfolioPosition]:
        raise RuntimeError("gateway down")

    client = _client(open_orders=[], executions=[])
    notify, sent = _notify()
    summary = await reconcile(
        tmp_db, client, notify=notify, now=NOW,
        positions_snapshot=positions_snapshot,
    )
    # Broker/account state is uncertain, so reconciliation must fail closed.
    assert summary.orphan_positions == 0
    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    assert any("KILL SWITCH" in message for message in sent)
