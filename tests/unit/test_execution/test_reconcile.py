"""Tests for broker reconciliation (IBK-128)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Engine, insert, select, update

from optionsbot.execution.orders import get_order, record_fill
from optionsbot.execution.reconcile import reconcile
from optionsbot.execution.state import load_state, trip_kill
from optionsbot.ibkr.types import ExecutionFill, OpenOrderSnapshot, PortfolioPosition
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
    ib_order_id: int | None = 11, limit_price: float = -1.0,
) -> int:
    with engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=LEGS, quantity=quantity, status=status,
            limit_price=limit_price,
            staged_ts=staged_ts or OLD, submitted_ts=submitted_ts,
            ib_order_id=ib_order_id, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(order_ref=f"obot-{order_id}"))
    return order_id


def _broker_order(
    order_id: int,
    *,
    ib_order_id: int = 11,
    status: str = "Submitted",
    con_ids: tuple[int, int] = (1580, 1575),
    quantity: int = 1,
    limit_price: float = -1.0,
) -> OpenOrderSnapshot:
    return OpenOrderSnapshot(
        ib_order_id=ib_order_id,
        order_ref=f"obot-{order_id}",
        status=status,
        sec_type="BAG",
        symbol="SPY",
        currency="USD",
        exchange="SMART",
        combo_legs=(
            (con_ids[0], 1, "SELL", "SMART"),
            (con_ids[1], 1, "BUY", "SMART"),
        ),
        order_action="BUY",
        total_quantity=quantity,
        order_type="LMT",
        tif="DAY",
        limit_price=limit_price,
    )


def _client(
    open_orders: list[Any] | None = None,
    executions: list[ExecutionFill] | None = None,
) -> MagicMock:
    normalized: list[Any] = []
    for row in open_orders or []:
        if isinstance(row, OpenOrderSnapshot):
            normalized.append(row)
            continue
        if isinstance(row, tuple) and len(row) == 3:
            ib_order_id, ref, status = row
            if isinstance(ref, str) and ref.startswith("obot-") and ref[5:].isdigit():
                normalized.append(
                    _broker_order(
                        int(ref[5:]),
                        ib_order_id=ib_order_id,
                        status=status,
                    )
                )
            else:
                normalized.append(OpenOrderSnapshot(ib_order_id, ref, status))
            continue
        normalized.append(row)
    client = MagicMock()
    client.adopt_open_orders = AsyncMock(return_value=normalized)
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


async def test_exact_broker_terms_matching_ledger_are_accepted(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    client = _client(open_orders=[_broker_order(order_id)])
    summary = await reconcile(tmp_db, client, now=NOW)

    assert summary.mismatches == 0
    assert not load_state(tmp_db).killed
    client.authorize_adoptions.assert_called_once_with((_broker_order(order_id),))


async def test_broker_normalized_combo_leg_order_is_accepted(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    exact = _broker_order(order_id)
    normalized = replace(exact, combo_legs=tuple(reversed(exact.combo_legs)))
    client = _client(open_orders=[normalized])

    summary = await reconcile(tmp_db, client, now=NOW)

    assert summary.mismatches == 0
    assert not load_state(tmp_db).killed
    client.authorize_adoptions.assert_called_once_with((normalized,))


@pytest.mark.parametrize(
    "snapshot",
    [
        replace(_broker_order(1), total_quantity=True),  # type: ignore[arg-type]
        replace(_broker_order(1), ib_order_id=0),
        replace(
            _broker_order(1),
            combo_legs=((1580, True, "SELL", "SMART"), (1575, 1, "BUY", "SMART")),  # type: ignore[arg-type]
        ),
        replace(_broker_order(1), order_ref="obot-01"),
    ],
)
async def test_malformed_or_noncanonical_snapshot_never_authorizes(
    tmp_db: Engine, snapshot: OpenOrderSnapshot
) -> None:
    _insert_order(tmp_db, "submitted")
    client = _client(open_orders=[snapshot])

    summary = await reconcile(tmp_db, client, now=NOW)

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_not_called()


async def test_later_snapshot_failure_prevents_authorization_and_walk_resume(
    tmp_db: Engine,
) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    client = _client(open_orders=[_broker_order(order_id)])
    client.recent_executions.side_effect = RuntimeError("snapshot unavailable")
    resume = AsyncMock(return_value=1)

    summary = await reconcile(
        tmp_db,
        client,
        now=NOW,
        walk_resume=resume,
        walk_md=MagicMock(),
        walk_tasks=set(),
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_not_called()
    resume.assert_not_awaited()


async def test_broker_contract_mismatch_halts_before_walk_resume(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    resume = AsyncMock(return_value=1)
    client = _client(
        open_orders=[_broker_order(order_id, con_ids=(9991, 9992))],
    )

    summary = await reconcile(
        tmp_db,
        client,
        now=NOW,
        walk_resume=resume,
        walk_md=MagicMock(),
        walk_tasks=set(),
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    resume.assert_not_awaited()
    client.authorize_adoptions.assert_not_called()


async def test_single_leg_ratio_requires_total_contract_quantity(tmp_db: Engine) -> None:
    single_leg = [dict(LEGS[0], quantity=2)]
    order_id = _insert_order(tmp_db, "submitted", quantity=3)
    with tmp_db.begin() as conn:
        conn.execute(
            update(orders).where(orders.c.id == order_id).values(legs_json=single_leg)
        )
    exact = OpenOrderSnapshot(
        ib_order_id=11, order_ref=f"obot-{order_id}", status="Submitted",
        sec_type="OPT", symbol="SPY", currency="USD", exchange="SMART",
        contract_con_id=1580, multiplier=100, expiry="20260717", strike=580.0,
        right="P", order_action="SELL", total_quantity=6, order_type="LMT",
        tif="DAY", limit_price=1.0,
    )
    client = _client(open_orders=[exact])

    summary = await reconcile(tmp_db, client, now=NOW)

    assert summary.mismatches == 0
    client.authorize_adoptions.assert_called_once_with((exact,))


async def test_missing_broker_transition_race_blocks_authorization(
    tmp_db: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from optionsbot.execution.orders import IllegalOrderTransition

    _insert_order(tmp_db, "submitted")

    def race(*args: Any, **kwargs: Any) -> None:
        raise IllegalOrderTransition("concurrent state change")

    monkeypatch.setattr("optionsbot.execution.reconcile.transition", race)
    client = _client(open_orders=[])
    summary = await reconcile(tmp_db, client, now=NOW)

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_not_called()


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
    record_fill(
        tmp_db,
        order_id,
        exec_id="x1",
        side="SELL",
        price=1.2,
        qty=1,
        ts=NOW,
        leg_con_id=1580,
    )
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


async def test_walk_resume_failure_revokes_new_authority(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    client = _client(open_orders=[_broker_order(order_id)])
    resume = AsyncMock(side_effect=RuntimeError("resume failed"))

    summary = await reconcile(
        tmp_db, client, now=NOW,
        walk_resume=resume, walk_md=MagicMock(), walk_tasks=set(),
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_called_once()
    client.revoke_adoptions.assert_called_once()


async def test_cancelled_walk_resume_revokes_new_authority(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    client = _client(open_orders=[_broker_order(order_id)])
    resume_started = asyncio.Event()
    keep_resuming = asyncio.Event()

    async def suspended_resume(**kwargs: Any) -> int:
        resume_started.set()
        await keep_resuming.wait()
        return 1

    task = asyncio.create_task(
        reconcile(
            tmp_db,
            client,
            now=NOW,
            walk_resume=suspended_resume,
            walk_md=MagicMock(),
            walk_tasks=set(),
        )
    )
    await resume_started.wait()
    client.authorize_adoptions.assert_called_once()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    client.revoke_adoptions.assert_called_once()


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


async def test_expired_broker_position_is_left_to_settlement_sweep(
    tmp_db: Engine,
) -> None:
    expired = _portfolio_pos(
        "SPY",
        strike=580.0,
        right="P",
        position=-1.0,
        expiry="20260610",
    )

    async def positions_snapshot() -> list[PortfolioPosition]:
        return [expired]

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 0
    assert summary.orphan_positions == 0
    assert not load_state(tmp_db).killed


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
    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    assert any("KILL SWITCH" in message for message in sent)


async def test_fractional_ledger_fill_quantity_halts(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    from optionsbot.execution.orders import transition

    transition(tmp_db, order_id, "filled", now=NOW)
    record_fill(
        tmp_db,
        order_id,
        exec_id="fractional-short",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=1580,
    )
    with tmp_db.begin() as conn:
        conn.execute(
            update(fills)
            .where(fills.c.ib_exec_id == "fractional-short")
            .values(qty=1.5)
        )
    record_fill(
        tmp_db,
        order_id,
        exec_id="fractional-long",
        side="BUY",
        price=0.40,
        qty=1,
        ts=NOW,
        leg_con_id=1575,
    )

    async def positions_snapshot() -> list[PortfolioPosition]:
        return [
            _portfolio_pos("SPY", strike=580.0, right="P", position=-1.0),
            _portfolio_pos("SPY", strike=575.0, right="P", position=1.0),
        ]

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


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


async def test_open_order_snapshot_failure_halts_reconciliation(tmp_db: Engine) -> None:
    client = _client(open_orders=[], executions=[])
    client.adopt_open_orders.side_effect = RuntimeError("open-order snapshot unavailable")
    notify, sent = _notify()

    summary = await reconcile(tmp_db, client, notify=notify, now=NOW)

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    assert any("KILL SWITCH" in message for message in sent)


async def test_clean_full_reconcile_rearms_transient_gateway_disconnect(
    tmp_db: Engine,
) -> None:
    trip_kill(
        tmp_db,
        "reconcile open-order snapshot unavailable: Socket disconnect",
        now=NOW,
    )
    client = _client(open_orders=[], executions=[])
    notify, sent = _notify()

    async def positions_snapshot() -> list[PortfolioPosition]:
        return []

    summary = await reconcile(
        tmp_db,
        client,
        notify=notify,
        now=NOW + timedelta(minutes=1),
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 0
    assert not load_state(tmp_db).killed
    client.authorize_adoptions.assert_called_once_with(())
    assert any("re-armed" in message for message in sent)


async def test_transient_disconnect_kill_needs_full_position_proof(
    tmp_db: Engine,
) -> None:
    trip_kill(
        tmp_db,
        "reconcile open-order snapshot unavailable: Socket disconnect",
        now=NOW,
    )
    client = _client(open_orders=[], executions=[])

    summary = await reconcile(
        tmp_db,
        client,
        now=NOW + timedelta(minutes=1),
    )

    assert summary.mismatches == 0
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_not_called()


async def test_clean_full_reconcile_rearms_resolved_mutation_uncertainty(
    tmp_db: Engine,
) -> None:
    trip_kill(
        tmp_db,
        "cancel request outcome unknown for order #383: broker mutation authority drifted",
        now=NOW,
    )
    client = _client(open_orders=[], executions=[])
    notify, sent = _notify()

    async def positions_snapshot() -> list[PortfolioPosition]:
        return []

    summary = await reconcile(
        tmp_db,
        client,
        notify=notify,
        now=NOW + timedelta(minutes=1),
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 0
    assert not load_state(tmp_db).killed
    client.authorize_adoptions.assert_called_once_with(())
    assert any("re-armed" in message for message in sent)


async def test_mutation_uncertainty_never_rearms_with_working_order(
    tmp_db: Engine,
) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    trip_kill(
        tmp_db,
        f"cancel request outcome unknown for order #{order_id}: timeout",
        now=NOW,
    )
    client = _client(open_orders=[_broker_order(order_id)], executions=[])

    async def positions_snapshot() -> list[PortfolioPosition]:
        return []

    summary = await reconcile(
        tmp_db,
        client,
        now=NOW + timedelta(minutes=1),
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 0
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_not_called()


async def test_clean_reconcile_never_clears_an_unrelated_kill_reason(
    tmp_db: Engine,
) -> None:
    trip_kill(tmp_db, "manual emergency halt", now=NOW)
    client = _client(open_orders=[], executions=[])

    async def positions_snapshot() -> list[PortfolioPosition]:
        return []

    summary = await reconcile(
        tmp_db,
        client,
        now=NOW + timedelta(minutes=1),
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 0
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_not_called()


async def test_none_position_snapshot_halts_reconciliation(tmp_db: Engine) -> None:
    async def positions_snapshot() -> list[PortfolioPosition]:
        return None  # type: ignore[return-value]

    notify, sent = _notify()
    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        notify=notify,
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    assert any("KILL SWITCH" in message for message in sent)


async def test_malformed_open_order_row_halts_reconciliation(tmp_db: Engine) -> None:
    client = _client(open_orders=[], executions=[])
    client.adopt_open_orders.return_value = [(11,)]

    summary = await reconcile(tmp_db, client, now=NOW)

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_execution_snapshot_failure_halts_reconciliation(tmp_db: Engine) -> None:
    client = _client(open_orders=[], executions=[])
    client.recent_executions.side_effect = RuntimeError("execution snapshot unavailable")

    summary = await reconcile(tmp_db, client, now=NOW)

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_malformed_position_row_halts_reconciliation(tmp_db: Engine) -> None:
    async def positions_snapshot() -> list[PortfolioPosition]:
        return [object()]  # type: ignore[list-item]

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


@pytest.mark.parametrize("fractional_source", ["broker", "ledger"])
async def test_fractional_contract_identity_halts(
    tmp_db: Engine,
    fractional_source: str,
) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    from optionsbot.execution.orders import transition

    if fractional_source == "ledger":
        fractional_legs = [dict(leg) for leg in LEGS]
        fractional_legs[0]["con_id"] = 1580.5
        with tmp_db.begin() as conn:
            conn.execute(
                update(orders)
                .where(orders.c.id == order_id)
                .values(legs_json=fractional_legs)
            )
    transition(tmp_db, order_id, "filled", now=NOW)
    record_fill(
        tmp_db,
        order_id,
        exec_id="identity-short",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=1580,
    )
    record_fill(
        tmp_db,
        order_id,
        exec_id="identity-long",
        side="BUY",
        price=0.40,
        qty=1,
        ts=NOW,
        leg_con_id=1575,
    )
    short = _portfolio_pos("SPY", strike=580.0, right="P", position=-1.0)
    if fractional_source == "broker":
        short = replace(short, con_id=1580.5)  # type: ignore[arg-type]

    async def positions_snapshot() -> list[PortfolioPosition]:
        return [
            short,
            _portfolio_pos("SPY", strike=575.0, right="P", position=1.0),
        ]

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


@pytest.mark.parametrize(
    ("field", "malformed", "broker_symbol", "broker_expiry"),
    [
        ("symbol", 123, "123", "20260717"),
        ("expiry", 20260717, "SPY", "20260717"),
    ],
)
async def test_coercible_malformed_persisted_position_leg_halts(
    tmp_db: Engine,
    field: str,
    malformed: object,
    broker_symbol: str,
    broker_expiry: str,
) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    malformed_legs = [dict(leg) for leg in LEGS]
    malformed_legs[0][field] = malformed
    with tmp_db.begin() as conn:
        conn.execute(
            update(orders)
            .where(orders.c.id == order_id)
            .values(legs_json=malformed_legs)
        )
    from optionsbot.execution.orders import transition

    transition(tmp_db, order_id, "filled", now=NOW)
    record_fill(
        tmp_db,
        order_id,
        exec_id="malformed-short",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=1580,
    )
    record_fill(
        tmp_db,
        order_id,
        exec_id="malformed-long",
        side="BUY",
        price=0.40,
        qty=1,
        ts=NOW,
        leg_con_id=1575,
    )

    async def positions_snapshot() -> list[PortfolioPosition]:
        return [
            _portfolio_pos(
                broker_symbol,
                strike=580.0,
                right="P",
                position=-1.0,
                expiry=broker_expiry,
            ),
            _portfolio_pos("SPY", strike=575.0, right="P", position=1.0),
        ]

    client = _client(open_orders=[], executions=[])
    summary = await reconcile(
        tmp_db,
        client,
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    client.authorize_adoptions.assert_not_called()


async def test_bot_owned_broker_order_without_ledger_row_halts(tmp_db: Engine) -> None:
    summary = await reconcile(
        tmp_db,
        _client(open_orders=[(11, "obot-999", "Submitted")], executions=[]),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_unknown_bot_owned_broker_status_halts(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    summary = await reconcile(
        tmp_db,
        _client(open_orders=[(11, f"obot-{order_id}", "MysteryWorking")], executions=[]),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_bot_execution_without_ledger_row_halts(tmp_db: Engine) -> None:
    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[_fill("obot-999", "missing-ledger")]),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_unreferenced_expiration_execution_is_ignored(
    tmp_db: Engine,
) -> None:
    expiration = ExecutionFill(
        ib_order_id=0,
        order_ref=None,
        exec_id="broker-expiration",
        side="BUY",
        price=0.0,
        qty=1,
        ts=NOW,
        con_id=1580,
        sec_type="OPT",
        commission=None,
    )

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[expiration]),
        now=NOW,
    )

    assert summary.mismatches == 0
    assert not load_state(tmp_db).killed


async def test_malformed_bag_execution_halts(tmp_db: Engine) -> None:
    malformed = ExecutionFill(
        ib_order_id=11,
        order_ref=None,  # type: ignore[arg-type]
        exec_id="",
        side="SELL",
        price=1.20,
        qty=1.5,  # type: ignore[arg-type]
        ts=NOW,
        con_id=1.5,  # type: ignore[arg-type]
        sec_type="BAG",
        commission=0.65,
    )
    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[malformed]),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_walk_resume_failure_halts_reconciliation(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")

    async def failing_resume(**kwargs: Any) -> int:
        raise RuntimeError("cannot restore persisted walk")

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[(11, f"obot-{order_id}", "Submitted")], executions=[]),
        now=NOW,
        walk_resume=failing_resume,
        walk_md=MagicMock(),
        walk_tasks=set(),
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_duplicate_live_broker_orders_for_one_ledger_ref_halt(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    ref = f"obot-{order_id}"
    summary = await reconcile(
        tmp_db,
        _client(
            open_orders=[(11, ref, "Submitted"), (22, ref, "Submitted")],
            executions=[],
        ),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_filled_multileg_order_requires_all_expected_leg_fills(
    tmp_db: Engine,
) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    from optionsbot.execution.orders import transition

    transition(tmp_db, order_id, "filled", now=NOW)
    record_fill(
        tmp_db,
        order_id,
        exec_id="only-short-leg",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=1580,
    )

    async def positions_snapshot() -> list[PortfolioPosition]:
        return [_portfolio_pos("SPY", strike=580.0, right="P", position=-1.0)]

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_unknown_execution_security_type_halts(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    malformed = replace(
        _fill(f"obot-{order_id}", "wrong-sec-type"),
        sec_type="bag",
    )

    summary = await reconcile(
        tmp_db,
        _client(
            open_orders=[(11, f"obot-{order_id}", "Submitted")],
            executions=[malformed],
        ),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_one_broker_order_id_cannot_claim_two_ledger_rows(tmp_db: Engine) -> None:
    first = _insert_order(tmp_db, "submitted", ib_order_id=11)
    second = _insert_order(tmp_db, "submitted", ib_order_id=22)
    summary = await reconcile(
        tmp_db,
        _client(
            open_orders=[
                (11, f"obot-{first}", "Submitted"),
                (11, f"obot-{second}", "Submitted"),
            ],
            executions=[],
        ),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_conflicting_duplicate_execution_identity_halts(tmp_db: Engine) -> None:
    first = _insert_order(tmp_db, "submitted", ib_order_id=11)
    second = _insert_order(tmp_db, "submitted", ib_order_id=22)
    record_fill(
        tmp_db,
        first,
        exec_id="same-provider-exec-id",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=1580,
    )
    conflicting = replace(
        _fill(f"obot-{second}", "same-provider-exec-id"),
        side="SELL",
        price=1.60,
        qty=1,
        con_id=1580,
    )

    summary = await reconcile(
        tmp_db,
        _client(
            open_orders=[
                (11, f"obot-{first}", "Submitted"),
                (22, f"obot-{second}", "Submitted"),
            ],
            executions=[conflicting],
        ),
        now=NOW,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_mixed_case_broker_position_security_type_halts(tmp_db: Engine) -> None:
    async def positions_snapshot() -> list[PortfolioPosition]:
        return [
            replace(
                _portfolio_pos("SPY", strike=580.0, right="P", position=-1.0),
                sec_type="opt",
            )
        ]

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_mixed_case_persisted_leg_security_type_halts(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    with tmp_db.begin() as conn:
        row = conn.execute(select(orders.c.legs_json).where(orders.c.id == order_id)).one()
        legs = [dict(leg) for leg in row.legs_json]
        legs[1]["sec_type"] = "opt"
        conn.execute(
            update(orders).where(orders.c.id == order_id).values(legs_json=legs)
        )
    from optionsbot.execution.orders import transition

    record_fill(
        tmp_db,
        order_id,
        exec_id="only-canonical-leg",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=1580,
    )
    transition(tmp_db, order_id, "filled", now=NOW)

    async def positions_snapshot() -> list[PortfolioPosition]:
        return [_portfolio_pos("SPY", strike=580.0, right="P", position=-1.0)]

    summary = await reconcile(
        tmp_db,
        _client(open_orders=[], executions=[]),
        now=NOW,
        positions_snapshot=positions_snapshot,
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed


async def test_broker_identity_mismatch_blocks_walk_resume(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted", ib_order_id=11)
    resume = AsyncMock(return_value=1)

    summary = await reconcile(
        tmp_db,
        _client(
            open_orders=[(22, f"obot-{order_id}", "Submitted")],
            executions=[],
        ),
        now=NOW,
        walk_resume=resume,
        walk_md=MagicMock(),
        walk_tasks=set(),
    )

    assert summary.mismatches == 1
    assert load_state(tmp_db).killed
    resume.assert_not_awaited()


async def test_malformed_persisted_legs_block_walk_resume(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted", ib_order_id=11)
    with tmp_db.begin() as conn:
        row = conn.execute(
            select(orders.c.legs_json).where(orders.c.id == order_id)
        ).one()
        legs = [dict(leg) for leg in row.legs_json]
        legs[1]["sec_type"] = "opt"
        conn.execute(
            update(orders).where(orders.c.id == order_id).values(legs_json=legs)
        )
    resume = AsyncMock(return_value=1)

    summary = await reconcile(
        tmp_db,
        _client(
            open_orders=[(11, f"obot-{order_id}", "Submitted")],
            executions=[],
        ),
        now=NOW,
        walk_resume=resume,
        walk_md=MagicMock(),
        walk_tasks=set(),
    )

    assert summary.mismatches == 1
    state = load_state(tmp_db)
    assert state.killed
    assert state.reason is not None and "invalid persisted semantics" in state.reason
    resume.assert_not_awaited()
