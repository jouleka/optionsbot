"""Tests for the order ledger + state machine (IBK-124)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, insert, select

from optionsbot.execution.orders import (
    LEGAL_TRANSITIONS,
    ORDER_STATUSES,
    TERMINAL_STATUSES,
    IllegalOrderTransition,
    bump_reprice,
    get_order,
    net_premium,
    open_orders,
    record_fill,
    set_fill_commission,
    stage_order,
    transition,
    working_orders,
)
from optionsbot.storage.schema import fills, orders, snapshots, strategy_scores

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)

CONDOR_LEGS = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 620.0, "right": "C", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 625.0, "right": "C", "quantity": 1},
]


def _insert_score(
    engine: Engine, *, suggested_quantity: int = 2, symbol: str = "SPY"
) -> int:
    """Persist a realistic snapshot + strategy_scores row, return the score id."""
    with engine.begin() as conn:
        snapshot_id = conn.execute(
            insert(snapshots).values(symbol=symbol, ts=NOW, spot=600.0)
        ).inserted_primary_key[0]
        score_id = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snapshot_id,
                strategy="iron_condor",
                score=78.0,
                rationale="test",
                legs_json=CONDOR_LEGS,
                suggestion_json={
                    "defined_risk": True,
                    "credit_or_debit": 155.0,
                    "max_loss": 345.0,
                    "max_profit": 155.0,
                    "prob_profit": 0.71,
                    "suggested_quantity": suggested_quantity,
                    "reward_risk": 0.45,
                    "expected_value": 21.0,
                    "risk_tier": "balanced",
                },
            )
        ).inserted_primary_key[0]
    return int(score_id)


def _insert_order(engine: Engine, status: str, **overrides: object) -> int:
    values: dict[str, object] = {
        "intent": "open",
        "symbol": "SPY",
        "strategy": "iron_condor",
        "legs_json": CONDOR_LEGS,
        "quantity": 1,
        "status": status,
        "staged_ts": NOW,
        "reprice_count": 0,
    }
    values.update(overrides)
    with engine.begin() as conn:
        order_id = conn.execute(insert(orders).values(**values)).inserted_primary_key[0]
    return int(order_id)


# --- staging -----------------------------------------------------------------


def test_stage_order_from_score_round_trips(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db)
    record = stage_order(tmp_db, score_id, now=NOW)
    assert record.status == "staged"
    assert record.intent == "open"
    assert record.symbol == "SPY"
    assert record.strategy == "iron_condor"
    assert record.quantity == 2  # suggestion_json.suggested_quantity
    assert record.legs == CONDOR_LEGS
    assert record.strategy_score_id == score_id
    assert record.order_ref == f"obot-{record.id}"
    assert record.staged_ts is not None and record.staged_ts.tzinfo is not None

    reloaded = get_order(tmp_db, record.id)
    assert reloaded is not None
    assert reloaded == record


def test_stage_order_quantity_override(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db, suggested_quantity=5)
    record = stage_order(tmp_db, score_id, quantity=1, now=NOW)
    assert record.quantity == 1


def test_stage_order_rejects_nonpositive_quantity(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db, suggested_quantity=0)
    with pytest.raises(ValueError, match="quantity"):
        stage_order(tmp_db, score_id, now=NOW)


def test_stage_order_unknown_score_raises(tmp_db: Engine) -> None:
    with pytest.raises(ValueError, match="strategy_scores"):
        stage_order(tmp_db, 999_999, now=NOW)


# --- state machine -----------------------------------------------------------


def test_status_universe_is_pinned() -> None:
    assert ORDER_STATUSES == frozenset(
        {
            "staged",
            "submitting",
            "submitted",
            "partial",
            "filled",
            "cancelled",
            "rejected",
            "abandoned",
            "skipped",
        }
    )
    assert TERMINAL_STATUSES == frozenset(
        {"filled", "cancelled", "rejected", "abandoned", "skipped"}
    )


EXPECTED_LEGAL: frozenset[tuple[str, str]] = frozenset(
    {
        ("staged", "submitting"),
        ("staged", "skipped"),
        ("submitting", "submitted"),
        ("submitting", "rejected"),
        ("submitting", "skipped"),
        ("submitted", "partial"),
        ("submitted", "filled"),
        ("submitted", "cancelled"),
        ("submitted", "rejected"),
        ("submitted", "abandoned"),
        ("partial", "filled"),
        ("partial", "cancelled"),
        ("partial", "abandoned"),
    }
)


def test_legal_transition_table_is_pinned() -> None:
    flattened = {
        (src, dst) for src, dsts in LEGAL_TRANSITIONS.items() for dst in dsts
    }
    assert flattened == EXPECTED_LEGAL


@pytest.mark.parametrize(("src", "dst"), sorted(EXPECTED_LEGAL))
def test_legal_transitions_apply(tmp_db: Engine, src: str, dst: str) -> None:
    order_id = _insert_order(tmp_db, src)
    record = transition(tmp_db, order_id, dst, now=NOW)
    assert record.status == dst


@pytest.mark.parametrize(
    ("src", "dst"),
    sorted(
        (s, d)
        for s in sorted({"staged", "submitting", "submitted", "partial", "filled",
                         "cancelled", "rejected", "abandoned", "skipped"})
        for d in sorted({"staged", "submitting", "submitted", "partial", "filled",
                         "cancelled", "rejected", "abandoned", "skipped"})
        if (s, d) not in EXPECTED_LEGAL
    ),
)
def test_illegal_transitions_raise(tmp_db: Engine, src: str, dst: str) -> None:
    order_id = _insert_order(tmp_db, src)
    with pytest.raises(IllegalOrderTransition):
        transition(tmp_db, order_id, dst, now=NOW)
    assert get_order(tmp_db, order_id).status == src  # type: ignore[union-attr]


def test_transition_unknown_order_raises(tmp_db: Engine) -> None:
    with pytest.raises(ValueError, match="order"):
        transition(tmp_db, 424_242, "submitting", now=NOW)


def test_submitted_records_ib_ids_and_ts(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitting")
    record = transition(
        tmp_db, order_id, "submitted", ib_order_id=11, ib_perm_id=987654, now=NOW
    )
    assert record.ib_order_id == 11
    assert record.ib_perm_id == 987654
    assert record.submitted_ts is not None and record.submitted_ts.tzinfo is not None
    assert record.terminal_ts is None


def test_terminal_transition_sets_terminal_ts_and_error(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitting")
    record = transition(
        tmp_db, order_id, "rejected", error="insufficient buying power", now=NOW
    )
    assert record.terminal_ts is not None and record.terminal_ts.tzinfo is not None
    assert record.last_error == "insufficient buying power"


# --- reprice -----------------------------------------------------------------


def test_bump_reprice_updates_price_and_count(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted", limit_price=-1.55)
    record = bump_reprice(tmp_db, order_id, new_limit_price=-1.45, now=NOW)
    assert record.limit_price == -1.45
    assert record.reprice_count == 1
    record = bump_reprice(tmp_db, order_id, new_limit_price=-1.35, now=NOW)
    assert record.reprice_count == 2


@pytest.mark.parametrize("status", ["staged", "filled", "cancelled", "skipped"])
def test_bump_reprice_requires_working_order(tmp_db: Engine, status: str) -> None:
    order_id = _insert_order(tmp_db, status, limit_price=-1.55)
    with pytest.raises(IllegalOrderTransition):
        bump_reprice(tmp_db, order_id, new_limit_price=-1.45, now=NOW)


# --- fills -------------------------------------------------------------------


def test_record_fill_and_dedupe_by_exec_id(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    first = record_fill(
        tmp_db, order_id, exec_id="0001.abc.01", side="SELL", price=1.20,
        qty=2, ts=NOW, leg_con_id=111,
    )
    duplicate = record_fill(
        tmp_db, order_id, exec_id="0001.abc.01", side="SELL", price=1.20,
        qty=2, ts=NOW, leg_con_id=111,
    )
    assert first is True
    assert duplicate is False
    with tmp_db.connect() as conn:
        rows = conn.execute(select(fills)).fetchall()
    assert len(rows) == 1


def test_set_fill_commission_by_exec_id(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    record_fill(
        tmp_db, order_id, exec_id="0002.def.01", side="BUY", price=0.40,
        qty=2, ts=NOW,
    )
    assert set_fill_commission(tmp_db, "0002.def.01", 1.31) is True
    assert set_fill_commission(tmp_db, "no-such-exec", 1.31) is False
    with tmp_db.connect() as conn:
        row = conn.execute(select(fills)).one()
    assert row.commission == 1.31


def test_net_premium_signed_per_leg_sum(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted", quantity=2)
    record_fill(
        tmp_db, order_id, exec_id="e1", side="SELL", price=1.20, qty=2, ts=NOW
    )
    record_fill(
        tmp_db, order_id, exec_id="e2", side="BUY", price=0.40, qty=2, ts=NOW
    )
    # (1.20*2 - 0.40*2) * 100 multiplier = +160 dollars net credit received.
    assert net_premium(tmp_db, order_id) == pytest.approx(160.0)


def test_net_premium_none_without_fills(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "submitted")
    assert net_premium(tmp_db, order_id) is None


# --- queries -----------------------------------------------------------------


def test_open_and_working_order_queries(tmp_db: Engine) -> None:
    by_status = {
        status: _insert_order(tmp_db, status)
        for status in (
            "staged", "submitting", "submitted", "partial",
            "filled", "cancelled", "rejected", "abandoned", "skipped",
        )
    }
    open_ids = {r.id for r in open_orders(tmp_db)}
    working_ids = {r.id for r in working_orders(tmp_db)}
    assert open_ids == {
        by_status["staged"], by_status["submitting"],
        by_status["submitted"], by_status["partial"],
    }
    assert working_ids == {by_status["submitted"], by_status["partial"]}


def test_get_order_unknown_returns_none(tmp_db: Engine) -> None:
    assert get_order(tmp_db, 31337) is None
