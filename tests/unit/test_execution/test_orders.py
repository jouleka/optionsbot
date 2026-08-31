"""Tests for the order ledger + state machine (IBK-124)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, insert, select
from sqlalchemy.exc import IntegrityError

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
    set_order_leg_contracts,
    stage_order,
    transition,
    working_orders,
)
from optionsbot.storage.schema import (
    entry_intent_consumptions,
    fills,
    managed_opportunities,
    orders,
    snapshots,
    strategy_scores,
)

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

MANAGED_PACKET = {
    "expected_value_model": "managed-stage-test",
    "managed_probability_model": "managed-stage-test",
    "managed_model_artifact_hash": "a" * 64,
    "managed_feature_schema_version": "managed_capture_features_v1",
    "managed_outcome_policy_version": "marketable_nbbo_15s_v1",
    "managed_model_trained_through": "2026-06-09",
    "managed_target_hit_probability": 0.5,
    "managed_stop_probability": 0.3,
    "managed_timeout_probability": 0.2,
}


def _insert_score(
    engine: Engine,
    *,
    suggested_quantity: int = 2,
    symbol: str = "SPY",
    suggestion_extra: dict[str, object] | None = None,
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
                    **(suggestion_extra or {}),
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


def _canonical_structure_hash(legs: list[dict[str, object]]) -> str:
    canonical = sorted(
        legs,
        key=lambda leg: (
            str(leg["symbol"]),
            str(leg["expiry"]),
            str(leg["right"]),
            float(leg["strike"]),
            str(leg["side"]),
            int(leg["quantity"]),
        ),
    )
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _insert_managed_binding(
    engine: Engine,
    score_id: int,
    *,
    admission_eligible: int = 1,
    shadow_only: int = 0,
    bot_action: str | None = "candidate",
    decided_at: datetime | None = NOW,
    captured_legs: list[dict[str, object]] | None = None,
    structure_hash: str | None = None,
) -> None:
    legs = captured_legs if captured_legs is not None else CONDOR_LEGS
    with engine.begin() as conn:
        row = conn.execute(
            select(
                strategy_scores.c.snapshot_id,
                strategy_scores.c.strategy,
                snapshots.c.symbol,
            )
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(strategy_scores.c.id == score_id)
        ).one()
        conn.execute(
            insert(managed_opportunities).values(
                opportunity_key=f"managed-stage-{score_id}",
                signal_id=f"managed-signal-{score_id}",
                session="2026-06-10",
                symbol=row.symbol,
                direction="bull",
                setup_type="fvg_retest",
                strategy=row.strategy,
                strategy_score_id=score_id,
                structure_hash=structure_hash or _canonical_structure_hash(legs),
                legs_json=legs,
                features_json={
                    "feature_schema_version": "managed_capture_features_v1",
                    "snapshot_id": row.snapshot_id,
                },
                policy_version="marketable_nbbo_15s_v1",
                decision_batch_id="managed-stage-batch",
                decision_score=78.0,
                decision_defined_risk=1,
                decision_max_loss=345.0,
                created_at=NOW,
                detected_at=NOW,
                baseline_action="hold",
                baseline_reason="calibration required",
                admission_eligible=admission_eligible,
                shadow_only=shadow_only,
                bot_action=bot_action,
                bot_reason=("scan admission" if bot_action is not None else None),
                bot_decided_at=(decided_at if bot_action is not None else None),
                decision_account_value_available=(1 if bot_action is not None else None),
                decision_account_value_usd=(50_000.0 if bot_action is not None else None),
                session_close_at=NOW + timedelta(hours=2),
                entry_cutoff_at=NOW + timedelta(hours=1),
                timeout_at=NOW + timedelta(hours=1),
                stop_pct=0.15,
                target_pct=0.225,
                commission_estimate=1.4,
                status="pending_entry",
                training_eligible=0,
            )
        )


def _assert_no_staged_side_effects(engine: Engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(select(orders.c.id)).fetchall() == []
        assert (
            conn.execute(select(entry_intent_consumptions.c.strategy_score_id)).fetchall()
            == []
        )


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


def test_set_order_leg_contracts_persists_exact_qualification(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db)
    record = stage_order(tmp_db, score_id, now=NOW)

    enriched = set_order_leg_contracts(
        tmp_db,
        record.id,
        ((1580, 100, "USD"), (1575, 100, "USD"),
         (1620, 100, "USD"), (1625, 100, "USD")),
    )

    assert [leg["con_id"] for leg in enriched.legs] == [1580, 1575, 1620, 1625]
    assert all(leg["multiplier"] == 100 for leg in enriched.legs)
    assert all(leg["currency"] == "USD" for leg in enriched.legs)


def test_stage_order_rejects_nonpositive_quantity(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db, suggested_quantity=0)
    with pytest.raises(ValueError, match="quantity"):
        stage_order(tmp_db, score_id, now=NOW)


def test_stage_order_unknown_score_raises(tmp_db: Engine) -> None:
    with pytest.raises(ValueError, match="strategy_scores"):
        stage_order(tmp_db, 999_999, now=NOW)


def test_stage_order_rejects_shadow_only_row_even_with_quantity_override(
    tmp_db: Engine,
) -> None:
    score_id = _insert_score(
        tmp_db,
        suggested_quantity=0,
        suggestion_extra={"shadow_only": True, "admission_enabled": False},
    )
    with pytest.raises(ValueError, match="shadow-only research"):
        stage_order(tmp_db, score_id, quantity=1, now=NOW)


def test_stage_order_accepts_exact_timely_managed_candidate(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    _insert_managed_binding(tmp_db, score_id)

    record = stage_order(tmp_db, score_id, now=NOW)

    assert record.strategy_score_id == score_id
    assert record.intent == "open"


@pytest.mark.parametrize("bot_action", [None, "hold"])
def test_stage_order_rejects_null_or_held_managed_disposition_atomically(
    tmp_db: Engine,
    bot_action: str | None,
) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    _insert_managed_binding(tmp_db, score_id, bot_action=bot_action)

    with pytest.raises(ValueError, match="candidate admission disposition"):
        stage_order(tmp_db, score_id, now=NOW)
    _assert_no_staged_side_effects(tmp_db)


def test_stage_order_uses_immutable_shadow_flags_not_clean_suggestion(
    tmp_db: Engine,
) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    _insert_managed_binding(
        tmp_db,
        score_id,
        admission_eligible=0,
        shadow_only=1,
        bot_action="hold",
    )

    with pytest.raises(ValueError, match="immutable research-only"):
        stage_order(tmp_db, score_id, now=NOW)
    _assert_no_staged_side_effects(tmp_db)


def test_stage_order_rejects_linked_candidate_without_model_packet(
    tmp_db: Engine,
) -> None:
    score_id = _insert_score(tmp_db)
    _insert_managed_binding(tmp_db, score_id)

    with pytest.raises(ValueError, match="complete managed model packet"):
        stage_order(tmp_db, score_id, now=NOW)
    _assert_no_staged_side_effects(tmp_db)


def test_stage_order_rejects_model_packet_without_managed_row_atomically(
    tmp_db: Engine,
) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)

    with pytest.raises(ValueError, match="no immutable opportunity binding"):
        stage_order(tmp_db, score_id, now=NOW)
    _assert_no_staged_side_effects(tmp_db)


def test_stage_order_rejects_future_candidate_disposition(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    _insert_managed_binding(
        tmp_db,
        score_id,
        decided_at=NOW + timedelta(minutes=30),
    )

    with pytest.raises(ValueError, match="future-dated"):
        stage_order(tmp_db, score_id, now=NOW)
    _assert_no_staged_side_effects(tmp_db)


@pytest.mark.parametrize(
    "execution_now",
    [NOW + timedelta(hours=1), NOW + timedelta(hours=1, microseconds=1)],
    ids=["at-cutoff", "after-cutoff"],
)
def test_stage_order_rejects_execution_at_or_after_frozen_cutoff_atomically(
    tmp_db: Engine,
    execution_now: datetime,
) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    _insert_managed_binding(tmp_db, score_id)

    with pytest.raises(ValueError, match="execution is at or after.*frozen entry cutoff"):
        stage_order(tmp_db, score_id, now=execution_now)
    _assert_no_staged_side_effects(tmp_db)


@pytest.mark.parametrize(
    "decided_at",
    [NOW - timedelta(seconds=1), NOW + timedelta(hours=1)],
)
def test_db_rejects_backdated_or_postcutoff_candidate(
    tmp_db: Engine,
    decided_at: datetime,
) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    with pytest.raises(IntegrityError, match="candidate_timing"):
        _insert_managed_binding(tmp_db, score_id, decided_at=decided_at)


def test_db_rejects_candidate_marked_admission_ineligible(tmp_db: Engine) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    with pytest.raises(IntegrityError, match="candidate_is_executable"):
        _insert_managed_binding(
            tmp_db,
            score_id,
            admission_eligible=0,
            shadow_only=0,
        )


@pytest.mark.parametrize(
    ("captured_legs", "structure_hash", "message"),
    [
        ([{**CONDOR_LEGS[0], "strike": 581.0}], None, "legs differ"),
        (None, "0" * 64, "structure hash is invalid"),
    ],
)
def test_stage_order_rejects_corrupt_managed_structure_atomically(
    tmp_db: Engine,
    captured_legs: list[dict[str, object]] | None,
    structure_hash: str | None,
    message: str,
) -> None:
    score_id = _insert_score(tmp_db, suggestion_extra=MANAGED_PACKET)
    _insert_managed_binding(
        tmp_db,
        score_id,
        captured_legs=captured_legs,
        structure_hash=structure_hash,
    )

    with pytest.raises(ValueError, match=message):
        stage_order(tmp_db, score_id, now=NOW)
    _assert_no_staged_side_effects(tmp_db)


def test_close_intent_is_not_subject_to_managed_entry_authorization(
    tmp_db: Engine,
) -> None:
    score_id = _insert_score(
        tmp_db,
        suggestion_extra={"shadow_only": True, "admission_enabled": False},
    )

    record = stage_order(tmp_db, score_id, intent="close", quantity=1, now=NOW)

    assert record.intent == "close"
    with tmp_db.connect() as conn:
        assert (
            conn.execute(select(entry_intent_consumptions.c.strategy_score_id)).fetchall()
            == []
        )


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
        # Self-loops on working states: ib_async re-delivers orderStatus with
        # the same status while fill quantities mutate — must be a no-op, not
        # an IllegalOrderTransition mid-pipeline.
        ("submitted", "submitted"),
        ("submitted", "partial"),
        ("submitted", "filled"),
        ("submitted", "cancelled"),
        ("submitted", "rejected"),
        ("submitted", "abandoned"),
        ("partial", "partial"),
        ("partial", "filled"),
        ("partial", "cancelled"),
        # IBKR can reject/deactivate the REMAINDER of a partially-filled order.
        ("partial", "rejected"),
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


def test_same_status_redelivery_is_a_noop(tmp_db: Engine) -> None:
    # ib_async fires orderStatus repeatedly with an unchanged status string
    # while filled/remaining mutate; re-delivery must not rewrite anything.
    later = datetime(2026, 6, 10, 16, 45, tzinfo=UTC)
    order_id = _insert_order(
        tmp_db, "submitted", submitted_ts=NOW, ib_order_id=7, reprice_count=3
    )
    record = transition(tmp_db, order_id, "submitted", now=later)
    assert record.status == "submitted"
    assert record.submitted_ts == NOW  # NOT rewritten to `later`
    assert record.reprice_count == 3

    partial_id = _insert_order(tmp_db, "partial")
    record = transition(tmp_db, partial_id, "partial", now=later)
    assert record.status == "partial"
    assert record.terminal_ts is None


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


def test_record_fill_correction_replaces_older_ibkr_revision(tmp_db: Engine) -> None:
    """Regression for the cumulative/revised GOOGL close fills on 2026-08-10."""
    order_id = _insert_order(tmp_db, "filled", quantity=2)
    prefix = "0000e22a.6a794f7d.01.01"
    assert record_fill(
        tmp_db, order_id, exec_id=f"{prefix}.01", side="SELL", price=0.92,
        qty=1, ts=NOW, leg_con_id=905627718,
    )
    assert set_fill_commission(tmp_db, f"{prefix}.01", 0.785245)
    assert record_fill(
        tmp_db, order_id, exec_id=f"{prefix}.02", side="SELL", price=0.92,
        qty=2, ts=NOW + timedelta(seconds=11), leg_con_id=905627718,
    )
    assert set_fill_commission(tmp_db, f"{prefix}.02", 0.87049)
    # A reconnect may replay the superseded .01 after .02; it stays stale.
    assert not record_fill(
        tmp_db, order_id, exec_id=f"{prefix}.01", side="SELL", price=0.92,
        qty=1, ts=NOW, leg_con_id=905627718,
    )

    with tmp_db.connect() as conn:
        rows = conn.execute(select(fills).where(fills.c.order_id == order_id)).fetchall()
    assert len(rows) == 1
    assert rows[0].ib_exec_id == f"{prefix}.02"
    assert rows[0].qty == 2
    assert rows[0].commission == pytest.approx(0.87049)


def test_record_fill_replay_repairs_preexisting_revision_double_count(
    tmp_db: Engine,
) -> None:
    """Deployment replay must repair ledgers written by the pre-fix daemon."""
    order_id = _insert_order(tmp_db, "filled", quantity=2)
    prefix = "0000e22a.6a794f7c.01.01"
    with tmp_db.begin() as conn:
        conn.execute(
            insert(fills),
            [
                {
                    "order_id": order_id, "ib_exec_id": f"{prefix}.01",
                    "side": "BUY", "price": 0.29, "qty": 1,
                    "ts": NOW, "commission": 0.6173, "leg_con_id": 905206190,
                },
                {
                    "order_id": order_id, "ib_exec_id": f"{prefix}.02",
                    "side": "BUY", "price": 0.29, "qty": 1,
                    "ts": NOW + timedelta(seconds=11), "commission": 0.6173,
                    "leg_con_id": 905206190,
                },
            ],
        )

    # Replaying the latest correction coalesces the already-corrupt family.
    assert record_fill(
        tmp_db, order_id, exec_id=f"{prefix}.02", side="BUY", price=0.29,
        qty=1, ts=NOW + timedelta(seconds=11), leg_con_id=905206190,
    )
    with tmp_db.connect() as conn:
        rows = conn.execute(select(fills).where(fills.c.order_id == order_id)).fetchall()
    assert [row.ib_exec_id for row in rows] == [f"{prefix}.02"]


def test_fill_commission_accepts_exchange_rebate(tmp_db: Engine) -> None:
    order_id = _insert_order(tmp_db, "filled")
    record_fill(
        tmp_db, order_id, exec_id="rebate.01", side="BUY", price=0.29,
        qty=1, ts=NOW, leg_con_id=905206190,
    )
    assert set_fill_commission(tmp_db, "rebate.01", -0.0827)


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


def test_net_premium_sums_multiple_fills_per_leg(tmp_db: Engine) -> None:
    # One leg can fill across several executions (each with its own execId).
    order_id = _insert_order(tmp_db, "partial", quantity=2)
    record_fill(tmp_db, order_id, exec_id="m1", side="SELL", price=1.20, qty=1, ts=NOW)
    record_fill(tmp_db, order_id, exec_id="m2", side="SELL", price=1.10, qty=1, ts=NOW)
    record_fill(tmp_db, order_id, exec_id="m3", side="BUY", price=0.40, qty=2, ts=NOW)
    # (1.20 + 1.10 - 0.40*2) * 100 = 150 dollars net credit.
    assert net_premium(tmp_db, order_id) == pytest.approx(150.0)


def test_record_fill_rejects_lowercase_side(tmp_db: Engine) -> None:
    # legs_json stores side as lowercase 'buy'/'sell'; fills require uppercase
    # IBKR execution sides. A clear ValueError beats an IntegrityError if
    # IBK-125 ever wires a leg side straight through.
    order_id = _insert_order(tmp_db, "submitted")
    with pytest.raises(ValueError, match="side"):
        record_fill(tmp_db, order_id, exec_id="x1", side="sell", price=1.0, qty=1, ts=NOW)


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


def test_broker_order_id_has_one_active_ledger_owner(tmp_db: Engine) -> None:
    _insert_order(tmp_db, "submitted", ib_order_id=77)
    with pytest.raises(IntegrityError):
        _insert_order(tmp_db, "submitted", ib_order_id=77)


def test_broker_order_id_can_be_reused_after_prior_owner_is_terminal(
    tmp_db: Engine,
) -> None:
    prior = _insert_order(tmp_db, "cancelled", ib_order_id=77)
    current = _insert_order(tmp_db, "submitting")

    transition(tmp_db, current, "submitted", ib_order_id=77, now=NOW)

    assert get_order(tmp_db, prior).ib_order_id == 77  # type: ignore[union-attr]
    assert get_order(tmp_db, current).ib_order_id == 77  # type: ignore[union-attr]


def test_get_order_unknown_returns_none(tmp_db: Engine) -> None:
    assert get_order(tmp_db, 31337) is None
