from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine, delete, insert, update
from sqlalchemy.exc import IntegrityError

from optionsbot.admission_policy import configured_admission_policy
from optionsbot.config import Settings
from optionsbot.execution.managed_boundary import (
    ManagedExecutionError,
    refresh_managed_prediction,
)
from optionsbot.learning.managed_model import (
    ManagedModelArtifact,
    ManagedSample,
    evaluate_prospective_rows,
    fit_managed_model,
    score_frozen_artifact,
)
from optionsbot.learning.repository import (
    PROSPECTIVE_HOLDOUT_PROTOCOL,
    artifact_path,
    load_training_samples,
    prospective_cohort_content_sha256,
)
from optionsbot.storage.schema import (
    managed_model_evaluations,
    managed_models,
    managed_opportunities,
    snapshots,
    strategy_scores,
)

DETECTED_AT = datetime(2026, 8, 28, 14, 0, tzinfo=UTC)
EXECUTION_NOW = DETECTED_AT + timedelta(minutes=30)


def _samples() -> list[ManagedSample]:
    outcomes = ("target", "stop", "timeout")
    first = date(2026, 6, 1)
    rows: list[ManagedSample] = []
    for index in range(30):
        outcome = outcomes[index % len(outcomes)]
        rows.append(
            ManagedSample(
                opportunity_id=index + 1,
                opportunity_key=f"train-{index}",
                signal_id=f"signal-{index}",
                session=(first + timedelta(days=index)).isoformat(),
                features={
                    "quality.impulse": {
                        "target": 2.0,
                        "stop": -2.0,
                        "timeout": 0.0,
                    }[outcome]
                },
                outcome=outcome,  # type: ignore[arg-type]
                basis_dollars=100.0,
                target_gain=30.0,
                stop_loss=15.0,
                timeout_gross_return=0.0,
                costs=1.4,
                realized_net_pnl={
                    "target": 28.6,
                    "stop": -16.4,
                    "timeout": -1.4,
                }[outcome],
            )
        )
    return rows


def _option_leg(*, strike: float = 650.0) -> dict[str, object]:
    return {
        "symbol": "SPY",
        "side": "buy",
        "sec_type": "OPT",
        "expiry": "20260828",
        "strike": strike,
        "right": "C",
        "quantity": 1,
    }


def _structure_hash(legs: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        legs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0.0 else "negative_infinity"
    if isinstance(value, dict):
        return {str(key): _evidence_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_evidence_safe(item) for item in value]
    return value


def _setup(
    engine: Engine,
    artifact_dir: Path,
    *,
    score_legs: list[dict[str, object]] | None = None,
    captured_legs: list[dict[str, object]] | None = None,
    captured_structure_hash: str | None = None,
    admission_eligible: int = 1,
    shadow_only: int = 0,
    bot_action: str | None = "candidate",
) -> tuple[Settings, int, dict[str, object], ManagedModelArtifact]:
    settings = Settings()
    settings.managed_learning.artifact_dir = artifact_dir
    admission_policy = configured_admission_policy(settings)
    artifact = fit_managed_model(
        _samples(),
        model_version="managed-boundary-test",
        feature_schema_version=settings.managed_learning.feature_schema_version,
        outcome_policy_version=settings.managed_learning.outcome_policy_version,
        ev_residuals=[-0.01, 0.0, 0.01],
    )
    artifact_path(settings.managed_learning.artifact_dir, artifact.model_version).write_text(
        artifact.to_json(), encoding="utf-8"
    )
    observed = DETECTED_AT
    suggestion: dict[str, object] = {
        "expected_value_model": artifact.model_version,
        "managed_probability_model": artifact.model_version,
        "managed_model_artifact_hash": artifact.artifact_hash,
        "managed_feature_schema_version": artifact.feature_schema_version,
        "managed_outcome_policy_version": artifact.outcome_policy_version,
        "managed_model_trained_through": artifact.trained_through_session,
        "managed_target_hit_probability": 0.5,
        "managed_stop_probability": 0.3,
        "managed_timeout_probability": 0.2,
    }
    executable_legs = score_legs if score_legs is not None else [_option_leg()]
    frozen_legs = captured_legs if captured_legs is not None else executable_legs
    frozen_hash = (
        captured_structure_hash
        if captured_structure_hash is not None
        else _structure_hash(frozen_legs)
    )
    with engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="SPY", ts=observed, spot=650.0, raw_json={})
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="long_call",
                    score=80.0,
                    rationale="test",
                    legs_json=executable_legs,
                    suggestion_json=suggestion,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(managed_opportunities).values(
                opportunity_key="managed-boundary-candidate",
                signal_id="2026-08-28:SPY:bull:test",
                session="2026-08-28",
                symbol="SPY",
                direction="bull",
                setup_type="opening_momentum",
                strategy="long_call",
                strategy_score_id=score_id,
                structure_hash=frozen_hash,
                legs_json=frozen_legs,
                features_json={
                    "feature_schema_version": artifact.feature_schema_version,
                    "snapshot_id": snapshot_id,
                    "quality": {"impulse": 2.0},
                },
                policy_version=artifact.outcome_policy_version,
                decision_batch_id="live-scan-batch",
                decision_score=80.0,
                decision_defined_risk=1,
                decision_max_loss=100.0,
                created_at=observed,
                detected_at=observed,
                baseline_action="hold",
                baseline_reason="calibration_required",
                admission_eligible=admission_eligible,
                shadow_only=shadow_only,
                bot_action=bot_action,
                bot_reason=("scan_admission_passed" if bot_action is not None else None),
                bot_decided_at=(observed if bot_action is not None else None),
                decision_account_value_available=(1 if bot_action is not None else None),
                decision_account_value_usd=(1_000_000.0 if bot_action is not None else None),
                session_close_at=observed + timedelta(hours=2),
                entry_cutoff_at=observed + timedelta(hours=1),
                timeout_at=observed + timedelta(hours=1),
                stop_pct=0.15,
                target_pct=0.30,
                commission_estimate=1.4,
                status="pending_entry",
                training_eligible=0,
            )
        )
        cohort_ids = [901, 902, 903]
        for cohort_id, session, outcome in (
            (901, "2026-08-20", "target"),
            (902, "2026-08-21", "target"),
            (903, "2026-08-22", "target"),
        ):
            detected = datetime.fromisoformat(f"{session}T14:00:00+00:00")
            cohort_snapshot_id = int(
                conn.execute(
                    insert(snapshots).values(
                        symbol="SPY",
                        ts=detected,
                        spot=640.0,
                        raw_json={},
                    )
                ).inserted_primary_key[0]
            )
            cohort_leg = {**_option_leg(), "expiry": session.replace("-", "")}
            cohort_score_id = int(
                conn.execute(
                    insert(strategy_scores).values(
                        snapshot_id=cohort_snapshot_id,
                        strategy="long_call",
                        score=75.0,
                        rationale="prospective holdout fixture",
                        legs_json=[cohort_leg],
                        suggestion_json={},
                    )
                ).inserted_primary_key[0]
            )
            gross_pnl = 30.0 if outcome == "target" else -15.0
            conn.execute(
                insert(managed_opportunities).values(
                    id=cohort_id,
                    opportunity_key=f"managed-boundary-holdout-{cohort_id}",
                    signal_id=f"holdout-signal-{cohort_id}",
                    session=session,
                    symbol="SPY",
                    direction="bull",
                    setup_type="fvg_retest",
                    strategy="long_call",
                    strategy_score_id=cohort_score_id,
                    structure_hash=_structure_hash([cohort_leg]),
                    legs_json=[cohort_leg],
                    features_json={
                        "feature_schema_version": artifact.feature_schema_version,
                        "snapshot_id": cohort_snapshot_id,
                    },
                    policy_version=artifact.outcome_policy_version,
                    decision_batch_id=f"holdout-batch-{cohort_id}",
                    decision_score=75.0,
                    decision_defined_risk=1,
                    decision_max_loss=100.0,
                    created_at=detected,
                    detected_at=detected,
                    baseline_action="hold",
                    baseline_reason="prospective fixture",
                    admission_eligible=1,
                    shadow_only=0,
                    bot_action="hold",
                    bot_reason="prospective fixture hold",
                    bot_decided_at=detected,
                    decision_account_value_available=1,
                    decision_account_value_usd=1_000_000.0,
                    session_close_at=detected + timedelta(hours=2),
                    entry_cutoff_at=detected + timedelta(hours=1),
                    timeout_at=detected + timedelta(hours=1),
                    entry_ts=detected + timedelta(minutes=1),
                    entry_net=-1.0,
                    basis_dollars=100.0,
                    stop_pct=0.15,
                    target_pct=0.30,
                    commission_estimate=1.4,
                    status="resolved",
                    outcome=outcome,
                    resolved_at=detected + timedelta(minutes=30),
                    exit_net=-1.3 if outcome == "target" else -0.85,
                    gross_pnl=gross_pnl,
                    net_pnl=gross_pnl - 1.4,
                    valid_marks=3,
                    training_eligible=1,
                    resolution_reason="prospective fixture outcome",
                )
            )
        model_id = int(
            conn.execute(
                insert(managed_models).values(
                    model_version=artifact.model_version,
                    artifact_hash=artifact.artifact_hash,
                    feature_schema_version=artifact.feature_schema_version,
                    outcome_policy_version=artifact.outcome_policy_version,
                    trained_from_session=artifact.trained_from_session,
                    trained_through_session=artifact.trained_through_session,
                    metrics_json={
                        "eligible": True,
                        "model_role": "causal_base",
                        "include_hermes_context": False,
                        "promotion_allowed": True,
                        "deployment_scope": "paper_admission_candidate",
                        "artifact_file": f"{artifact.model_version}.json",
                        "admission_selection_policy": admission_policy.payload(),
                    },
                    status="promoted",
                    created_at=observed - timedelta(days=1),
                    promoted_at=observed - timedelta(days=1),
                )
            ).inserted_primary_key[0]
        )
    cohort_content_sha256 = prospective_cohort_content_sha256(engine, cohort_ids)
    samples_by_id = {
        sample.opportunity_id: sample
        for sample in load_training_samples(
            engine,
            feature_schema_version=artifact.feature_schema_version,
            canonical_per_signal=False,
        )
    }
    scored_holdout = score_frozen_artifact(
        artifact,
        [samples_by_id[row_id] for row_id in cohort_ids],
    )
    evaluation_policy = {
        "min_sessions": 2,
        "min_independent_signals": 3,
        "min_admitted": 2,
        "min_admitted_sessions": 2,
        "min_profit_factor": 1.0,
        "score_floor": admission_policy.score_floor,
        "single_trade_cap_pct": admission_policy.single_trade_cap_pct,
        "max_candidates_per_batch": admission_policy.max_candidates_per_batch,
        "max_admitted_per_session": admission_policy.max_admitted_per_session,
        "bootstrap_iterations": 200,
    }
    report = evaluate_prospective_rows(scored_holdout, **evaluation_policy)
    absolute = asdict(report)
    absolute.pop("rows", None)
    with engine.begin() as conn:
        conn.execute(
            insert(managed_model_evaluations).values(
                model_id=model_id,
                evaluation_kind="holdout",
                fold_index=0,
                train_from_session=artifact.trained_from_session,
                train_through_session=artifact.trained_through_session,
                test_from_session="2026-08-20",
                test_through_session="2026-08-22",
                metrics_json={
                    "eligible": True,
                    "model_version": artifact.model_version,
                    "artifact_hash": artifact.artifact_hash,
                    "feature_schema_version": artifact.feature_schema_version,
                    "outcome_policy_version": artifact.outcome_policy_version,
                    "trained_through_session": artifact.trained_through_session,
                    "cohort_sha256": hashlib.sha256(b"901|902|903").hexdigest(),
                    "cohort_opportunity_ids": cohort_ids,
                    "cohort_content_sha256": cohort_content_sha256,
                    "evaluation_policy": evaluation_policy,
                    "absolute": _evidence_safe(absolute),
                    "incumbent": None,
                    "incremental_policy": None,
                    "promotion_allowed": True,
                    "evaluation_protocol": PROSPECTIVE_HOLDOUT_PROTOCOL,
                },
                created_at=observed - timedelta(days=1),
            )
        )
    return settings, score_id, suggestion, artifact


def test_refresh_reinfers_from_frozen_features_and_fresh_costs(tmp_db: Engine, tmp_path) -> None:
    settings, score_id, suggestion, artifact = _setup(tmp_db, tmp_path)
    low_cost = refresh_managed_prediction(
        tmp_db,
        settings,
        score_id,
        suggestion,
        fresh_basis_dollars=100.0,
        fresh_costs=1.4,
        maximum_profit=None,
        now=EXECUTION_NOW,
    )
    higher_cost = refresh_managed_prediction(
        tmp_db,
        settings,
        score_id,
        suggestion,
        fresh_basis_dollars=100.0,
        fresh_costs=4.4,
        maximum_profit=None,
        now=EXECUTION_NOW,
    )
    assert low_cost.model_version == artifact.model_version
    assert low_cost.artifact_hash == artifact.artifact_hash
    assert low_cost.expected_value_lcb > 0.0
    assert higher_cost.expected_value_lcb == pytest.approx(low_cost.expected_value_lcb - 3.0)


@pytest.mark.parametrize("bot_action", [None, "hold"])
def test_refresh_rejects_null_or_held_admission_disposition(
    tmp_db: Engine,
    tmp_path,
    bot_action: str | None,
) -> None:
    settings, score_id, suggestion, _ = _setup(
        tmp_db,
        tmp_path,
        bot_action=bot_action,
    )
    with pytest.raises(ManagedExecutionError, match="candidate admission disposition"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_refresh_rejects_shadow_after_mutable_suggestion_flags_are_removed(
    tmp_db: Engine,
    tmp_path,
) -> None:
    # The score packet deliberately claims no shadow status, simulating a
    # corrupted suggestion_json. Immutable opportunity authority must win.
    settings, score_id, suggestion, _ = _setup(
        tmp_db,
        tmp_path,
        admission_eligible=0,
        shadow_only=1,
        bot_action="hold",
    )
    assert "shadow_only" not in suggestion
    assert "admission_enabled" not in suggestion
    with pytest.raises(ManagedExecutionError, match="immutable research-only"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_refresh_rejects_non_shadow_but_admission_ineligible_opportunity(
    tmp_db: Engine,
    tmp_path,
) -> None:
    settings, score_id, suggestion, _ = _setup(
        tmp_db,
        tmp_path,
        admission_eligible=0,
        shadow_only=0,
        bot_action="hold",
    )
    with pytest.raises(ManagedExecutionError, match="immutable research-only"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_refresh_rejects_score_legs_that_differ_from_frozen_capture(
    tmp_db: Engine, tmp_path
) -> None:
    settings, score_id, suggestion, _ = _setup(
        tmp_db,
        tmp_path,
        score_legs=[_option_leg(strike=655.0)],
        captured_legs=[_option_leg(strike=650.0)],
    )
    with pytest.raises(ManagedExecutionError, match="legs differ"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_refresh_rejects_invalid_frozen_structure_hash(tmp_db: Engine, tmp_path) -> None:
    settings, score_id, suggestion, _ = _setup(
        tmp_db,
        tmp_path,
        captured_structure_hash="0" * 64,
    )
    with pytest.raises(ManagedExecutionError, match="structure hash is invalid"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_managed_score_structure_and_binding_are_storage_immutable(
    tmp_db: Engine, tmp_path
) -> None:
    _, score_id, _, _ = _setup(tmp_db, tmp_path)
    with pytest.raises(IntegrityError, match="managed strategy score structure is immutable"):
        with tmp_db.begin() as conn:
            conn.execute(
                update(strategy_scores)
                .where(strategy_scores.c.id == score_id)
                .values(legs_json=[_option_leg(strike=655.0)])
            )
    with pytest.raises(IntegrityError, match="managed opportunity identity is immutable"):
        with tmp_db.begin() as conn:
            conn.execute(
                update(managed_opportunities)
                .where(managed_opportunities.c.strategy_score_id == score_id)
                .values(strategy_score_id=None)
            )
    with pytest.raises(IntegrityError, match="managed strategy score cannot be deleted"):
        with tmp_db.begin() as conn:
            conn.execute(delete(strategy_scores).where(strategy_scores.c.id == score_id))


def test_managed_binding_remains_non_null_without_identity_trigger(
    tmp_db: Engine, tmp_path
) -> None:
    _, score_id, _, _ = _setup(tmp_db, tmp_path)
    # The column-level invariant is independent of the SQLite immutability
    # trigger, so even a damaged deployment cannot orphan managed evidence.
    with pytest.raises(IntegrityError, match="strategy_score_id"):
        with tmp_db.begin() as conn:
            conn.exec_driver_sql("DROP TRIGGER managed_opportunities_protect_identity")
            conn.execute(
                update(managed_opportunities)
                .where(managed_opportunities.c.strategy_score_id == score_id)
                .values(strategy_score_id=None)
            )


def test_refresh_rejects_legacy_binary_probability_packet(tmp_db: Engine, tmp_path) -> None:
    settings, score_id, suggestion, _ = _setup(tmp_db, tmp_path)
    suggestion.pop("managed_timeout_probability")
    with pytest.raises(ManagedExecutionError, match="three-event"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("managed_probability_model", "retired-version"),
        ("managed_model_artifact_hash", "0" * 64),
        ("managed_feature_schema_version", "obsolete-schema"),
        ("managed_outcome_policy_version", "obsolete-policy"),
    ],
)
def test_refresh_rejects_stale_candidate_artifact_identity(
    tmp_db: Engine, tmp_path, field: str, value: str
) -> None:
    settings, score_id, suggestion, _ = _setup(tmp_db, tmp_path)
    suggestion[field] = value
    with pytest.raises(ManagedExecutionError, match="no longer the promoted"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_refresh_rejects_retired_model(tmp_db: Engine, tmp_path) -> None:
    settings, score_id, suggestion, artifact = _setup(tmp_db, tmp_path)
    with tmp_db.begin() as conn:
        conn.execute(
            update(managed_models)
            .where(managed_models.c.model_version == artifact.model_version)
            .values(status="retired")
        )
    with pytest.raises(ManagedExecutionError, match="currently promoted"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_refresh_rejects_tampered_artifact_file(tmp_db: Engine, tmp_path) -> None:
    settings, score_id, suggestion, artifact = _setup(tmp_db, tmp_path)
    path = artifact_path(tmp_path, artifact.model_version)
    path.write_text(
        artifact.to_json().replace('"sample_count":30', '"sample_count":31'),
        encoding="utf-8",
    )
    with pytest.raises(ManagedExecutionError, match="failed verification"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


def test_refresh_rejects_non_positive_fresh_lcb(tmp_db: Engine, tmp_path) -> None:
    settings, score_id, suggestion, _ = _setup(tmp_db, tmp_path)
    with pytest.raises(ManagedExecutionError, match="lower bound is not positive"):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=100.0,
            maximum_profit=None,
            now=EXECUTION_NOW,
        )


@pytest.mark.parametrize(
    "execution_now",
    [DETECTED_AT + timedelta(hours=1), DETECTED_AT + timedelta(hours=1, microseconds=1)],
    ids=["at-cutoff", "after-cutoff"],
)
def test_refresh_rejects_execution_at_or_after_frozen_cutoff(
    tmp_db: Engine,
    tmp_path: Path,
    execution_now: datetime,
) -> None:
    settings, score_id, suggestion, _ = _setup(tmp_db, tmp_path)

    with pytest.raises(
        ManagedExecutionError,
        match="execution is at or after.*frozen entry cutoff",
    ):
        refresh_managed_prediction(
            tmp_db,
            settings,
            score_id,
            suggestion,
            fresh_basis_dollars=100.0,
            fresh_costs=1.4,
            maximum_profit=None,
            now=execution_now,
        )
