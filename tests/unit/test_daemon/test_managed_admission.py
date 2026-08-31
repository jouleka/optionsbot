from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import insert, select

from optionsbot.admission_policy import configured_admission_policy
from optionsbot.config import Settings
from optionsbot.daemon.managed_admission import apply_promoted_managed_model
from optionsbot.learning.managed_model import (
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
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.storage.schema import (
    managed_model_evaluations,
    managed_models,
    managed_opportunities,
    snapshots,
    strategy_scores,
)
from optionsbot.strategies import Leg, StrategySuggestion


def _artifact_samples() -> list[ManagedSample]:
    outcomes = ("target", "stop", "timeout")
    result: list[ManagedSample] = []
    first = date(2026, 6, 1)
    for index in range(30):
        outcome = outcomes[index % 3]
        result.append(
            ManagedSample(
                opportunity_id=index + 1,
                opportunity_key=f"train-{index}",
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
                timeout_gross_return=0.01 if outcome == "timeout" else 0.0,
                costs=1.4,
                realized_net_pnl={"target": 28.6, "stop": -16.4, "timeout": -0.4}[outcome],
                signal_id=f"signal-{index}",
            )
        )
    return result


def _scored() -> ScoredStrategy:
    suggestion = StrategySuggestion(
        strategy_name="long_call",
        legs=(
            Leg(
                symbol="SPY",
                side="buy",
                expiry="20260828",
                strike=650.0,
                right="C",
            ),
        ),
        credit_or_debit=-100.0,
        max_loss=100.0,
        max_profit=None,
        prob_profit=0.5,
        suggested_quantity=1,
        defined_risk=True,
        rationale="test",
        expected_value=None,
    )
    return ScoredStrategy(
        "long_call",
        80.0,
        FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        suggestion,
        "test",
    )


def _evidence_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0.0 else "negative_infinity"
    if isinstance(value, dict):
        return {str(key): _evidence_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_evidence_safe(item) for item in value]
    return value


def test_promoted_base_model_revalues_frozen_candidate(tmp_db, tmp_path) -> None:
    settings = Settings()
    settings.managed_learning.artifact_dir = tmp_path
    admission_policy = configured_admission_policy(settings)
    artifact = fit_managed_model(
        _artifact_samples(),
        model_version="managed-base-test",
        feature_schema_version=settings.managed_learning.feature_schema_version,
        outcome_policy_version=settings.managed_learning.outcome_policy_version,
        ev_residuals=[-0.01, 0.0, 0.01],
    )
    artifact_path(tmp_path, artifact.model_version).write_text(artifact.to_json(), encoding="utf-8")
    observed = datetime(2026, 8, 28, 14, 0, tzinfo=UTC)
    raw_plan = {
        "status": "entry_confirmed",
        "source": "trusted_daemon",
        "signal_id": "2026-08-28:SPY:bull:fvg",
        "session": "2026-08-28",
        "direction": "bull",
        "setup_type": "fvg_retest",
        "stop_pct": 0.15,
        "target_pct": 0.30,
    }
    with tmp_db.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=observed,
                    spot=650.0,
                    regime_dir="bull",
                    regime_iv="neutral",
                    raw_json={"opening_range_fvg": raw_plan},
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="long_call",
                    score=80.0,
                    rationale="test",
                    legs_json=[],
                    suggestion_json={
                        "credit_or_debit": -100.0,
                        "max_loss": 100.0,
                        "max_profit": None,
                        "managed_marketable_basis_dollars": 100.0,
                    },
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(managed_opportunities).values(
                opportunity_key="candidate-key",
                signal_id=raw_plan["signal_id"],
                session=raw_plan["session"],
                symbol="SPY",
                direction="bull",
                setup_type="fvg_retest",
                strategy="long_call",
                strategy_score_id=score_id,
                structure_hash="structure",
                legs_json=[],
                features_json={"quality": {"impulse": 2.0}},
                policy_version=settings.managed_learning.outcome_policy_version,
                decision_batch_id="live-scan-batch",
                decision_score=80.0,
                decision_defined_risk=1,
                decision_max_loss=100.0,
                created_at=observed,
                detected_at=observed,
                baseline_action="hold",
                baseline_reason="calibration_required",
                admission_eligible=1,
                shadow_only=0,
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
        cohort_ids = [801, 802, 803]
        for cohort_id, session, outcome in (
            (801, "2026-08-20", "target"),
            (802, "2026-08-21", "target"),
            (803, "2026-08-22", "target"),
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
            leg = {
                "symbol": "SPY",
                "side": "buy",
                "sec_type": "OPT",
                "expiry": session.replace("-", ""),
                "strike": 640.0,
                "right": "C",
                "quantity": 1,
            }
            cohort_score_id = int(
                conn.execute(
                    insert(strategy_scores).values(
                        snapshot_id=cohort_snapshot_id,
                        strategy="long_call",
                        score=75.0,
                        rationale="prospective holdout fixture",
                        legs_json=[leg],
                        suggestion_json={},
                    )
                ).inserted_primary_key[0]
            )
            gross_pnl = 30.0 if outcome == "target" else -15.0
            conn.execute(
                insert(managed_opportunities).values(
                    id=cohort_id,
                    opportunity_key=f"managed-admission-holdout-{cohort_id}",
                    signal_id=f"holdout-signal-{cohort_id}",
                    session=session,
                    symbol="SPY",
                    direction="bull",
                    setup_type="fvg_retest",
                    strategy="long_call",
                    strategy_score_id=cohort_score_id,
                    structure_hash=f"structure-{cohort_id}",
                    legs_json=[leg],
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
    cohort_content_sha256 = prospective_cohort_content_sha256(tmp_db, cohort_ids)
    samples_by_id = {
        sample.opportunity_id: sample
        for sample in load_training_samples(
            tmp_db,
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
    with tmp_db.begin() as conn:
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
                    "cohort_sha256": hashlib.sha256(b"801|802|803").hexdigest(),
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

    source = _scored()
    result = apply_promoted_managed_model(tmp_db, settings, snapshot_id, (source,))
    assert len(result) == 1
    assert result[0].suggestion.expected_value is not None
    with tmp_db.connect() as conn:
        stored = conn.execute(
            select(strategy_scores.c.suggestion_json).where(strategy_scores.c.id == score_id)
        ).scalar_one()
    assert stored["expected_value"] == result[0].suggestion.expected_value
    assert stored["expected_value_model"] == artifact.model_version
    assert stored["managed_stop_probability"] is not None
    assert stored["managed_timeout_probability"] is not None
    assert stored["managed_model_artifact_hash"] == artifact.artifact_hash
    assert stored["managed_feature_schema_version"] == artifact.feature_schema_version
    assert stored["managed_outcome_policy_version"] == artifact.outcome_policy_version


def test_no_promoted_model_preserves_unavailable_managed_ev(tmp_db) -> None:
    settings = Settings()
    source = _scored()
    assert apply_promoted_managed_model(tmp_db, settings, 999, (source,)) == (source,)
