from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine, insert, select, update

from optionsbot.admission_policy import AdmissionSelectionPolicy
from optionsbot.learning.managed_model import (
    ManagedModelArtifact,
    ManagedSample,
    compare_context_incremental_value,
    evaluate_prospective_rows,
    fit_managed_model,
    score_frozen_artifact,
)
from optionsbot.learning.repository import (
    BASE_MODEL_ROLE,
    CONTEXT_MODEL_ROLE,
    PROSPECTIVE_HOLDOUT_PROTOCOL,
    _managed_sample_from_row,
    artifact_path,
    load_promoted_model,
    load_training_samples,
    promote_challenger,
    prospective_cohort_content_sha256,
    train_challenger,
)
from optionsbot.managed_contract import MANAGED_FEATURE_SCHEMA_VERSION
from optionsbot.storage.schema import (
    managed_context_reviews,
    managed_model_evaluations,
    managed_models,
    managed_opportunities,
    snapshots,
    strategy_scores,
)


def _insert_opportunity(
    engine: Engine,
    *,
    row_id: int,
    signal_id: str,
    strategy: str,
    commission: float,
    outcome: str,
    net_pnl: float,
    entry_offset_minutes: int | None = 2,
    shadow_only: bool = False,
    admission_eligible: bool | None = None,
    session: str = "2026-08-28",
    policy_version: str = "marketable_nbbo_15s_v1",
    feature_schema_version: str = MANAGED_FEATURE_SCHEMA_VERSION,
    decision_offset_minutes: int = 0,
    decision_account_value_usd: float | None = 10_000.0,
) -> None:
    now = datetime.fromisoformat(f"{session}T14:00:00+00:00")
    score_id = 100_000 + row_id
    snapshot_id = 200_000 + row_id
    is_admission_eligible = not shadow_only if admission_eligible is None else admission_eligible
    resolved_at = now + timedelta(minutes=30)
    timeout_at = now + timedelta(hours=1)
    if outcome == "timeout":
        resolved_at = timeout_at
    gross_pnl = net_pnl + commission
    exit_net = -1.0 - gross_pnl / 100.0
    is_runtime_candidate = (
        is_admission_eligible
        and not shadow_only
        and decision_account_value_usd is not None
    )
    with engine.begin() as conn:
        conn.execute(
            insert(snapshots).values(
                id=snapshot_id,
                symbol="SPY",
                ts=now,
                spot=600.0,
                raw_json={},
            )
        )
        conn.execute(
            insert(strategy_scores).values(
                id=score_id,
                snapshot_id=snapshot_id,
                strategy=strategy,
                score=75.0,
                legs_json=[],
                suggestion_json={},
            )
        )
        conn.execute(
            insert(managed_opportunities).values(
                id=row_id,
                opportunity_key=f"key-{row_id}",
                signal_id=signal_id,
                session=session,
                symbol="SPY",
                direction="bull",
                setup_type="fvg_retest",
                strategy=strategy,
                strategy_score_id=score_id,
                structure_hash=f"structure-{row_id}",
                legs_json=[],
                features_json={
                    "feature_schema_version": feature_schema_version,
                    "snapshot_id": 999_999,
                    "suggestion": {"shadow_only": shadow_only},
                    "quality": {
                        "breakout": {"displacement": float(row_id)},
                        "vwap": {"direction_aligned": True},
                    },
                },
                policy_version=policy_version,
                decision_batch_id=f"scan-batch:{snapshot_id}",
                decision_score=75.0,
                decision_defined_risk=1,
                decision_max_loss=100.0,
                created_at=now,
                detected_at=now,
                baseline_action="hold",
                baseline_reason="shadow",
                admission_eligible=int(is_admission_eligible),
                shadow_only=int(shadow_only),
                bot_action="candidate" if is_runtime_candidate else "hold",
                bot_reason=(
                    "scan_admission_passed"
                    if is_runtime_candidate
                    else "not_runtime_admission_eligible"
                ),
                bot_decided_at=now + timedelta(minutes=decision_offset_minutes),
                decision_account_value_available=int(is_runtime_candidate),
                decision_account_value_usd=(
                    decision_account_value_usd if is_runtime_candidate else None
                ),
                session_close_at=now + timedelta(hours=2),
                entry_cutoff_at=now + timedelta(hours=1),
                timeout_at=timeout_at,
                entry_ts=(
                    now + timedelta(minutes=entry_offset_minutes)
                    if entry_offset_minutes is not None
                    else None
                ),
                entry_net=-1.0,
                basis_dollars=100.0,
                stop_pct=0.15,
                target_pct=0.225,
                commission_estimate=commission,
                status="resolved",
                outcome=outcome,
                resolved_at=resolved_at,
                exit_net=exit_net,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                valid_marks=10,
                training_eligible=1,
                resolution_reason="test",
            )
        )


def _model_samples() -> list[ManagedSample]:
    rows: list[ManagedSample] = []
    for index, outcome in enumerate(("target", "stop", "timeout") * 4, start=1):
        rows.append(
            ManagedSample(
                opportunity_id=index,
                opportunity_key=f"model-{index}",
                session=f"2026-08-{index:02d}",
                features={"signal": float(index % 3)},
                outcome=outcome,  # type: ignore[arg-type]
                basis_dollars=100.0,
                target_gain=22.5,
                stop_loss=15.0,
                timeout_gross_return=0.02 if outcome == "timeout" else 0.0,
                costs=1.4,
                realized_net_pnl={"target": 21.1, "stop": -16.4, "timeout": 0.6}[outcome],
            )
        )
    return rows


def _eligible_registry_metrics(artifact: ManagedModelArtifact) -> dict[str, object]:
    return {
        "eligible": True,
        "include_hermes_context": False,
        "model_role": BASE_MODEL_ROLE,
        "promotion_allowed": True,
        "deployment_scope": "paper_admission_candidate",
        "artifact_file": f"{artifact.model_version}.json",
        "admission_selection_policy": _default_admission_policy().payload(),
    }


def _default_admission_policy() -> AdmissionSelectionPolicy:
    return AdmissionSelectionPolicy(
        score_floor=0.0,
        single_trade_cap_pct=1.0,
        max_candidates_per_batch=3,
        max_admitted_per_session=3,
    )


def _evidence_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0.0 else "negative_infinity"
    if isinstance(value, dict):
        return {str(key): _evidence_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_evidence_safe(item) for item in value]
    return value


def _eligible_holdout_metrics(
    engine: Engine,
    artifact: ManagedModelArtifact,
    *,
    cohort: list[int] | None = None,
) -> dict[str, object]:
    cohort = cohort or [101, 102, 103]
    samples = {
        sample.opportunity_id: sample
        for sample in load_training_samples(
            engine,
            feature_schema_version=artifact.feature_schema_version,
            canonical_per_signal=False,
            include_shadow_structures=True,
        )
    }
    scored = score_frozen_artifact(
        artifact,
        [samples[row_id] for row_id in cohort if row_id in samples],
    )
    evaluation_policy: dict[str, int | float] = {
        "min_sessions": 2,
        "min_independent_signals": 3,
        "min_admitted": 2,
        "min_admitted_sessions": 2,
        "min_profit_factor": 1.0,
        "score_floor": 0.0,
        "single_trade_cap_pct": 1.0,
        "max_candidates_per_batch": 3,
        "max_admitted_per_session": 3,
        "bootstrap_iterations": 200,
    }
    report = evaluate_prospective_rows(
        scored,
        min_sessions=2,
        min_independent_signals=3,
        min_admitted=2,
        min_admitted_sessions=2,
        min_profit_factor=1.0,
        score_floor=0.0,
        single_trade_cap_pct=1.0,
        max_candidates_per_batch=3,
        max_admitted_per_session=3,
        bootstrap_iterations=200,
    )
    absolute = asdict(report)
    absolute.pop("rows", None)
    content_digest = (
        prospective_cohort_content_sha256(engine, cohort)
        if set(cohort).issubset(samples)
        else "0" * 64
    )
    return {
        "eligible": True,
        "model_version": artifact.model_version,
        "artifact_hash": artifact.artifact_hash,
        "feature_schema_version": artifact.feature_schema_version,
        "outcome_policy_version": artifact.outcome_policy_version,
        "trained_through_session": artifact.trained_through_session,
        "cohort_sha256": hashlib.sha256(
            "|".join(str(row_id) for row_id in cohort).encode()
        ).hexdigest(),
        "cohort_content_sha256": content_digest,
        "cohort_opportunity_ids": cohort,
        "evaluation_policy": evaluation_policy,
        "absolute": _evidence_safe(absolute),
        "incumbent": None,
        "incremental_policy": None,
        "promotion_allowed": True,
        "evaluation_protocol": PROSPECTIVE_HOLDOUT_PROTOCOL,
    }


def _seed_promoted_model(
    engine: Engine,
    artifact_dir: Path,
    *,
    model_version: str,
    malformed_evidence: str,
) -> tuple[ManagedModelArtifact, int]:
    artifact = fit_managed_model(
        _model_samples(),
        model_version=model_version,
        feature_schema_version="managed_capture_features_v1",
        outcome_policy_version="marketable_nbbo_15s_v1",
        ev_residuals=[1.0, 1.0, 1.0],
    )
    artifact_path(artifact_dir, model_version).write_text(artifact.to_json(), encoding="utf-8")
    for row_id, session in ((101, "2026-08-20"), (102, "2026-08-21"), (103, "2026-08-28")):
        if malformed_evidence == "missing_cohort" and row_id == 103:
            continue
        _insert_opportunity(
            engine,
            row_id=row_id,
            signal_id=f"holdout-signal-{row_id}",
            strategy=f"long_call_{row_id}",
            commission=1.0,
            outcome="target",
            net_pnl=30.0,
            session=(
                "2026-08-29" if malformed_evidence == "out_of_window" and row_id == 103 else session
            ),
            policy_version=(
                "wrong-policy-v1"
                if malformed_evidence == "wrong_policy" and row_id == 103
                else artifact.outcome_policy_version
            ),
            feature_schema_version=(
                "wrong-feature-schema-v1"
                if malformed_evidence == "wrong_schema" and row_id == 103
                else artifact.feature_schema_version
            ),
            shadow_only=(malformed_evidence == "research_only" and row_id == 103),
        )
    now = datetime(2026, 8, 30, tzinfo=UTC)
    registry_metrics = _eligible_registry_metrics(artifact)
    holdout_metrics = _eligible_holdout_metrics(engine, artifact)
    if malformed_evidence == "registry_ineligible":
        registry_metrics["eligible"] = False
    elif malformed_evidence == "holdout_identity":
        holdout_metrics["artifact_hash"] = "0" * 64
    elif malformed_evidence == "cohort_digest":
        holdout_metrics["cohort_sha256"] = "0" * 64
    elif malformed_evidence == "changed_label":
        holdout_metrics["cohort_content_sha256"] = "0" * 64
    elif malformed_evidence == "fabricated_metrics":
        absolute = holdout_metrics["absolute"]
        assert isinstance(absolute, dict)
        absolute["admitted_mean_pnl_lcb"] = 999_999.0
    elif malformed_evidence == "nonpositive_lcb":
        absolute = holdout_metrics["absolute"]
        assert isinstance(absolute, dict)
        absolute["admitted_mean_pnl_lcb"] = 0.0
    elif malformed_evidence == "selector_policy":
        evaluation_policy = holdout_metrics["evaluation_policy"]
        assert isinstance(evaluation_policy, dict)
        evaluation_policy["score_floor"] = 1.0
    with engine.begin() as conn:
        model_id = int(
            conn.execute(
                insert(managed_models).values(
                    model_version=model_version,
                    artifact_hash=artifact.artifact_hash,
                    feature_schema_version=artifact.feature_schema_version,
                    outcome_policy_version=artifact.outcome_policy_version,
                    trained_from_session=artifact.trained_from_session,
                    trained_through_session=artifact.trained_through_session,
                    metrics_json=registry_metrics,
                    status="promoted",
                    created_at=now,
                    promoted_at=now,
                )
            ).inserted_primary_key[0]
        )
        if malformed_evidence != "remove_holdout":
            conn.execute(
                insert(managed_model_evaluations).values(
                    model_id=model_id,
                    evaluation_kind="holdout",
                    fold_index=0,
                    train_from_session=artifact.trained_from_session,
                    train_through_session=artifact.trained_through_session,
                    test_from_session="2026-08-20",
                    test_through_session="2026-08-28",
                    metrics_json=holdout_metrics,
                    created_at=now,
                )
            )
    return artifact, model_id


def test_loader_propagates_pre_entry_context_across_signal_structures(
    tmp_db: Engine,
) -> None:
    _insert_opportunity(
        tmp_db,
        row_id=1,
        signal_id="signal-a",
        strategy="long_call",
        commission=3.0,
        outcome="target",
        net_pnl=22.0,
    )
    _insert_opportunity(
        tmp_db,
        row_id=2,
        signal_id="signal-a",
        strategy="bull_call_spread",
        commission=1.0,
        outcome="target",
        net_pnl=22.0,
    )
    now = datetime(2026, 8, 28, 14, 1, tzinfo=UTC)
    with tmp_db.begin() as conn:
        conn.execute(
            insert(managed_context_reviews).values(
                # Hermes reviewed only the base opportunity, but its contract
                # describes directional context for the shared signal.
                opportunity_id=1,
                received_at=now,
                timing="pretrade",
                response_json={},
                response_hash="context-1",
                context_probability=0.7,
                event_conflict=0,
                anomaly_json=["source_disagreement"],
                evidence_json=["news:1"],
                model_version="hermes-test",
                prompt_version="prompt-test",
            )
        )
    rows = load_training_samples(
        tmp_db,
        outcome_policy_version="marketable_nbbo_15s_v1",
        include_hermes_context=True,
    )
    assert len(rows) == 1
    assert rows[0].opportunity_id == 2
    assert "snapshot_id" not in rows[0].features
    assert rows[0].features["quality.breakout.displacement"] == 2.0
    assert rows[0].features["hermes.context_probability"] == 0.7
    assert rows[0].features["hermes.anomaly=source_disagreement"] == 1.0

    all_structures = load_training_samples(
        tmp_db,
        outcome_policy_version="marketable_nbbo_15s_v1",
        include_hermes_context=True,
        canonical_per_signal=False,
        include_shadow_structures=True,
    )
    assert len(all_structures) == 2
    assert {row.features["hermes.review_present"] for row in all_structures} == {1.0}
    assert {row.features["hermes.context_probability"] for row in all_structures} == {0.7}


def test_loader_excludes_context_not_strictly_before_shadow_entry(
    tmp_db: Engine,
) -> None:
    _insert_opportunity(
        tmp_db,
        row_id=3,
        signal_id="signal-late",
        strategy="long_call",
        commission=1.0,
        outcome="target",
        net_pnl=22.0,
        entry_offset_minutes=0,
    )
    received_at = datetime(2026, 8, 28, 14, 1, tzinfo=UTC)
    with tmp_db.begin() as conn:
        conn.execute(
            insert(managed_context_reviews).values(
                opportunity_id=3,
                received_at=received_at,
                # Deliberately simulate the coarse broker-relative bucket:
                # the loader must still enforce shadow-entry event time.
                timing="pretrade",
                response_json={},
                response_hash="context-late",
                context_probability=0.8,
                event_conflict=0,
                anomaly_json=[],
                evidence_json=["news:late"],
                model_version="hermes-test",
                prompt_version="prompt-test",
            )
        )

    rows = load_training_samples(
        tmp_db,
        outcome_policy_version="marketable_nbbo_15s_v1",
        include_hermes_context=True,
        canonical_per_signal=False,
        include_shadow_structures=True,
    )
    assert len(rows) == 1
    for row in rows:
        assert row.features["hermes.review_present"] == 0.0
        assert row.features["hermes.context_probability"] is None


def test_loader_excludes_backdated_context_before_detection(tmp_db: Engine) -> None:
    _insert_opportunity(
        tmp_db,
        row_id=4,
        signal_id="signal-backdated",
        strategy="long_call",
        commission=1.0,
        outcome="target",
        net_pnl=22.0,
    )
    with tmp_db.begin() as conn:
        conn.execute(
            insert(managed_context_reviews).values(
                opportunity_id=4,
                received_at=datetime(2026, 8, 28, 13, 59, tzinfo=UTC),
                timing="pretrade",
                response_json={},
                response_hash="context-backdated",
                context_probability=0.95,
                event_conflict=0,
                anomaly_json=[],
                evidence_json=["news:backdated"],
                model_version="hermes-test",
                prompt_version="prompt-test",
            )
        )

    rows = load_training_samples(
        tmp_db,
        include_hermes_context=True,
        canonical_per_signal=False,
    )
    assert len(rows) == 1
    assert rows[0].features["hermes.review_present"] == 0.0
    assert rows[0].features["hermes.context_probability"] is None


def test_loader_excludes_context_received_before_bot_decision(tmp_db: Engine) -> None:
    _insert_opportunity(
        tmp_db,
        row_id=11,
        signal_id="signal-before-decision",
        strategy="long_call",
        commission=1.0,
        outcome="target",
        net_pnl=22.0,
        decision_offset_minutes=1,
    )
    with tmp_db.begin() as conn:
        conn.execute(
            insert(managed_context_reviews).values(
                opportunity_id=11,
                received_at=datetime(2026, 8, 28, 14, 0, 30, tzinfo=UTC),
                timing="pretrade",
                response_json={},
                response_hash="context-before-decision",
                context_probability=0.9,
                event_conflict=0,
                anomaly_json=[],
                evidence_json=["news:before-decision"],
                model_version="hermes-test",
                prompt_version="prompt-test",
            )
        )

    rows = load_training_samples(
        tmp_db,
        include_hermes_context=True,
        canonical_per_signal=False,
    )
    assert len(rows) == 1
    assert rows[0].features["hermes.review_present"] == 0.0
    assert rows[0].features["hermes.context_probability"] is None


def test_production_loader_excludes_unexecutable_shadow_structures(
    tmp_db: Engine,
) -> None:
    _insert_opportunity(
        tmp_db,
        row_id=5,
        signal_id="signal-parity",
        strategy="long_call",
        commission=3.0,
        outcome="stop",
        net_pnl=-18.0,
    )
    _insert_opportunity(
        tmp_db,
        row_id=6,
        signal_id="signal-parity",
        strategy="bull_call_spread",
        commission=1.0,
        outcome="target",
        net_pnl=22.0,
        shadow_only=True,
    )
    _insert_opportunity(
        tmp_db,
        row_id=9,
        signal_id="signal-authoritative-ineligible",
        strategy="long_call",
        commission=1.0,
        outcome="target",
        net_pnl=22.0,
        admission_eligible=False,
    )
    _insert_opportunity(
        tmp_db,
        row_id=10,
        signal_id="signal-wrong-schema",
        strategy="long_call",
        commission=1.0,
        outcome="target",
        net_pnl=22.0,
        feature_schema_version="legacy-or-fabricated-schema",
    )

    production = load_training_samples(
        tmp_db,
        outcome_policy_version="marketable_nbbo_15s_v1",
        canonical_per_signal=False,
    )
    research = load_training_samples(
        tmp_db,
        outcome_policy_version="marketable_nbbo_15s_v1",
        canonical_per_signal=False,
        include_shadow_structures=True,
    )

    assert [row.opportunity_id for row in production] == [5]
    assert {row.opportunity_id for row in research} == {5, 6, 9}


def test_loader_preserves_label_when_decision_equity_was_unavailable(
    tmp_db: Engine,
) -> None:
    _insert_opportunity(
        tmp_db,
        row_id=12,
        signal_id="signal-equity-unavailable",
        strategy="long_call",
        commission=1.0,
        outcome="target",
        net_pnl=30.0,
        decision_account_value_usd=None,
    )

    rows = load_training_samples(tmp_db, canonical_per_signal=False)

    assert len(rows) == 1
    assert rows[0].decision_account_value_available is False
    assert rows[0].decision_account_value_usd is None


def test_sample_builder_rejects_corrupt_exit_gross_and_boundary_labels(
    tmp_db: Engine,
) -> None:
    _insert_opportunity(
        tmp_db,
        row_id=7,
        signal_id="signal-coherent-target",
        strategy="long_call",
        commission=1.0,
        outcome="target",
        net_pnl=30.0,
    )
    _insert_opportunity(
        tmp_db,
        row_id=8,
        signal_id="signal-coherent-stop",
        strategy="long_put",
        commission=1.0,
        outcome="stop",
        net_pnl=-20.0,
    )
    with tmp_db.connect() as conn:
        target = dict(
            conn.execute(select(managed_opportunities).where(managed_opportunities.c.id == 7))
            .mappings()
            .one()
        )
        stop = dict(
            conn.execute(select(managed_opportunities).where(managed_opportunities.c.id == 8))
            .mappings()
            .one()
        )
    assert _managed_sample_from_row(target) is not None  # type: ignore[arg-type]
    assert _managed_sample_from_row(stop) is not None  # type: ignore[arg-type]

    corrupt_exit = dict(target)
    corrupt_exit["exit_net"] = float(corrupt_exit["exit_net"]) + 0.01
    assert _managed_sample_from_row(corrupt_exit) is None  # type: ignore[arg-type]

    corrupt_gross = dict(target)
    corrupt_gross["gross_pnl"] = 30.0
    corrupt_gross["net_pnl"] = 29.0
    assert _managed_sample_from_row(corrupt_gross) is None  # type: ignore[arg-type]

    false_target = dict(target)
    false_target.update(exit_net=-1.10, gross_pnl=10.0, net_pnl=9.0)
    assert _managed_sample_from_row(false_target) is None  # type: ignore[arg-type]

    false_stop = dict(stop)
    false_stop.update(exit_net=-0.90, gross_pnl=-10.0, net_pnl=-11.0)
    assert _managed_sample_from_row(false_stop) is None  # type: ignore[arg-type]

    incomplete_account_evidence = dict(target)
    incomplete_account_evidence.update(
        decision_account_value_available=1,
        decision_account_value_usd=None,
    )
    assert _managed_sample_from_row(incomplete_account_evidence) is None  # type: ignore[arg-type]


def test_train_challenger_fails_closed_without_walk_forward_rows(
    tmp_db: Engine, tmp_path: Path
) -> None:
    run = train_challenger(
        tmp_db,
        tmp_path,
        model_version="empty-v1",
        feature_schema_version="managed_capture_features_v1",
        outcome_policy_version="marketable_nbbo_15s_v1",
    )
    assert run.status == "insufficient_data"
    assert run.artifact is None
    assert not artifact_path(tmp_path, "empty-v1").exists()


def test_promotion_verifies_registry_and_artifact_and_retires_prior(
    tmp_db: Engine, tmp_path: Path
) -> None:
    artifact = fit_managed_model(
        _model_samples(),
        model_version="candidate-v1",
        feature_schema_version="managed_capture_features_v1",
        outcome_policy_version="marketable_nbbo_15s_v1",
        ev_residuals=[1.0, 1.0, 1.0],
    )
    artifact_path(tmp_path, "candidate-v1").write_text(artifact.to_json(), encoding="utf-8")
    for row_id, session in ((101, "2026-08-20"), (102, "2026-08-21"), (103, "2026-08-28")):
        _insert_opportunity(
            tmp_db,
            row_id=row_id,
            signal_id=f"holdout-signal-{row_id}",
            strategy=f"long_call_{row_id}",
            commission=1.0,
            outcome="target",
            net_pnl=30.0,
            session=session,
        )
    holdout_metrics = _eligible_holdout_metrics(tmp_db, artifact)
    assert holdout_metrics["eligible"] is True
    assert isinstance(holdout_metrics["absolute"], dict)
    assert holdout_metrics["absolute"]["eligible"] is True
    now = datetime(2026, 8, 30, tzinfo=UTC)
    with tmp_db.begin() as conn:
        model_id = int(
            conn.execute(
                insert(managed_models).values(
                    model_version="candidate-v1",
                    artifact_hash=artifact.artifact_hash,
                    feature_schema_version=artifact.feature_schema_version,
                    outcome_policy_version=artifact.outcome_policy_version,
                    trained_from_session=artifact.trained_from_session,
                    trained_through_session=artifact.trained_through_session,
                    metrics_json=_eligible_registry_metrics(artifact),
                    status="challenger",
                    created_at=now,
                )
            ).inserted_primary_key[0]
        )
    with pytest.raises(ValueError, match="prospective holdout"):
        promote_challenger(
            tmp_db,
            tmp_path,
            "candidate-v1",
            paper_only=True,
            now=now,
        )
    with tmp_db.begin() as conn:
        conn.execute(
            insert(managed_model_evaluations).values(
                model_id=model_id,
                evaluation_kind="holdout",
                fold_index=0,
                train_from_session=artifact.trained_from_session,
                train_through_session=artifact.trained_through_session,
                test_from_session="2026-08-20",
                test_through_session="2026-08-28",
                metrics_json=holdout_metrics,
                created_at=now,
            )
        )
    promoted = promote_challenger(tmp_db, tmp_path, "candidate-v1", paper_only=True, now=now)
    assert promoted.artifact_hash == artifact.artifact_hash
    assert load_promoted_model(tmp_db, tmp_path) == artifact
    with tmp_db.connect() as conn:
        status = conn.execute(
            select(managed_models.c.status).where(managed_models.c.model_version == "candidate-v1")
        ).scalar_one()
    assert status == "promoted"


@pytest.mark.parametrize("tamper_incremental", [False, True])
def test_replacement_recomputes_frozen_incumbent_incremental_evidence(
    tmp_db: Engine,
    tmp_path: Path,
    tamper_incremental: bool,
) -> None:
    fitted = fit_managed_model(
        _model_samples(),
        model_version="replacement-template",
        feature_schema_version=MANAGED_FEATURE_SCHEMA_VERSION,
        outcome_policy_version="marketable_nbbo_15s_v1",
        ev_residuals=[0.0, 0.0, 0.0],
    )
    incumbent = replace(
        fitted,
        model_version="incumbent-v1",
        ev_residual_q05=-1.0,
        artifact_hash="",
    ).with_hash()
    challenger = replace(
        fitted,
        model_version="replacement-v2",
        ev_residual_q05=1.0,
        artifact_hash="",
    ).with_hash()
    artifact_path(tmp_path, incumbent.model_version).write_text(
        incumbent.to_json(), encoding="utf-8"
    )
    artifact_path(tmp_path, challenger.model_version).write_text(
        challenger.to_json(), encoding="utf-8"
    )
    for row_id, session in ((101, "2026-08-20"), (102, "2026-08-21"), (103, "2026-08-28")):
        _insert_opportunity(
            tmp_db,
            row_id=row_id,
            signal_id=f"replacement-signal-{row_id}",
            strategy=f"long_call_{row_id}",
            commission=1.0,
            outcome="target",
            net_pnl=30.0,
            session=session,
        )
    samples = load_training_samples(tmp_db, canonical_per_signal=False)
    incumbent_rows = score_frozen_artifact(incumbent, samples)
    challenger_rows = score_frozen_artifact(challenger, samples)
    incremental_policy = {
        "min_disagreements": 3,
        "min_sessions": 2,
        "bootstrap_iterations": 200,
        "score_floor": 0.0,
        "single_trade_cap_pct": 1.0,
        "max_candidates_per_batch": 3,
        "max_admitted_per_session": 3,
    }
    incremental = compare_context_incremental_value(
        incumbent_rows,
        challenger_rows,
        min_disagreements=3,
        min_sessions=2,
        bootstrap_iterations=200,
        score_floor=0.0,
        single_trade_cap_pct=1.0,
        max_candidates_per_batch=3,
        max_admitted_per_session=3,
    )
    assert incremental.eligible is True
    holdout_metrics = _eligible_holdout_metrics(tmp_db, challenger)
    incremental_metrics = _evidence_safe(asdict(incremental))
    assert isinstance(incremental_metrics, dict)
    if tamper_incremental:
        incremental_metrics["mean_incremental_pnl"] = 999_999.0
    holdout_metrics["incumbent"] = {
        "model_version": incumbent.model_version,
        "artifact_hash": incumbent.artifact_hash,
        "feature_schema_version": incumbent.feature_schema_version,
        "outcome_policy_version": incumbent.outcome_policy_version,
        "trained_through_session": incumbent.trained_through_session,
        "incremental_eligible": True,
        "incremental": incremental_metrics,
    }
    holdout_metrics["incremental_policy"] = incremental_policy
    now = datetime(2026, 8, 30, tzinfo=UTC)
    with tmp_db.begin() as conn:
        conn.execute(
            insert(managed_models).values(
                model_version=incumbent.model_version,
                artifact_hash=incumbent.artifact_hash,
                feature_schema_version=incumbent.feature_schema_version,
                outcome_policy_version=incumbent.outcome_policy_version,
                trained_from_session=incumbent.trained_from_session,
                trained_through_session=incumbent.trained_through_session,
                metrics_json=_eligible_registry_metrics(incumbent),
                status="promoted",
                created_at=now,
                promoted_at=now,
            )
        )
        challenger_id = int(
            conn.execute(
                insert(managed_models).values(
                    model_version=challenger.model_version,
                    artifact_hash=challenger.artifact_hash,
                    feature_schema_version=challenger.feature_schema_version,
                    outcome_policy_version=challenger.outcome_policy_version,
                    trained_from_session=challenger.trained_from_session,
                    trained_through_session=challenger.trained_through_session,
                    metrics_json=_eligible_registry_metrics(challenger),
                    status="challenger",
                    created_at=now,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(managed_model_evaluations).values(
                model_id=challenger_id,
                evaluation_kind="holdout",
                fold_index=0,
                train_from_session=challenger.trained_from_session,
                train_through_session=challenger.trained_through_session,
                test_from_session="2026-08-20",
                test_through_session="2026-08-28",
                metrics_json=holdout_metrics,
                created_at=now,
            )
        )

    if tamper_incremental:
        with pytest.raises(ValueError, match="replacement metrics differ"):
            promote_challenger(
                tmp_db,
                tmp_path,
                challenger.model_version,
                paper_only=True,
                now=now,
            )
    else:
        promoted = promote_challenger(
            tmp_db,
            tmp_path,
            challenger.model_version,
            paper_only=True,
            now=now,
        )
        assert promoted == challenger
        assert load_promoted_model(tmp_db, tmp_path) == challenger


def test_loader_rejects_manually_promoted_status_without_holdout(
    tmp_db: Engine, tmp_path: Path
) -> None:
    artifact = fit_managed_model(
        _model_samples(),
        model_version="manual-status-flip",
        feature_schema_version="managed_capture_features_v1",
        outcome_policy_version="marketable_nbbo_15s_v1",
        ev_residuals=[-2.0, -1.0, 1.0],
    )
    artifact_path(tmp_path, artifact.model_version).write_text(artifact.to_json(), encoding="utf-8")
    now = datetime(2026, 8, 29, tzinfo=UTC)
    with tmp_db.begin() as conn:
        model_id = int(
            conn.execute(
                insert(managed_models).values(
                    model_version=artifact.model_version,
                    artifact_hash=artifact.artifact_hash,
                    feature_schema_version=artifact.feature_schema_version,
                    outcome_policy_version=artifact.outcome_policy_version,
                    trained_from_session=artifact.trained_from_session,
                    trained_through_session=artifact.trained_through_session,
                    metrics_json=_eligible_registry_metrics(artifact),
                    status="challenger",
                    created_at=now,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            update(managed_models)
            .where(managed_models.c.id == model_id)
            .values(status="promoted", promoted_at=now)
        )

    with pytest.raises(ValueError, match="prospective holdout"):
        load_promoted_model(tmp_db, tmp_path)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("remove_holdout", "prospective holdout"),
        ("registry_ineligible", "eligible causal-base promotion metrics"),
        ("holdout_identity", "holdout identity"),
        ("cohort_digest", "cohort digest"),
        ("missing_cohort", "nonexistent opportunities"),
        ("wrong_policy", "outcome policy differs"),
        ("wrong_schema", "feature schema differs"),
        ("research_only", "research-only"),
        ("out_of_window", "outside its declared future range"),
        ("changed_label", "row-content digest"),
        ("fabricated_metrics", "absolute metrics differ"),
        ("nonpositive_lcb", "absolute metrics differ"),
        ("selector_policy", "selector differs from model registry"),
    ],
)
def test_loader_rechecks_immutable_promotion_evidence(
    tmp_db: Engine, tmp_path: Path, tamper: str, message: str
) -> None:
    _seed_promoted_model(
        tmp_db,
        tmp_path,
        model_version=f"tampered-{tamper}",
        malformed_evidence=tamper,
    )

    with pytest.raises(ValueError, match=message):
        load_promoted_model(tmp_db, tmp_path)


def test_loader_rejects_runtime_selector_policy_drift(
    tmp_db: Engine,
    tmp_path: Path,
) -> None:
    artifact, _model_id = _seed_promoted_model(
        tmp_db,
        tmp_path,
        model_version="runtime-selector-drift",
        malformed_evidence="",
    )
    policy = _default_admission_policy()

    assert load_promoted_model(
        tmp_db,
        tmp_path,
        expected_admission_policy=policy,
    ) == artifact
    with pytest.raises(ValueError, match="differs from runtime"):
        load_promoted_model(
            tmp_db,
            tmp_path,
            expected_admission_policy=replace(policy, score_floor=1.0),
        )


def test_context_challenger_cannot_be_promoted_even_when_eligible(
    tmp_db: Engine, tmp_path: Path
) -> None:
    context_samples = [
        replace(
            row,
            features={**row.features, "hermes.review_present": 1.0},
        )
        for row in _model_samples()
    ]
    artifact = fit_managed_model(
        context_samples,
        model_version="context-v1",
        feature_schema_version="managed_capture_features_v1+hermes_context_v1",
        outcome_policy_version="marketable_nbbo_15s_v1",
        ev_residuals=[-1.0, 0.0, 1.0],
    )
    artifact_path(tmp_path, "context-v1").write_text(artifact.to_json(), encoding="utf-8")
    now = datetime(2026, 8, 29, tzinfo=UTC)
    with tmp_db.begin() as conn:
        conn.execute(
            insert(managed_models).values(
                model_version="context-v1",
                artifact_hash=artifact.artifact_hash,
                feature_schema_version=artifact.feature_schema_version,
                outcome_policy_version=artifact.outcome_policy_version,
                trained_from_session=artifact.trained_from_session,
                trained_through_session=artifact.trained_through_session,
                metrics_json={
                    "eligible": True,
                    "include_hermes_context": True,
                    "model_role": CONTEXT_MODEL_ROLE,
                    "promotion_allowed": False,
                    "deployment_scope": "shadow_reporting_only",
                },
                status="challenger",
                created_at=now,
            )
        )

    with pytest.raises(ValueError, match="causal-base challenger"):
        promote_challenger(
            tmp_db,
            tmp_path,
            "context-v1",
            paper_only=True,
            now=now,
        )

    with tmp_db.connect() as conn:
        assert (
            conn.execute(
                select(managed_models.c.status).where(
                    managed_models.c.model_version == "context-v1"
                )
            ).scalar_one()
            == "challenger"
        )

    # Even a corrupted/legacy registry status cannot cross the live loader.
    with tmp_db.begin() as conn:
        conn.execute(
            update(managed_models)
            .where(managed_models.c.model_version == "context-v1")
            .values(status="promoted", promoted_at=now)
        )
    with pytest.raises(ValueError, match="eligible causal-base promotion metrics"):
        load_promoted_model(tmp_db, tmp_path)
