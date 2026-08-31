from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

import optionsbot.learning.managed_model as managed_model
from optionsbot.learning.managed_model import (
    ManagedModelArtifact,
    ManagedSample,
    PromotionPolicy,
    WalkForwardRow,
    compare_context_incremental_value,
    evaluate_prospective_rows,
    fit_managed_model,
    predict_managed_outcome,
    walk_forward_evaluate,
)


def _selector_row(
    opportunity_id: int,
    *,
    batch: str,
    detected_at: datetime,
    expected_value_lcb: float,
    realized_net_pnl: float,
    signal_id: str | None = None,
    score: float = 80.0,
    defined_risk: bool = True,
    max_loss: float | None = 100.0,
    account_available: bool = True,
    account_value: float | None = 10_000.0,
) -> WalkForwardRow:
    return WalkForwardRow(
        opportunity_id=opportunity_id,
        session=detected_at.date().isoformat(),
        outcome="target" if realized_net_pnl >= 0.0 else "stop",
        probabilities=(0.6, 0.2, 0.2),
        expected_value=expected_value_lcb + 1.0,
        expected_value_lcb=expected_value_lcb,
        realized_net_pnl=realized_net_pnl,
        signal_id=signal_id or f"signal-{opportunity_id}",
        decision_batch_id=batch,
        detected_at=detected_at,
        decision_score=score,
        decision_defined_risk=defined_risk,
        decision_max_loss=max_loss,
        decision_account_value_available=account_available,
        decision_account_value_usd=account_value,
    )


def _selector_report(
    rows: list[WalkForwardRow],
    *,
    daily_cap: int = 3,
    batch_cap: int = 3,
) -> managed_model.ProspectiveReport:
    return evaluate_prospective_rows(
        rows,
        min_sessions=1,
        min_independent_signals=1,
        min_admitted=1,
        min_admitted_sessions=1,
        min_profit_factor=0.0,
        max_admitted_per_session=daily_cap,
        max_candidates_per_batch=batch_cap,
        bootstrap_iterations=1,
        score_floor=60.0,
        single_trade_cap_pct=0.10,
    )


def _samples(session_count: int = 45, per_session: int = 4) -> list[ManagedSample]:
    rows: list[ManagedSample] = []
    start = date(2026, 1, 2)
    outcome_cycle = ("target", "stop", "timeout")
    identifier = 1
    for session_offset in range(session_count):
        session = (start + timedelta(days=session_offset)).isoformat()
        for within in range(per_session):
            outcome = outcome_cycle[(session_offset + within) % 3]
            # Features deliberately carry a strong, learnable pre-trade event
            # signature. Missing RVOL exercises the encoder's missingness path.
            signature = {"target": 2.0, "stop": -2.0, "timeout": 0.0}[outcome]
            realized = {"target": 22.0, "stop": -15.0, "timeout": 2.0}[outcome]
            rows.append(
                ManagedSample(
                    opportunity_id=identifier,
                    opportunity_key=f"op-{identifier}",
                    session=session,
                    features={
                        "displacement": signature + within * 0.01,
                        "rejection": signature,
                        "rvol": None if within == 0 else signature + 3.0,
                    },
                    outcome=outcome,  # type: ignore[arg-type]
                    basis_dollars=100.0,
                    target_gain=22.5,
                    stop_loss=15.0,
                    timeout_gross_return=0.025 if outcome == "timeout" else 0.0,
                    costs=0.5,
                    realized_net_pnl=realized,
                )
            )
            identifier += 1
    return rows


def test_artifact_is_plain_json_checksummed_and_predicts_three_events() -> None:
    rows = _samples(20)
    artifact = fit_managed_model(
        rows,
        model_version="managed-test-v1",
        feature_schema_version="quality-v1",
        outcome_policy_version="path-v1",
        ev_residuals=[-2.0, -1.0, 0.0, 1.0],
    )
    raw = artifact.to_json()
    assert "managed-test-v1" in raw
    assert json.loads(raw)["artifact_hash"] == artifact.artifact_hash
    restored = ManagedModelArtifact.from_json(raw)
    prediction = predict_managed_outcome(
        restored,
        {"displacement": 2.0, "rejection": 2.0, "rvol": 5.0},
        basis_dollars=100.0,
        target_gain=22.5,
        stop_loss=15.0,
        costs=0.5,
    )
    assert prediction.target_probability > prediction.stop_probability
    assert sum(
        (
            prediction.target_probability,
            prediction.stop_probability,
            prediction.timeout_probability,
        )
    ) == pytest.approx(1.0)
    assert prediction.expected_value_lcb < prediction.expected_value


def test_artifact_tampering_is_rejected() -> None:
    artifact = fit_managed_model(
        _samples(10),
        model_version="v1",
        feature_schema_version="f1",
        outcome_policy_version="o1",
        ev_residuals=[-1.0],
    )
    payload = json.loads(artifact.to_json())
    payload["temperature"] = 9.0
    with pytest.raises(ValueError, match="checksum"):
        ManagedModelArtifact.from_json(json.dumps(payload))


def test_missing_residual_evidence_forces_negative_infinite_lcb() -> None:
    artifact = fit_managed_model(
        _samples(10),
        model_version="v1",
        feature_schema_version="f1",
        outcome_policy_version="o1",
    )
    prediction = predict_managed_outcome(
        artifact,
        {"displacement": 2.0},
        basis_dollars=100.0,
        target_gain=22.5,
        stop_loss=15.0,
        costs=0.5,
    )
    assert prediction.expected_value_lcb == float("-inf")


def test_residual_lower_bound_scales_as_return_not_historical_dollars() -> None:
    artifact = fit_managed_model(
        _samples(10),
        model_version="v1",
        feature_schema_version="f1",
        outcome_policy_version="o1",
        ev_residuals=[-0.05],
    )
    small = predict_managed_outcome(
        artifact,
        {"displacement": 2.0},
        basis_dollars=100.0,
        target_gain=22.5,
        stop_loss=15.0,
        costs=0.5,
    )
    large = predict_managed_outcome(
        artifact,
        {"displacement": 2.0},
        basis_dollars=200.0,
        target_gain=45.0,
        stop_loss=30.0,
        costs=1.0,
    )
    assert small.expected_value - small.expected_value_lcb == pytest.approx(5.0)
    assert large.expected_value - large.expected_value_lcb == pytest.approx(10.0)


def test_residual_tail_weights_each_signal_once() -> None:
    rows = _samples(10)
    base = fit_managed_model(
        rows,
        model_version="base",
        feature_schema_version="f1",
        outcome_policy_version="o1",
        ev_residuals=[-0.20, 0.10],
        ev_residual_groups=["loser", "winner"],
    )
    duplicated = fit_managed_model(
        rows,
        model_version="duplicated",
        feature_schema_version="f1",
        outcome_policy_version="o1",
        ev_residuals=[-0.20, *([0.10] * 20)],
        ev_residual_groups=["loser", *(["winner"] * 20)],
    )

    assert base.ev_residual_q05 == -0.20
    assert duplicated.ev_residual_q05 == base.ev_residual_q05


def test_walk_forward_is_session_grouped_and_can_promote_learnable_data() -> None:
    report = walk_forward_evaluate(
        _samples(),
        policy=PromotionPolicy(
            min_sessions=30,
            min_samples=100,
            min_oof_samples=50,
            min_folds=10,
            min_admitted=20,
            min_admitted_sessions=10,
            min_profit_factor=1.01,
            bootstrap_iterations=400,
        ),
        min_train_sessions=15,
        embargo_sessions=1,
    )
    assert report.metrics.folds >= 10
    assert report.metrics.multiclass_brier < report.metrics.baseline_brier
    assert report.metrics.log_loss < report.metrics.baseline_log_loss
    assert report.metrics.admitted_mean_pnl_lcb > 0.0
    assert report.eligible, report.reasons


def test_walk_forward_rejects_duplicate_opportunity_trials() -> None:
    rows = _samples(5)
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate opportunity"):
        walk_forward_evaluate(rows)


def test_residual_calibration_respects_walk_forward_embargo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = managed_model.fit_managed_model
    residual_counts: list[int] = []

    def recording_fit(
        rows: list[ManagedSample],
        **kwargs: object,
    ) -> ManagedModelArtifact:
        residual_counts.append(len(kwargs.get("ev_residuals", ())))  # type: ignore[arg-type]
        return original_fit(rows, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(managed_model, "fit_managed_model", recording_fit)
    managed_model.walk_forward_evaluate(
        _samples(session_count=8, per_session=3),
        min_train_sessions=3,
        embargo_sessions=1,
    )

    # Session four's residuals cannot enter session five's model because that
    # session is the one-period embargo. They become eligible one fold later.
    assert residual_counts == [0, 0, 3, 6]


def test_small_dataset_fails_closed_with_explicit_reasons() -> None:
    report = walk_forward_evaluate(_samples(8), min_train_sessions=3)
    assert not report.eligible
    assert any(reason.startswith("sessions_below_minimum") for reason in report.reasons)
    assert any(reason.startswith("samples_below_minimum") for reason in report.reasons)


def test_many_correlated_structures_do_not_satisfy_independent_signal_minimum() -> None:
    rows = [
        replace(row, signal_id=f"session-signal:{row.session}")
        for row in _samples(session_count=45, per_session=4)
    ]
    report = walk_forward_evaluate(rows, min_train_sessions=15)
    assert report.metrics.independent_signals < report.metrics.samples
    assert any(reason.startswith("independent_signals_below_minimum") for reason in report.reasons)


def test_correlated_duplicate_does_not_change_feature_normalization() -> None:
    rows = [
        replace(row, signal_id=f"signal:{row.opportunity_id}")
        for row in _samples(session_count=10, per_session=1)
    ]
    duplicate = replace(
        rows[0],
        opportunity_id=10_001,
        opportunity_key="duplicate-structure",
    )
    base = fit_managed_model(
        rows,
        model_version="base",
        feature_schema_version="f1",
        outcome_policy_version="o1",
        ev_residuals=[-0.1, 0.0, 0.1],
    )
    duplicated = fit_managed_model(
        [*rows, duplicate],
        model_version="duplicated",
        feature_schema_version="f1",
        outcome_policy_version="o1",
        ev_residuals=[-0.1, 0.0, 0.1],
    )

    assert duplicated.encoder == base.encoder
    assert duplicated.weights == base.weights
    assert duplicated.temperature == base.temperature


def test_context_gets_credit_only_for_profitable_disagreements() -> None:
    baseline: list[WalkForwardRow] = []
    context: list[WalkForwardRow] = []
    for index in range(30):
        session = f"2026-02-{index // 2 + 1:02d}"
        pnl = 10.0
        baseline.append(
            WalkForwardRow(
                opportunity_id=index + 1,
                session=session,
                outcome="target",
                probabilities=(0.6, 0.2, 0.2),
                expected_value=1.0,
                expected_value_lcb=-1.0,
                realized_net_pnl=pnl,
            )
        )
        context.append(
            WalkForwardRow(
                opportunity_id=index + 1,
                session=session,
                outcome="target",
                probabilities=(0.7, 0.15, 0.15),
                expected_value=3.0,
                expected_value_lcb=1.0,
                realized_net_pnl=pnl,
            )
        )
    report = compare_context_incremental_value(baseline, context, bootstrap_iterations=200)
    assert report.eligible
    assert report.disagreements == 30
    assert report.incremental_net_pnl == 300.0


def test_context_agreement_receives_no_causal_credit() -> None:
    row = WalkForwardRow(
        opportunity_id=1,
        session="2026-02-01",
        outcome="target",
        probabilities=(0.7, 0.2, 0.1),
        expected_value=5.0,
        expected_value_lcb=1.0,
        realized_net_pnl=100.0,
    )
    report = compare_context_incremental_value([row], [row], min_disagreements=1, min_sessions=1)
    assert report.disagreements == 0
    assert report.incremental_net_pnl == 0.0
    assert not report.eligible


def test_selector_ranks_each_batch_by_conservative_ev_per_defined_risk() -> None:
    detected_at = datetime(2026, 2, 2, 14, 30, tzinfo=UTC)
    expensive = _selector_row(
        1,
        batch="scan-1",
        detected_at=detected_at,
        expected_value_lcb=50.0,
        max_loss=1_000.0,
        realized_net_pnl=-100.0,
    )
    efficient = _selector_row(
        2,
        batch="scan-1",
        detected_at=detected_at,
        expected_value_lcb=20.0,
        max_loss=100.0,
        realized_net_pnl=20.0,
    )

    report = _selector_report([expensive, efficient], daily_cap=1, batch_cap=1)

    assert report.admitted == 1
    assert report.admitted_net_pnl == 20.0


def test_selector_replays_batches_causally_without_later_replacement() -> None:
    early = _selector_row(
        10,
        batch="scan-early",
        detected_at=datetime(2026, 2, 3, 14, 30, tzinfo=UTC),
        expected_value_lcb=1.0,
        realized_net_pnl=5.0,
    )
    later = _selector_row(
        1,
        batch="scan-later",
        detected_at=datetime(2026, 2, 3, 14, 35, tzinfo=UTC),
        expected_value_lcb=100.0,
        realized_net_pnl=-50.0,
    )

    report = _selector_report([later, early], daily_cap=1, batch_cap=1)

    assert report.admitted == 1
    assert report.admitted_net_pnl == 5.0


def test_selector_excludes_below_floor_undefined_and_unaffordable_rows() -> None:
    detected_at = datetime(2026, 2, 4, 14, 30, tzinfo=UTC)
    rows = [
        _selector_row(
            1,
            batch="scan-1",
            detected_at=detected_at,
            expected_value_lcb=10.0,
            realized_net_pnl=-1.0,
            score=59.0,
        ),
        _selector_row(
            2,
            batch="scan-1",
            detected_at=detected_at,
            expected_value_lcb=10.0,
            realized_net_pnl=-2.0,
            defined_risk=False,
            max_loss=None,
        ),
        _selector_row(
            3,
            batch="scan-1",
            detected_at=detected_at,
            expected_value_lcb=10.0,
            realized_net_pnl=-3.0,
            account_available=False,
            account_value=None,
        ),
        _selector_row(
            4,
            batch="scan-1",
            detected_at=detected_at,
            expected_value_lcb=10.0,
            realized_net_pnl=-4.0,
            max_loss=1_001.0,
        ),
        _selector_row(
            5,
            batch="scan-1",
            detected_at=detected_at,
            expected_value_lcb=10.0,
            realized_net_pnl=7.0,
            max_loss=100.0,
        ),
    ]

    report = _selector_report(rows)

    assert report.samples == 5
    assert report.admitted == 1
    assert report.admitted_net_pnl == 7.0


def test_selector_admits_only_one_structure_per_signal_in_a_batch() -> None:
    detected_at = datetime(2026, 2, 5, 14, 30, tzinfo=UTC)
    rows = [
        _selector_row(
            1,
            batch="scan-1",
            detected_at=detected_at,
            signal_id="one-thesis",
            expected_value_lcb=5.0,
            max_loss=100.0,
            realized_net_pnl=-10.0,
        ),
        _selector_row(
            2,
            batch="scan-1",
            detected_at=detected_at,
            signal_id="one-thesis",
            expected_value_lcb=20.0,
            max_loss=100.0,
            realized_net_pnl=15.0,
        ),
    ]

    report = _selector_report(rows)

    assert report.admitted == 1
    assert report.admitted_net_pnl == 15.0
