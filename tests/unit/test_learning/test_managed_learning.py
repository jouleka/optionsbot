from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.exc import IntegrityError

from optionsbot.config import Settings, StorageSettings
from optionsbot.daemon import managed_learning
from optionsbot.daemon.context import DaemonContext
from optionsbot.learning.managed_model import (
    ContextIncrementalReport,
    ManagedSample,
    PromotionReport,
    ProspectiveReport,
    WalkForwardMetrics,
    WalkForwardRow,
)
from optionsbot.learning.repository import (
    BASE_MODEL_ROLE,
    CONTEXT_MODEL_ROLE,
    TrainingRun,
)
from optionsbot.storage.schema import managed_model_evaluations, managed_models


def _sample() -> ManagedSample:
    return ManagedSample(
        opportunity_id=1,
        opportunity_key="opportunity-1",
        session="2026-08-28",
        features={"quality": 1.0},
        outcome="stop",
        basis_dollars=100.0,
        target_gain=22.5,
        stop_loss=15.0,
        timeout_gross_return=0.0,
        costs=1.30,
        realized_net_pnl=-16.30,
    )


def _report(*, eligible: bool, expected_value_lcb: float) -> PromotionReport:
    row = WalkForwardRow(
        opportunity_id=1,
        session="2026-08-28",
        outcome="stop",
        probabilities=(0.2, 0.7, 0.1),
        expected_value=-2.0,
        expected_value_lcb=expected_value_lcb,
        realized_net_pnl=-16.30,
    )
    metrics = WalkForwardMetrics(
        samples=1,
        sessions=1,
        independent_signals=1,
        folds=1,
        multiclass_brier=0.2,
        baseline_brier=0.3,
        log_loss=0.4,
        baseline_log_loss=0.5,
        calibration_error=0.1,
        admitted=1,
        admitted_sessions=1,
        admitted_net_pnl=-16.30,
        admitted_mean_pnl=-16.30,
        admitted_mean_pnl_lcb=-16.30,
        profit_factor=0.0,
        max_drawdown=16.30,
    )
    return PromotionReport(
        eligible=eligible,
        reasons=() if eligible else ("standalone_failed",),
        metrics=metrics,
        rows=(row,),
    )


def _context(
    engine: Engine,
    tmp_path: Path,
    *,
    auto_promote: bool = False,
) -> DaemonContext:
    settings = Settings(storage=StorageSettings(db_path=tmp_path / "settings.db"))
    settings.managed_learning.artifact_dir = tmp_path / "artifacts"
    settings.managed_learning.min_context_disagreements = 1
    settings.managed_learning.min_context_disagreement_sessions = 1
    settings.managed_learning.bootstrap_iterations = 200
    settings.managed_learning.auto_promote = auto_promote
    return cast(DaemonContext, SimpleNamespace(engine=engine, settings=settings))


def _insert_registration(
    engine: Engine,
    *,
    version: str,
    role: str,
    include_context: bool,
) -> int:
    with engine.begin() as conn:
        pk = conn.execute(
            insert(managed_models).values(
                model_version=version,
                artifact_hash=f"hash-{version}",
                feature_schema_version=(
                    "managed_capture_features_v1+hermes_context_v1"
                    if include_context
                    else "managed_capture_features_v1"
                ),
                outcome_policy_version="marketable_nbbo_15s_v1",
                trained_from_session="2026-08-01",
                trained_through_session="2026-08-28",
                metrics_json={
                    "eligible": True,
                    "include_hermes_context": include_context,
                    "model_role": role,
                    "promotion_allowed": not include_context,
                },
                status="challenger",
                created_at=datetime(2026, 8, 29, tzinfo=UTC),
            )
        ).inserted_primary_key
    assert pk is not None
    return int(pk[0])


def test_database_serializes_base_challenger_without_blocking_context(
    tmp_db: Engine,
) -> None:
    first_base_id = _insert_registration(
        tmp_db,
        version="managed-base-first",
        role=BASE_MODEL_ROLE,
        include_context=False,
    )
    _insert_registration(
        tmp_db,
        version="managed-context-paired",
        role=CONTEXT_MODEL_ROLE,
        include_context=True,
    )

    with pytest.raises(IntegrityError, match="UNIQUE constraint failed"):
        _insert_registration(
            tmp_db,
            version="managed-base-racing",
            role=BASE_MODEL_ROLE,
            include_context=False,
        )

    with tmp_db.begin() as conn:
        conn.execute(
            update(managed_models)
            .where(managed_models.c.id == first_base_id)
            .values(status="rejected")
        )
    _insert_registration(
        tmp_db,
        version="managed-base-racing",
        role=BASE_MODEL_ROLE,
        include_context=False,
    )

    with tmp_db.connect() as conn:
        challengers = (
            conn.execute(
                select(managed_models.c.metrics_json).where(managed_models.c.status == "challenger")
            )
            .scalars()
            .all()
        )
    assert sorted(row["model_role"] for row in challengers) == [
        BASE_MODEL_ROLE,
        CONTEXT_MODEL_ROLE,
    ]


def test_active_base_challenger_rejects_inconsistent_context_role(
    tmp_db: Engine,
    tmp_path: Path,
) -> None:
    context = _context(tmp_db, tmp_path)
    _insert_registration(
        tmp_db,
        version="managed-base-malformed",
        role=BASE_MODEL_ROLE,
        include_context=True,
    )

    with pytest.raises(RuntimeError, match="inconsistent Hermes-context"):
        managed_learning._active_base_challenger(context)


def test_existing_base_does_not_skip_missing_context_evaluation(
    tmp_db: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_db, tmp_path)
    base_version = "managed-base-20260828-n1"
    context_version = "managed-context-20260828-n1"
    _insert_registration(
        tmp_db,
        version=base_version,
        role=BASE_MODEL_ROLE,
        include_context=False,
    )
    base_report = _report(eligible=True, expected_value_lcb=1.0)
    context_report = _report(eligible=True, expected_value_lcb=-1.0)
    trained: list[str] = []

    monkeypatch.setattr(managed_learning, "load_training_samples", lambda *a, **k: [_sample()])

    def evaluate(*args: object, **kwargs: object) -> PromotionReport:
        assert kwargs["include_hermes_context"] is False
        return base_report

    def train(*args: object, **kwargs: Any) -> TrainingRun:
        assert kwargs["include_hermes_context"] is True
        trained.append(str(kwargs["model_version"]))
        model_id = _insert_registration(
            tmp_db,
            version=context_version,
            role=CONTEXT_MODEL_ROLE,
            include_context=True,
        )
        return TrainingRun(None, context_report, model_id, "challenger")

    monkeypatch.setattr(managed_learning, "evaluate_challenger", evaluate)
    monkeypatch.setattr(managed_learning, "train_challenger", train)

    summary = managed_learning._training_pass(
        context,
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert trained == [context_version]
    assert summary.base_model_version == base_version
    assert summary.context_model_version == context_version
    assert summary.context_eligible is True
    assert summary.context_incremental_eligible is True
    assert summary.context_reporting_eligible is True
    with tmp_db.connect() as conn:
        registrations = conn.execute(select(managed_models.c.model_version)).scalars().all()
        evidence = conn.execute(
            select(managed_model_evaluations).where(
                managed_model_evaluations.c.evaluation_kind == "paper_shadow"
            )
        ).one()
    assert registrations == [base_version, context_version]
    assert evidence.metrics_json["standalone_eligible"] is True
    assert evidence.metrics_json["context_incremental"]["eligible"] is True
    assert evidence.metrics_json["reporting_eligible"] is True
    assert evidence.metrics_json["promotion_allowed"] is False


def test_new_oof_challenger_is_frozen_not_promoted_on_same_history(
    tmp_db: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_db, tmp_path, auto_promote=True)
    base_report = _report(eligible=True, expected_value_lcb=1.0)
    context_report = _report(eligible=True, expected_value_lcb=-1.0)
    promoted: list[str] = []

    monkeypatch.setattr(managed_learning, "load_training_samples", lambda *a, **k: [_sample()])

    def train(*args: object, **kwargs: Any) -> TrainingRun:
        is_context = bool(kwargs.get("include_hermes_context", False))
        return TrainingRun(
            None,
            context_report if is_context else base_report,
            2 if is_context else 1,
            "challenger",
        )

    def record(*args: object, **kwargs: object) -> bool:
        return True

    def promote(
        engine: Engine,
        artifact_dir: Path,
        model_version: str,
        **kwargs: object,
    ) -> None:
        promoted.append(model_version)

    monkeypatch.setattr(managed_learning, "train_challenger", train)
    monkeypatch.setattr(managed_learning, "_record_context_incremental", record)
    monkeypatch.setattr(managed_learning, "promote_challenger", promote)

    summary = managed_learning._training_pass(
        context,
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert promoted == []
    assert summary.promoted_model_version is None
    assert summary.base_model_version == "managed-base-20260828-n1"
    assert summary.context_reporting_eligible is True


def test_active_challenger_waits_for_strictly_future_block(
    tmp_db: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_db, tmp_path, auto_promote=True)
    version = "managed-base-awaiting"
    _insert_registration(
        tmp_db,
        version=version,
        role=BASE_MODEL_ROLE,
        include_context=False,
    )
    promoted: list[str] = []
    monkeypatch.setattr(
        managed_learning,
        "load_training_samples",
        lambda *args, **kwargs: [_sample()],
    )
    monkeypatch.setattr(
        managed_learning,
        "promote_challenger",
        lambda *args, **kwargs: promoted.append(str(args[2])),
    )

    summary = managed_learning._training_pass(
        context,
        now=datetime(2026, 8, 30, tzinfo=UTC),
    )

    assert summary.status == "awaiting_prospective_holdout"
    assert summary.base_model_version == version
    assert promoted == []


def test_frozen_challenger_promotes_once_future_block_passes(
    tmp_db: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_db, tmp_path, auto_promote=True)
    context.settings.managed_learning.prospective_min_sessions = 2
    context.settings.managed_learning.prospective_min_independent_signals = 3
    context.settings.managed_learning.prospective_min_admitted = 2
    context.settings.managed_learning.prospective_min_admitted_sessions = 2
    version = "managed-base-frozen"
    model_id = _insert_registration(
        tmp_db,
        version=version,
        role=BASE_MODEL_ROLE,
        include_context=False,
    )
    future = [
        ManagedSample(
            opportunity_id=index,
            opportunity_key=f"future-{index}",
            session="2026-08-29" if index < 3 else "2026-08-30",
            features={"quality": float(index)},
            outcome="target",
            basis_dollars=100.0,
            target_gain=22.5,
            stop_loss=15.0,
            timeout_gross_return=0.0,
            costs=1.3,
            realized_net_pnl=20.0,
            signal_id=f"future-signal-{index}",
        )
        for index in range(1, 5)
    ]
    rows = tuple(
        WalkForwardRow(
            opportunity_id=sample.opportunity_id,
            session=sample.session,
            outcome=sample.outcome,
            probabilities=(0.8, 0.1, 0.1),
            expected_value=10.0,
            expected_value_lcb=5.0,
            realized_net_pnl=sample.realized_net_pnl,
            signal_id=sample.signal_id,
        )
        for sample in future
    )
    report = ProspectiveReport(
        eligible=True,
        samples=4,
        independent_signals=4,
        sessions=2,
        admitted=4,
        admitted_sessions=2,
        admitted_net_pnl=80.0,
        admitted_mean_pnl=20.0,
        admitted_mean_pnl_lcb=20.0,
        profit_factor=float("inf"),
        max_drawdown=0.0,
        reasons=(),
        rows=rows,
    )
    promoted: list[str] = []
    monkeypatch.setattr(
        managed_learning,
        "load_training_samples",
        lambda *args, **kwargs: future,
    )
    monkeypatch.setattr(managed_learning, "_registered_artifact", lambda *args: object())
    monkeypatch.setattr(managed_learning, "score_frozen_artifact", lambda *args: rows)
    monkeypatch.setattr(
        managed_learning, "evaluate_prospective_rows", lambda *args, **kwargs: report
    )
    monkeypatch.setattr(
        managed_learning,
        "load_promoted_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        managed_learning,
        "prospective_cohort_content_sha256",
        lambda *args: "a" * 64,
    )
    monkeypatch.setattr(
        managed_learning,
        "promote_challenger",
        lambda *args, **kwargs: promoted.append(str(args[2])),
    )

    summary = managed_learning._training_pass(
        context,
        now=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert summary.status == "promoted"
    assert promoted == [version]
    with tmp_db.connect() as conn:
        evidence = conn.execute(
            select(managed_model_evaluations).where(
                managed_model_evaluations.c.model_id == model_id
            )
        ).one()
    assert evidence.evaluation_kind == "holdout"
    assert evidence.metrics_json["eligible"] is True
    assert evidence.metrics_json["evaluation_protocol"].startswith("serial_frozen")


def test_context_reporting_requires_standalone_and_incremental_evidence() -> None:
    incremental_pass = ContextIncrementalReport(
        eligible=True,
        disagreements=10,
        disagreement_sessions=5,
        incremental_net_pnl=50.0,
        mean_incremental_pnl=5.0,
        mean_incremental_pnl_lcb=1.0,
        reasons=(),
    )
    incremental_fail = ContextIncrementalReport(
        eligible=False,
        disagreements=1,
        disagreement_sessions=1,
        incremental_net_pnl=5.0,
        mean_incremental_pnl=5.0,
        mean_incremental_pnl_lcb=-1.0,
        reasons=("incremental_failed",),
    )
    standalone_pass = TrainingRun(
        None,
        _report(eligible=True, expected_value_lcb=-1.0),
        1,
        "challenger",
    )
    standalone_fail = TrainingRun(
        None,
        _report(eligible=False, expected_value_lcb=-1.0),
        1,
        "rejected",
    )

    assert (
        managed_learning._context_evaluation_metrics(standalone_pass, incremental_pass)[
            "reporting_eligible"
        ]
        is True
    )
    assert (
        managed_learning._context_evaluation_metrics(standalone_fail, incremental_pass)[
            "reporting_eligible"
        ]
        is False
    )
    assert (
        managed_learning._context_evaluation_metrics(standalone_pass, incremental_fail)[
            "reporting_eligible"
        ]
        is False
    )
