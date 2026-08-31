"""Post-session managed-outcome challenger training and guarded promotion."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError

from optionsbot.admission_policy import configured_admission_policy
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.learning.managed_model import (
    ContextIncrementalReport,
    ManagedModelArtifact,
    ManagedSample,
    PromotionPolicy,
    ProspectiveReport,
    compare_context_incremental_value,
    evaluate_prospective_rows,
    score_frozen_artifact,
)
from optionsbot.learning.repository import (
    BASE_MODEL_ROLE,
    PROSPECTIVE_HOLDOUT_PROTOCOL,
    TrainingRun,
    artifact_path,
    evaluate_challenger,
    load_promoted_model,
    load_training_samples,
    promote_challenger,
    prospective_cohort_content_sha256,
    train_challenger,
)
from optionsbot.storage.schema import (
    managed_model_evaluations,
    managed_models,
    managed_opportunities,
)


@dataclass(frozen=True, slots=True)
class ManagedLearningSummary:
    status: str
    samples: int
    sessions: int
    base_model_version: str | None = None
    context_model_version: str | None = None
    promoted_model_version: str | None = None
    base_eligible: bool = False
    context_eligible: bool = False
    context_incremental_eligible: bool = False
    context_reporting_eligible: bool = False


@dataclass(frozen=True, slots=True)
class _ModelRegistration:
    model_id: int
    model_version: str
    artifact_hash: str
    feature_schema_version: str
    outcome_policy_version: str
    trained_from_session: str
    trained_through_session: str
    status: str
    metrics: Mapping[str, object]


def _policy(context: DaemonContext) -> PromotionPolicy:
    config = context.settings.managed_learning
    selection = configured_admission_policy(context.settings)
    return PromotionPolicy(
        min_sessions=config.min_sessions,
        min_samples=config.min_samples,
        min_independent_signals=config.min_independent_signals,
        min_oof_samples=config.min_oof_samples,
        min_folds=config.min_folds,
        min_admitted=config.min_admitted,
        min_admitted_sessions=config.min_admitted_sessions,
        min_profit_factor=config.min_profit_factor,
        score_floor=selection.score_floor,
        single_trade_cap_pct=selection.single_trade_cap_pct,
        max_candidates_per_batch=selection.max_candidates_per_batch,
        max_admitted_per_session=selection.max_admitted_per_session,
        bootstrap_iterations=config.bootstrap_iterations,
    )


def _registration(context: DaemonContext, model_version: str) -> _ModelRegistration | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(
                managed_models.c.id,
                managed_models.c.model_version,
                managed_models.c.artifact_hash,
                managed_models.c.feature_schema_version,
                managed_models.c.outcome_policy_version,
                managed_models.c.trained_from_session,
                managed_models.c.trained_through_session,
                managed_models.c.status,
                managed_models.c.metrics_json,
            ).where(managed_models.c.model_version == model_version)
        ).one_or_none()
    if row is None:
        return None
    metrics = row.metrics_json if isinstance(row.metrics_json, Mapping) else {}
    return _ModelRegistration(
        model_id=int(row.id),
        model_version=str(row.model_version),
        artifact_hash=str(row.artifact_hash),
        feature_schema_version=str(row.feature_schema_version),
        outcome_policy_version=str(row.outcome_policy_version),
        trained_from_session=str(row.trained_from_session),
        trained_through_session=str(row.trained_through_session),
        status=str(row.status),
        metrics=metrics,
    )


def _active_base_challenger(context: DaemonContext) -> _ModelRegistration | None:
    with context.engine.connect() as conn:
        rows = conn.execute(
            select(
                managed_models.c.id,
                managed_models.c.model_version,
                managed_models.c.artifact_hash,
                managed_models.c.feature_schema_version,
                managed_models.c.outcome_policy_version,
                managed_models.c.trained_from_session,
                managed_models.c.trained_through_session,
                managed_models.c.status,
                managed_models.c.metrics_json,
            )
            .where(managed_models.c.status == "challenger")
            .order_by(managed_models.c.id)
        ).all()
    base_rows = [
        row
        for row in rows
        if isinstance(row.metrics_json, Mapping)
        and row.metrics_json.get("model_role") == BASE_MODEL_ROLE
    ]
    if len(base_rows) > 1:
        raise RuntimeError("multiple active causal-base challengers violate serial testing")
    if not base_rows:
        return None
    row = base_rows[0]
    if row.metrics_json.get("include_hermes_context") is not False:
        raise RuntimeError(
            "causal-base challenger has inconsistent Hermes-context registry metadata"
        )
    return _ModelRegistration(
        model_id=int(row.id),
        model_version=str(row.model_version),
        artifact_hash=str(row.artifact_hash),
        feature_schema_version=str(row.feature_schema_version),
        outcome_policy_version=str(row.outcome_policy_version),
        trained_from_session=str(row.trained_from_session),
        trained_through_session=str(row.trained_through_session),
        status=str(row.status),
        metrics=row.metrics_json,
    )


def _stored_holdout_evidence(context: DaemonContext, model_id: int) -> Mapping[str, object] | None:
    with context.engine.connect() as conn:
        metrics = conn.execute(
            select(managed_model_evaluations.c.metrics_json)
            .where(managed_model_evaluations.c.model_id == model_id)
            .where(managed_model_evaluations.c.evaluation_kind == "holdout")
            .where(managed_model_evaluations.c.fold_index == 0)
        ).scalar_one_or_none()
    return metrics if isinstance(metrics, Mapping) else None


def _prospective_block(
    samples: list[ManagedSample],
    *,
    trained_through_session: str,
    min_sessions: int,
    min_independent_signals: int,
) -> list[ManagedSample]:
    """Freeze the shortest complete future-session prefix meeting the gate."""
    future = [sample for sample in samples if sample.session > trained_through_session]
    by_session: dict[str, list[ManagedSample]] = {}
    for sample in future:
        by_session.setdefault(sample.session, []).append(sample)
    selected: list[ManagedSample] = []
    signals: set[str] = set()
    for session in sorted(by_session):
        session_rows = sorted(by_session[session], key=lambda row: row.opportunity_id)
        selected.extend(session_rows)
        signals.update(row.signal_id or row.opportunity_key for row in session_rows)
        if (
            len({row.session for row in selected}) >= min_sessions
            and len(signals) >= min_independent_signals
        ):
            return selected
    return []


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0.0 else "negative_infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _prospective_payload(
    registration: _ModelRegistration,
    report: ProspectiveReport,
    *,
    incumbent: ManagedModelArtifact | None,
    incremental: ContextIncrementalReport | None,
    cohort_content_sha256: str,
    evaluation_policy: Mapping[str, int | float],
    incremental_policy: Mapping[str, int | float] | None,
) -> dict[str, object]:
    absolute = asdict(report)
    absolute.pop("rows", None)
    cohort_identity = "|".join(str(row.opportunity_id) for row in report.rows)
    incumbent_payload: dict[str, object] | None = None
    if incumbent is not None:
        if incremental is None:
            raise ValueError("incumbent replacement requires incremental evidence")
        incumbent_payload = {
            "model_version": incumbent.model_version,
            "artifact_hash": incumbent.artifact_hash,
            "feature_schema_version": incumbent.feature_schema_version,
            "outcome_policy_version": incumbent.outcome_policy_version,
            "trained_through_session": incumbent.trained_through_session,
            "incremental_eligible": incremental.eligible,
            "incremental": _incremental_payload(incremental),
        }
    eligible = report.eligible and (
        incumbent is None or (incremental is not None and incremental.eligible)
    )
    return {
        "eligible": eligible,
        "model_version": registration.model_version,
        "artifact_hash": registration.artifact_hash,
        "feature_schema_version": registration.feature_schema_version,
        "outcome_policy_version": registration.outcome_policy_version,
        "trained_through_session": registration.trained_through_session,
        "cohort_sha256": hashlib.sha256(cohort_identity.encode()).hexdigest(),
        "cohort_content_sha256": cohort_content_sha256,
        "cohort_opportunity_ids": [row.opportunity_id for row in report.rows],
        "evaluation_policy": dict(evaluation_policy),
        "incremental_policy": (
            dict(incremental_policy) if incremental_policy is not None else None
        ),
        "absolute": _json_safe(absolute),
        "incumbent": incumbent_payload,
        "promotion_allowed": True,
        "evaluation_protocol": PROSPECTIVE_HOLDOUT_PROTOCOL,
    }


def _stored_context_evidence(context: DaemonContext, model_id: int) -> tuple[bool, bool] | None:
    with context.engine.connect() as conn:
        metrics = conn.execute(
            select(managed_model_evaluations.c.metrics_json)
            .where(managed_model_evaluations.c.model_id == model_id)
            .where(managed_model_evaluations.c.evaluation_kind == "paper_shadow")
            .where(managed_model_evaluations.c.fold_index == 0)
        ).scalar_one_or_none()
    if not isinstance(metrics, Mapping):
        return None
    incremental = metrics.get("context_incremental")
    incremental_eligible = isinstance(incremental, Mapping) and incremental.get("eligible") is True
    return incremental_eligible, metrics.get("reporting_eligible") is True


def _evaluate_registered(
    context: DaemonContext,
    registration: _ModelRegistration,
    *,
    feature_schema_version: str,
    include_hermes_context: bool,
    policy: PromotionPolicy,
) -> TrainingRun:
    config = context.settings.managed_learning
    report = evaluate_challenger(
        context.engine,
        feature_schema_version=feature_schema_version,
        outcome_policy_version=config.outcome_policy_version,
        policy=policy,
        include_hermes_context=include_hermes_context,
        min_train_sessions=config.min_train_sessions,
        embargo_sessions=config.embargo_sessions,
    )
    return TrainingRun(
        artifact=None,
        report=report,
        model_id=registration.model_id,
        status=registration.status,
    )


def _incremental_payload(incremental: ContextIncrementalReport) -> dict[str, object]:
    payload = asdict(incremental)
    if not math.isfinite(incremental.mean_incremental_pnl_lcb):
        payload["mean_incremental_pnl_lcb"] = "negative_infinity"
    return payload


def _registered_artifact(
    registration: _ModelRegistration, artifact_dir: Path
) -> ManagedModelArtifact:
    artifact = ManagedModelArtifact.from_json(
        artifact_path(artifact_dir, registration.model_version).read_text(encoding="utf-8")
    )
    expected = (
        artifact.model_version == registration.model_version
        and artifact.artifact_hash == registration.artifact_hash
        and artifact.feature_schema_version == registration.feature_schema_version
        and artifact.outcome_policy_version == registration.outcome_policy_version
        and artifact.trained_from_session == registration.trained_from_session
        and artifact.trained_through_session == registration.trained_through_session
    )
    if not expected:
        raise ValueError("challenger artifact differs from immutable registry")
    return artifact


def _evaluate_active_base_challenger(
    context: DaemonContext,
    registration: _ModelRegistration,
    samples: list[ManagedSample],
    sessions: list[str],
    *,
    artifact_dir: Path,
    now: datetime,
) -> ManagedLearningSummary:
    """Evaluate exactly one frozen challenger on one future cohort."""
    config = context.settings.managed_learning
    selection_policy = _policy(context).admission_selection_policy()
    stored = _stored_holdout_evidence(context, registration.model_id)
    if stored is not None:
        promoted: str | None = None
        if stored.get("eligible") is True and config.auto_promote:
            promote_challenger(
                context.engine,
                artifact_dir,
                registration.model_version,
                paper_only=(context.settings.execution.paper_only and context.settings.ibkr.paper),
                expected_admission_policy=selection_policy,
                now=now,
            )
            promoted = registration.model_version
        return ManagedLearningSummary(
            status=(
                "promoted"
                if promoted is not None
                else "prospective_eligible_manual_promotion"
                if stored.get("eligible") is True
                else "prospective_rejected"
            ),
            samples=len(samples),
            sessions=len(sessions),
            base_model_version=registration.model_version,
            promoted_model_version=promoted,
            base_eligible=stored.get("eligible") is True,
        )

    block = _prospective_block(
        samples,
        trained_through_session=registration.trained_through_session,
        min_sessions=config.prospective_min_sessions,
        min_independent_signals=config.prospective_min_independent_signals,
    )
    if not block:
        return ManagedLearningSummary(
            "awaiting_prospective_holdout",
            len(samples),
            len(sessions),
            base_model_version=registration.model_version,
            base_eligible=registration.metrics.get("eligible") is True,
        )

    artifact = _registered_artifact(registration, artifact_dir)
    rows = score_frozen_artifact(artifact, block)
    report = evaluate_prospective_rows(
        rows,
        min_sessions=config.prospective_min_sessions,
        min_independent_signals=config.prospective_min_independent_signals,
        min_admitted=config.prospective_min_admitted,
        min_admitted_sessions=config.prospective_min_admitted_sessions,
        min_profit_factor=config.min_profit_factor,
        max_admitted_per_session=(context.settings.execution.opening_range_max_entries_per_day),
        bootstrap_iterations=config.bootstrap_iterations,
        score_floor=selection_policy.score_floor,
        single_trade_cap_pct=selection_policy.single_trade_cap_pct,
        max_candidates_per_batch=selection_policy.max_candidates_per_batch,
    )
    incumbent = load_promoted_model(
        context.engine,
        artifact_dir,
        expected_admission_policy=selection_policy,
    )
    incremental: ContextIncrementalReport | None = None
    if incumbent is not None:
        incumbent_rows = score_frozen_artifact(incumbent, block)
        incremental = compare_context_incremental_value(
            incumbent_rows,
            rows,
            min_disagreements=(config.prospective_min_incumbent_disagreements),
            min_sessions=config.prospective_min_admitted_sessions,
            bootstrap_iterations=config.bootstrap_iterations,
            max_admitted_per_session=(context.settings.execution.opening_range_max_entries_per_day),
            score_floor=selection_policy.score_floor,
            single_trade_cap_pct=selection_policy.single_trade_cap_pct,
            max_candidates_per_batch=selection_policy.max_candidates_per_batch,
        )
    payload = _prospective_payload(
        registration,
        report,
        incumbent=incumbent,
        incremental=incremental,
        cohort_content_sha256=prospective_cohort_content_sha256(
            context.engine,
            [row.opportunity_id for row in report.rows],
        ),
        evaluation_policy={
            "min_sessions": config.prospective_min_sessions,
            "min_independent_signals": config.prospective_min_independent_signals,
            "min_admitted": config.prospective_min_admitted,
            "min_admitted_sessions": config.prospective_min_admitted_sessions,
            "min_profit_factor": config.min_profit_factor,
            "score_floor": selection_policy.score_floor,
            "single_trade_cap_pct": selection_policy.single_trade_cap_pct,
            "max_candidates_per_batch": selection_policy.max_candidates_per_batch,
            "max_admitted_per_session": (
                context.settings.execution.opening_range_max_entries_per_day
            ),
            "bootstrap_iterations": config.bootstrap_iterations,
        },
        incremental_policy=(
            {
                "min_disagreements": config.prospective_min_incumbent_disagreements,
                "min_sessions": config.prospective_min_admitted_sessions,
                "bootstrap_iterations": config.bootstrap_iterations,
                "score_floor": selection_policy.score_floor,
                "single_trade_cap_pct": selection_policy.single_trade_cap_pct,
                "max_candidates_per_batch": selection_policy.max_candidates_per_batch,
                "max_admitted_per_session": (
                    context.settings.execution.opening_range_max_entries_per_day
                ),
            }
            if incumbent is not None
            else None
        ),
    )
    try:
        with context.engine.begin() as conn:
            conn.execute(
                insert(managed_model_evaluations).values(
                    model_id=registration.model_id,
                    evaluation_kind="holdout",
                    fold_index=0,
                    train_from_session=registration.trained_from_session,
                    train_through_session=registration.trained_through_session,
                    test_from_session=rows[0].session,
                    test_through_session=rows[-1].session,
                    metrics_json=payload,
                    created_at=now,
                )
            )
            if payload["eligible"] is not True:
                conn.execute(
                    update(managed_models)
                    .where(managed_models.c.id == registration.model_id)
                    .where(managed_models.c.status == "challenger")
                    .values(status="rejected")
                )
    except IntegrityError:
        existing = _stored_holdout_evidence(context, registration.model_id)
        if existing is None:
            raise
        payload = dict(existing)

    promoted = None
    if payload.get("eligible") is True and config.auto_promote:
        promote_challenger(
            context.engine,
            artifact_dir,
            registration.model_version,
            paper_only=(context.settings.execution.paper_only and context.settings.ibkr.paper),
            expected_admission_policy=selection_policy,
            now=now,
        )
        promoted = registration.model_version
    return ManagedLearningSummary(
        status=(
            "promoted"
            if promoted is not None
            else "prospective_eligible_manual_promotion"
            if payload.get("eligible") is True
            else "prospective_rejected"
        ),
        samples=len(samples),
        sessions=len(sessions),
        base_model_version=registration.model_version,
        promoted_model_version=promoted,
        base_eligible=payload.get("eligible") is True,
    )


def _context_evaluation_metrics(
    run: TrainingRun,
    incremental: ContextIncrementalReport,
) -> dict[str, object]:
    standalone_eligible = run.status == "challenger" and run.report.eligible
    reporting_eligible = standalone_eligible and incremental.eligible
    return {
        "standalone_eligible": standalone_eligible,
        "context_incremental": _incremental_payload(incremental),
        "reporting_eligible": reporting_eligible,
        "deployment_scope": "shadow_reporting_only",
        "promotion_allowed": False,
    }


def _record_context_incremental(
    context: DaemonContext,
    run: TrainingRun,
    incremental: ContextIncrementalReport,
    *,
    now: datetime,
) -> bool:
    if run.model_id is None or not run.report.rows:
        return False
    stored = _stored_context_evidence(context, run.model_id)
    if stored is not None:
        return stored[1]
    metrics = _context_evaluation_metrics(run, incremental)
    rows = run.report.rows
    try:
        with context.engine.begin() as conn:
            model = conn.execute(
                select(
                    managed_models.c.trained_from_session,
                    managed_models.c.trained_through_session,
                ).where(managed_models.c.id == run.model_id)
            ).one()
            conn.execute(
                insert(managed_model_evaluations).values(
                    model_id=run.model_id,
                    evaluation_kind="paper_shadow",
                    fold_index=0,
                    train_from_session=model.trained_from_session,
                    train_through_session=model.trained_through_session,
                    test_from_session=rows[0].session,
                    test_through_session=rows[-1].session,
                    metrics_json=metrics,
                    created_at=now,
                )
            )
            if metrics["reporting_eligible"] is not True:
                conn.execute(
                    update(managed_models)
                    .where(managed_models.c.id == run.model_id)
                    .where(managed_models.c.status == "challenger")
                    .values(status="rejected")
                )
    except IntegrityError:
        stored = _stored_context_evidence(context, run.model_id)
        if stored is None:
            raise
        return stored[1]
    return metrics["reporting_eligible"] is True


def _training_pass(
    context: DaemonContext,
    *,
    now: datetime,
) -> ManagedLearningSummary:
    config = context.settings.managed_learning
    samples = load_training_samples(
        context.engine,
        feature_schema_version=config.feature_schema_version,
        outcome_policy_version=config.outcome_policy_version,
        canonical_per_signal=False,
    )
    sessions = sorted({sample.session for sample in samples})
    if not samples or not sessions:
        return ManagedLearningSummary("insufficient_data", len(samples), len(sessions))
    suffix = f"{sessions[-1].replace('-', '')}-n{len(samples)}"
    base_version = f"managed-base-{suffix}"
    context_version = f"managed-context-{suffix}"
    artifact_dir = config.artifact_dir
    if not isinstance(artifact_dir, Path):
        raise ValueError("managed learning artifact directory is unresolved")
    active_base = _active_base_challenger(context)
    active_context_version = (
        active_base.model_version.replace("managed-base-", "managed-context-", 1)
        if active_base is not None
        else None
    )
    active_context_registration = (
        _registration(context, active_context_version)
        if active_context_version is not None and config.train_hermes_context_challenger
        else None
    )
    active_context_stored = (
        _stored_context_evidence(context, active_context_registration.model_id)
        if active_context_registration is not None
        else None
    )
    if active_base is not None and (
        not config.train_hermes_context_challenger
        or active_context_stored is not None
        # A recovery pass may finish the paired context evaluation only while
        # no future cohort exists. Once future data arrives, serial base
        # evaluation takes precedence and no model may be refit on that block.
        or active_base.model_version != base_version
    ):
        return _evaluate_active_base_challenger(
            context,
            active_base,
            samples,
            sessions,
            artifact_dir=artifact_dir,
            now=now,
        )
    base_registration = _registration(context, base_version)
    context_registration = (
        _registration(context, context_version) if config.train_hermes_context_challenger else None
    )
    stored_context = (
        _stored_context_evidence(context, context_registration.model_id)
        if context_registration is not None
        else None
    )
    if base_registration is not None and (
        not config.train_hermes_context_challenger
        or (context_registration is not None and stored_context is not None)
    ):
        return ManagedLearningSummary(
            "already_registered",
            len(samples),
            len(sessions),
            base_model_version=base_version,
            context_model_version=(context_version if context_registration else None),
            base_eligible=base_registration.metrics.get("eligible") is True,
            context_eligible=(
                context_registration is not None
                and context_registration.metrics.get("eligible") is True
            ),
            context_incremental_eligible=(
                stored_context[0] if stored_context is not None else False
            ),
            context_reporting_eligible=(stored_context[1] if stored_context is not None else False),
        )
    policy = _policy(context)
    base_run = (
        train_challenger(
            context.engine,
            artifact_dir,
            model_version=base_version,
            feature_schema_version=config.feature_schema_version,
            outcome_policy_version=config.outcome_policy_version,
            policy=policy,
            min_train_sessions=config.min_train_sessions,
            embargo_sessions=config.embargo_sessions,
            now=now,
        )
        if base_registration is None
        else _evaluate_registered(
            context,
            base_registration,
            feature_schema_version=config.feature_schema_version,
            include_hermes_context=False,
            policy=policy,
        )
    )
    context_run: TrainingRun | None = None
    incremental: ContextIncrementalReport | None = None
    context_reporting_eligible = False
    if config.train_hermes_context_challenger and base_run.report.rows:
        context_schema = config.feature_schema_version + "+hermes_context_v1"
        context_run = (
            train_challenger(
                context.engine,
                artifact_dir,
                model_version=context_version,
                feature_schema_version=context_schema,
                outcome_policy_version=config.outcome_policy_version,
                policy=policy,
                include_hermes_context=True,
                min_train_sessions=config.min_train_sessions,
                embargo_sessions=config.embargo_sessions,
                now=now,
            )
            if context_registration is None
            else _evaluate_registered(
                context,
                context_registration,
                feature_schema_version=context_schema,
                include_hermes_context=True,
                policy=policy,
            )
        )
        if context_run.report.rows:
            incremental = compare_context_incremental_value(
                base_run.report.rows,
                context_run.report.rows,
                min_disagreements=config.min_context_disagreements,
                min_sessions=config.min_context_disagreement_sessions,
                bootstrap_iterations=config.bootstrap_iterations,
                max_admitted_per_session=(
                    context.settings.execution.opening_range_max_entries_per_day
                ),
                score_floor=policy.score_floor,
                single_trade_cap_pct=policy.single_trade_cap_pct,
                max_candidates_per_batch=policy.max_candidates_per_batch,
            )
            context_reporting_eligible = _record_context_incremental(
                context,
                context_run,
                incremental,
                now=now,
            )
    return ManagedLearningSummary(
        status="trained",
        samples=len(samples),
        sessions=len(sessions),
        base_model_version=(
            base_version if base_registration is not None or base_run.model_id is not None else None
        ),
        context_model_version=(
            context_version
            if context_run is not None
            and (context_registration is not None or context_run.model_id is not None)
            else None
        ),
        base_eligible=base_run.report.eligible,
        context_eligible=(context_run.report.eligible if context_run is not None else False),
        context_incremental_eligible=(incremental.eligible if incremental is not None else False),
        context_reporting_eligible=context_reporting_eligible,
    )


async def run_managed_learning_tick(
    context: DaemonContext,
    *,
    now: datetime | None = None,
) -> ManagedLearningSummary:
    """Train off-loop only outside RTH and after active paths have resolved."""
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    if not context.settings.managed_learning.enabled:
        return ManagedLearningSummary("disabled", 0, 0)
    if is_market_open(observed_at):
        return ManagedLearningSummary("market_open", 0, 0)
    with context.engine.connect() as conn:
        active = int(
            conn.execute(
                select(func.count())
                .select_from(managed_opportunities)
                .where(managed_opportunities.c.status.in_(["pending_entry", "active"]))
            ).scalar_one()
        )
    if active:
        return ManagedLearningSummary("capture_active", 0, 0)
    return await asyncio.to_thread(_training_pass, context, now=observed_at)
