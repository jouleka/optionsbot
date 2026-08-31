"""Database and artifact boundary for managed-outcome learning.

Only prospectively captured, explicitly training-eligible rows enter this
module.  Artifact files are canonical JSON and are verified against both their
embedded checksum and the immutable registry row before use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from sqlalchemy import Engine, insert, select, update
from sqlalchemy.engine import Connection, RowMapping

from optionsbot.admission_policy import AdmissionSelectionPolicy
from optionsbot.learning.features import model_features
from optionsbot.learning.managed_model import (
    ManagedModelArtifact,
    ManagedSample,
    Outcome,
    PromotionPolicy,
    PromotionReport,
    compare_context_incremental_value,
    evaluate_prospective_rows,
    fit_managed_model,
    score_frozen_artifact,
    walk_forward_evaluate,
)
from optionsbot.managed_contract import MANAGED_FEATURE_SCHEMA_VERSION
from optionsbot.storage.schema import (
    managed_context_reviews,
    managed_model_evaluations,
    managed_models,
    managed_opportunities,
)

BASE_MODEL_ROLE = "causal_base"
CONTEXT_MODEL_ROLE = "hermes_context_shadow"
PROSPECTIVE_HOLDOUT_PROTOCOL = "serial_frozen_challenger_future_prefix_v2"


@dataclass(frozen=True, slots=True)
class TrainingRun:
    artifact: ManagedModelArtifact | None
    report: PromotionReport
    model_id: int | None
    status: str


def _as_utc(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _approved_base_metrics(
    row: RowMapping,
    *,
    expected_admission_policy: AdmissionSelectionPolicy | None = None,
) -> Mapping[str, object]:
    metrics = row.get("metrics_json")
    model_version = row.get("model_version")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("eligible") is not True
        or metrics.get("model_role") != BASE_MODEL_ROLE
        or metrics.get("include_hermes_context") is not False
        or metrics.get("promotion_allowed") is not True
        or metrics.get("deployment_scope") != "paper_admission_candidate"
        or not isinstance(model_version, str)
        or metrics.get("artifact_file") != artifact_path(Path(), model_version).name
    ):
        raise ValueError("model registry lacks eligible causal-base promotion metrics")
    try:
        registered_policy = AdmissionSelectionPolicy.from_payload(
            metrics.get("admission_selection_policy")
        )
    except ValueError as exc:
        raise ValueError("model registry lacks a valid admission selection policy") from exc
    if (
        expected_admission_policy is not None
        and registered_policy != expected_admission_policy
    ):
        raise ValueError("promoted model admission selection policy differs from runtime")
    return metrics


def _canonical_cohort_value(value: object) -> object:
    """Normalize persisted SQL/JSON values for a cross-process stable digest."""
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_cohort_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [_canonical_cohort_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("prospective holdout cohort contains non-finite content")
        # JSON's shortest round-trippable representation is deterministic for
        # Python floats and avoids lossy display rounding in the evidence hash.
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    raise ValueError("prospective holdout cohort contains unsupported content")


def _cohort_content_sha256(rows: Sequence[RowMapping]) -> str:
    """Bind every persisted decision field and realized label in cohort order."""
    canonical = [
        {
            str(column.name): _canonical_cohort_value(row[column.name])
            for column in managed_opportunities.columns
        }
        for row in rows
    ]
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cohort_rows(
    conn: Connection,
    opportunity_ids: Sequence[int],
) -> list[RowMapping]:
    rows = list(
        conn.execute(
            select(managed_opportunities).where(managed_opportunities.c.id.in_(opportunity_ids))
        ).mappings()
    )
    by_id = {int(row["id"]): row for row in rows}
    if len(by_id) != len(opportunity_ids) or any(row_id not in by_id for row_id in opportunity_ids):
        raise ValueError("prospective holdout cohort contains nonexistent opportunities")
    return [by_id[row_id] for row_id in opportunity_ids]


def prospective_cohort_content_sha256(
    engine: Engine,
    opportunity_ids: Sequence[int],
) -> str:
    """Return the canonical digest recorded with one immutable holdout cohort."""
    if not opportunity_ids or any(_positive_int(value) is None for value in opportunity_ids):
        raise ValueError("prospective holdout cohort provenance is malformed")
    normalized_ids = [int(value) for value in opportunity_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("prospective holdout cohort provenance is malformed")
    with engine.connect() as conn:
        rows = _load_cohort_rows(conn, normalized_ids)
    return _cohort_content_sha256(rows)


def _validated_prospective_holdout(
    conn: Connection,
    model: RowMapping,
    holdout: RowMapping | None,
    artifact: ManagedModelArtifact,
    artifact_dir: Path,
    *,
    require_promoted_at: bool,
) -> Mapping[str, object]:
    """Validate immutable future-cohort evidence against one registry identity."""
    if holdout is None:
        raise ValueError("model lacks successful prospective holdout evidence")
    if (
        holdout.get("evaluation_kind") != "holdout"
        or holdout.get("fold_index") != 0
        or holdout.get("model_id") != model.get("id")
        or holdout.get("train_from_session") != model.get("trained_from_session")
        or holdout.get("train_through_session") != model.get("trained_through_session")
    ):
        raise ValueError("prospective holdout provenance differs from model registry")

    trained_through_raw = model.get("trained_through_session")
    test_from_raw = holdout.get("test_from_session")
    test_through_raw = holdout.get("test_through_session")
    if not all(
        isinstance(value, str) for value in (trained_through_raw, test_from_raw, test_through_raw)
    ):
        raise ValueError("prospective holdout session provenance is malformed")
    try:
        trained_through = date.fromisoformat(cast(str, trained_through_raw))
        test_from = date.fromisoformat(cast(str, test_from_raw))
        test_through = date.fromisoformat(cast(str, test_through_raw))
    except ValueError as exc:
        raise ValueError("prospective holdout session provenance is malformed") from exc
    if test_from <= trained_through or test_through < test_from:
        raise ValueError("prospective holdout is not strictly after the training window")

    created_at = _as_utc(model.get("created_at"))
    evaluated_at = _as_utc(holdout.get("created_at"))
    promoted_at = _as_utc(model.get("promoted_at"))
    if created_at is None or evaluated_at is None or evaluated_at < created_at:
        raise ValueError("prospective holdout timestamp provenance is invalid")
    if require_promoted_at and (promoted_at is None or promoted_at < evaluated_at):
        raise ValueError("promotion predates its prospective holdout evidence")

    evidence = holdout.get("metrics_json")
    if not isinstance(evidence, Mapping):
        raise ValueError("prospective holdout metrics are malformed")
    identity_fields = (
        "model_version",
        "artifact_hash",
        "feature_schema_version",
        "outcome_policy_version",
        "trained_through_session",
    )
    if any(evidence.get(name) != model.get(name) for name in identity_fields):
        raise ValueError("prospective holdout identity differs from model registry")
    if (
        evidence.get("eligible") is not True
        or evidence.get("promotion_allowed") is not True
        or evidence.get("evaluation_protocol") != PROSPECTIVE_HOLDOUT_PROTOCOL
    ):
        raise ValueError("model lacks successful prospective holdout evidence")

    cohort_raw = evidence.get("cohort_opportunity_ids")
    if not isinstance(cohort_raw, list) or not cohort_raw:
        raise ValueError("prospective holdout cohort provenance is malformed")
    cohort: list[int] = []
    for value in cohort_raw:
        parsed = _positive_int(value)
        if parsed is None:
            raise ValueError("prospective holdout cohort provenance is malformed")
        cohort.append(parsed)
    if len(set(cohort)) != len(cohort):
        raise ValueError("prospective holdout cohort provenance is malformed")
    cohort_digest = hashlib.sha256("|".join(str(value) for value in cohort).encode()).hexdigest()
    if evidence.get("cohort_sha256") != cohort_digest:
        raise ValueError("prospective holdout cohort digest is invalid")

    cohort_rows = _load_cohort_rows(conn, cohort)
    expected_order = [
        int(row["id"])
        for row in sorted(cohort_rows, key=lambda item: (str(item["session"]), int(item["id"])))
    ]
    if cohort != expected_order:
        raise ValueError("prospective holdout cohort order is not deterministic")
    expected_policy = model.get("outcome_policy_version")
    expected_schema = model.get("feature_schema_version")
    sessions: list[date] = []
    signals: set[str] = set()
    for row in cohort_rows:
        session_raw = row["session"]
        try:
            session = date.fromisoformat(str(session_raw))
        except ValueError as exc:
            raise ValueError("prospective holdout cohort session is malformed") from exc
        if session <= trained_through or not test_from <= session <= test_through:
            raise ValueError("prospective holdout cohort falls outside its declared future range")
        if (
            row["status"] != "resolved"
            or row["training_eligible"] != 1
            or row["outcome"] not in {"target", "stop", "timeout"}
            or row["basis_dollars"] is None
            or row["net_pnl"] is None
        ):
            raise ValueError("prospective holdout cohort is not resolved training-eligible data")
        if row["admission_eligible"] != 1 or row["shadow_only"] != 0:
            raise ValueError("prospective holdout cohort contains research-only opportunities")
        if row["policy_version"] != expected_policy:
            raise ValueError("prospective holdout cohort outcome policy differs from artifact")
        features = row["features_json"]
        if (
            not isinstance(features, Mapping)
            or features.get("feature_schema_version") != expected_schema
        ):
            raise ValueError("prospective holdout cohort feature schema differs from artifact")
        resolved_at = _as_utc(row["resolved_at"])
        if resolved_at is None or resolved_at > evaluated_at:
            raise ValueError("prospective holdout predates cohort resolution")
        sessions.append(session)
        signals.add(str(row["signal_id"]))
    if min(sessions) != test_from or max(sessions) != test_through:
        raise ValueError("prospective holdout cohort does not match its declared test range")
    expected_content_digest = evidence.get("cohort_content_sha256")
    if (
        not isinstance(expected_content_digest, str)
        or len(expected_content_digest) != 64
        or expected_content_digest != _cohort_content_sha256(cohort_rows)
    ):
        raise ValueError("prospective holdout cohort row-content digest is invalid")

    evaluation_policy = evidence.get("evaluation_policy")
    if not isinstance(evaluation_policy, Mapping):
        raise ValueError("prospective holdout evaluation policy is malformed")
    integer_policy: dict[str, int] = {}
    for name, minimum, maximum in (
        ("min_sessions", 2, None),
        ("min_independent_signals", 3, None),
        ("min_admitted", 1, None),
        ("min_admitted_sessions", 1, None),
        ("max_candidates_per_batch", 1, None),
        ("max_admitted_per_session", 1, None),
        ("bootstrap_iterations", 200, 20_000),
    ):
        value = _positive_int(evaluation_policy.get(name))
        if value is None or value < minimum or (maximum is not None and value > maximum):
            raise ValueError("prospective holdout evaluation policy is malformed")
        integer_policy[name] = value
    min_profit_factor = _finite_float(evaluation_policy.get("min_profit_factor"))
    score_floor = _finite_float(evaluation_policy.get("score_floor"))
    single_trade_cap_pct = _finite_float(
        evaluation_policy.get("single_trade_cap_pct")
    )
    if (
        min_profit_factor is None
        or min_profit_factor < 1.0
        or score_floor is None
        or single_trade_cap_pct is None
        or integer_policy["min_admitted_sessions"] > integer_policy["min_sessions"]
    ):
        raise ValueError("prospective holdout evaluation policy is malformed")
    try:
        selection_policy = AdmissionSelectionPolicy(
            score_floor=score_floor,
            single_trade_cap_pct=single_trade_cap_pct,
            max_candidates_per_batch=integer_policy["max_candidates_per_batch"],
            max_admitted_per_session=integer_policy["max_admitted_per_session"],
        )
        registry_policy = AdmissionSelectionPolicy.from_payload(
            _approved_base_metrics(model).get("admission_selection_policy")
        )
    except ValueError as exc:
        raise ValueError("prospective holdout evaluation policy is malformed") from exc
    if selection_policy != registry_policy:
        raise ValueError("prospective holdout selector differs from model registry")

    cohort_samples = [_managed_sample_from_row(row) for row in cohort_rows]
    if any(sample is None for sample in cohort_samples):
        raise ValueError("prospective holdout cohort lifecycle or economics are invalid")
    scored = score_frozen_artifact(
        artifact,
        [cast(ManagedSample, sample) for sample in cohort_samples],
    )
    recomputed = evaluate_prospective_rows(
        scored,
        min_sessions=integer_policy["min_sessions"],
        min_independent_signals=integer_policy["min_independent_signals"],
        min_admitted=integer_policy["min_admitted"],
        min_admitted_sessions=integer_policy["min_admitted_sessions"],
        min_profit_factor=min_profit_factor,
        max_admitted_per_session=integer_policy["max_admitted_per_session"],
        bootstrap_iterations=integer_policy["bootstrap_iterations"],
        score_floor=selection_policy.score_floor,
        single_trade_cap_pct=selection_policy.single_trade_cap_pct,
        max_candidates_per_batch=selection_policy.max_candidates_per_batch,
    )
    recomputed_absolute = asdict(recomputed)
    recomputed_absolute.pop("rows", None)
    absolute = evidence.get("absolute")
    if absolute != _json_safe(recomputed_absolute):
        raise ValueError("prospective holdout absolute metrics differ from frozen cohort scoring")
    if (
        not isinstance(absolute, Mapping)
        or absolute.get("eligible") is not True
        or absolute.get("reasons") != []
        or _positive_int(absolute.get("samples")) != len(cohort)
        or _positive_int(absolute.get("sessions")) != len(set(sessions))
        or _positive_int(absolute.get("independent_signals")) != len(signals)
        or _positive_int(absolute.get("admitted")) is None
        or _positive_int(absolute.get("admitted_sessions")) is None
    ):
        raise ValueError("prospective holdout did not record eligible absolute metrics")
    lcb = absolute.get("admitted_mean_pnl_lcb")
    if (
        isinstance(lcb, bool)
        or not isinstance(lcb, int | float)
        or not math.isfinite(float(lcb))
        or float(lcb) <= 0.0
    ):
        raise ValueError("prospective holdout after-cost LCB is not positive")

    incumbent = evidence.get("incumbent")
    incremental_policy = evidence.get("incremental_policy")
    if incumbent is None:
        if incremental_policy is not None:
            raise ValueError("prospective holdout records incremental policy without incumbent")
        return evidence
    incremental = incumbent.get("incremental") if isinstance(incumbent, Mapping) else None
    identity_names = (
        "model_version",
        "artifact_hash",
        "feature_schema_version",
        "outcome_policy_version",
        "trained_through_session",
    )
    if (
        not isinstance(incumbent, Mapping)
        or any(not isinstance(incumbent.get(name), str) for name in identity_names)
        or incumbent.get("incremental_eligible") is not True
        or not isinstance(incremental, Mapping)
        or not isinstance(incremental_policy, Mapping)
    ):
        raise ValueError("prospective replacement evidence is not eligible")
    parsed_incremental_policy: dict[str, int] = {}
    for name, minimum, maximum in (
        ("min_disagreements", 1, None),
        ("min_sessions", 1, None),
        ("bootstrap_iterations", 200, 20_000),
        ("max_candidates_per_batch", 1, None),
        ("max_admitted_per_session", 1, None),
    ):
        value = _positive_int(incremental_policy.get(name))
        if value is None or value < minimum or (maximum is not None and value > maximum):
            raise ValueError("prospective holdout incremental policy is malformed")
        parsed_incremental_policy[name] = value
    incremental_score_floor = _finite_float(incremental_policy.get("score_floor"))
    incremental_cap_pct = _finite_float(incremental_policy.get("single_trade_cap_pct"))
    try:
        incremental_selection_policy = AdmissionSelectionPolicy(
            score_floor=(
                float(incremental_score_floor)
                if incremental_score_floor is not None
                else float("nan")
            ),
            single_trade_cap_pct=(
                float(incremental_cap_pct)
                if incremental_cap_pct is not None
                else float("nan")
            ),
            max_candidates_per_batch=parsed_incremental_policy[
                "max_candidates_per_batch"
            ],
            max_admitted_per_session=parsed_incremental_policy[
                "max_admitted_per_session"
            ],
        )
    except ValueError as exc:
        raise ValueError("prospective holdout incremental policy is malformed") from exc
    if incremental_selection_policy != selection_policy:
        raise ValueError("prospective incremental selector differs from absolute selector")

    incumbent_row = (
        conn.execute(
            select(managed_models).where(
                managed_models.c.model_version == incumbent["model_version"]
            )
        )
        .mappings()
        .one_or_none()
    )
    if (
        incumbent_row is None
        or incumbent_row["status"] not in {"promoted", "retired"}
        or any(incumbent.get(name) != incumbent_row[name] for name in identity_names)
        or incumbent_row["feature_schema_version"] != model["feature_schema_version"]
        or incumbent_row["outcome_policy_version"] != model["outcome_policy_version"]
    ):
        raise ValueError("prospective replacement incumbent differs from immutable registry")
    _approved_base_metrics(incumbent_row)
    incumbent_artifact = ManagedModelArtifact.from_json(
        artifact_path(artifact_dir, str(incumbent_row["model_version"])).read_text(encoding="utf-8")
    )
    _reject_context_artifact(incumbent_artifact)
    artifact_identity = (
        incumbent_artifact.model_version,
        incumbent_artifact.artifact_hash,
        incumbent_artifact.feature_schema_version,
        incumbent_artifact.outcome_policy_version,
        incumbent_artifact.trained_through_session,
    )
    registry_identity = tuple(str(incumbent_row[name]) for name in identity_names)
    if artifact_identity != registry_identity:
        raise ValueError("prospective replacement incumbent artifact differs from registry")
    incumbent_scored = score_frozen_artifact(
        incumbent_artifact,
        [cast(ManagedSample, sample) for sample in cohort_samples],
    )
    recomputed_incremental = compare_context_incremental_value(
        incumbent_scored,
        scored,
        min_disagreements=parsed_incremental_policy["min_disagreements"],
        min_sessions=parsed_incremental_policy["min_sessions"],
        bootstrap_iterations=parsed_incremental_policy["bootstrap_iterations"],
        max_admitted_per_session=parsed_incremental_policy["max_admitted_per_session"],
        score_floor=incremental_selection_policy.score_floor,
        single_trade_cap_pct=incremental_selection_policy.single_trade_cap_pct,
        max_candidates_per_batch=incremental_selection_policy.max_candidates_per_batch,
    )
    if incremental != _json_safe(asdict(recomputed_incremental)):
        raise ValueError("prospective replacement metrics differ from frozen incumbent comparison")
    if not recomputed_incremental.eligible:
        raise ValueError("prospective replacement evidence is not eligible")
    return evidence


def _context_features(engine: Engine) -> dict[int, dict[str, float | None]]:
    """Latest causally pre-entry Hermes observation per signal structure.

    Hermes judges directional context for the shared signal, not the option
    structure it happened to receive in its packet.  Reviews are therefore
    propagated to every captured structure with the same immutable signal
    identity.  The storage timing bucket alone is insufficient here: it is
    classified against a real broker entry, while managed shadow paths start at
    ``managed_opportunities.entry_ts``.  Require the review to precede each
    target structure's own shadow entry strictly so late context can never
    become a feature of a path already in progress.
    """
    source = managed_opportunities.alias("context_source_opportunity")
    target = managed_opportunities.alias("context_target_opportunity")
    stmt = (
        select(
            target.c.id.label("target_opportunity_id"),
            managed_context_reviews.c.received_at,
            managed_context_reviews.c.context_probability,
            managed_context_reviews.c.event_conflict,
            managed_context_reviews.c.anomaly_json,
            managed_context_reviews.c.id.label("review_id"),
        )
        .select_from(managed_context_reviews)
        .join(
            source,
            source.c.id == managed_context_reviews.c.opportunity_id,
        )
        .join(
            target,
            (target.c.signal_id == source.c.signal_id) & (target.c.session == source.c.session),
        )
        .where(managed_context_reviews.c.timing == "pretrade")
        .where(target.c.entry_ts.is_not(None))
        .where(managed_context_reviews.c.received_at >= source.c.created_at)
        .where(managed_context_reviews.c.received_at >= source.c.detected_at)
        .where(managed_context_reviews.c.received_at >= target.c.detected_at)
        .where(
            target.c.bot_decided_at.is_(None)
            | (managed_context_reviews.c.received_at >= target.c.bot_decided_at)
        )
        .where(managed_context_reviews.c.received_at < target.c.entry_ts)
        .order_by(
            target.c.id,
            managed_context_reviews.c.received_at,
            managed_context_reviews.c.id,
        )
    )
    latest: dict[int, dict[str, float | None]] = {}
    with engine.connect() as conn:
        for row in conn.execute(stmt):
            values: dict[str, float | None] = {
                "hermes.review_present": 1.0,
                "hermes.context_probability": (
                    float(row.context_probability) if row.context_probability is not None else None
                ),
                "hermes.event_conflict": (
                    float(row.event_conflict) if row.event_conflict is not None else None
                ),
            }
            anomalies = row.anomaly_json
            if isinstance(anomalies, list):
                for anomaly in anomalies:
                    if isinstance(anomaly, str):
                        values[f"hermes.anomaly={anomaly}"] = 1.0
            latest[int(row.target_opportunity_id)] = values
    return latest


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _managed_sample_from_row(
    row: RowMapping,
    *,
    context: Mapping[str, float | None] | None = None,
) -> ManagedSample | None:
    """Build one sample only from a coherent, completely resolved lifecycle."""
    basis = _finite_float(row["basis_dollars"])
    entry_net = _finite_float(row["entry_net"])
    exit_net = _finite_float(row["exit_net"])
    costs = _finite_float(row["commission_estimate"])
    stop_pct = _finite_float(row["stop_pct"])
    target_pct = _finite_float(row["target_pct"])
    gross_pnl = _finite_float(row["gross_pnl"])
    net_pnl = _finite_float(row["net_pnl"])
    created_at = _as_utc(row["created_at"])
    detected_at = _as_utc(row["detected_at"])
    bot_decided_at = _as_utc(row["bot_decided_at"])
    entry_ts = _as_utc(row["entry_ts"])
    resolved_at = _as_utc(row["resolved_at"])
    timeout_at = _as_utc(row["timeout_at"])
    outcome = str(row["outcome"])
    features_raw = row["features_json"]
    valid_marks = _positive_int(row["valid_marks"])
    decision_batch_id = row["decision_batch_id"]
    decision_score = _finite_float(row["decision_score"])
    decision_defined_risk = row["decision_defined_risk"]
    decision_max_loss_raw = row["decision_max_loss"]
    decision_max_loss = (
        _finite_float(decision_max_loss_raw) if decision_max_loss_raw is not None else None
    )
    account_available_raw = row["decision_account_value_available"]
    account_value_raw = row["decision_account_value_usd"]
    account_value = _finite_float(account_value_raw) if account_value_raw is not None else None
    if (
        basis is None
        or basis <= 0.0
        or entry_net is None
        or exit_net is None
        or costs is None
        or costs < 0.0
        or stop_pct is None
        or not 0.0 < stop_pct < 1.0
        or target_pct is None
        or target_pct <= 0.0
        or gross_pnl is None
        or net_pnl is None
        or created_at is None
        or detected_at is None
        or bot_decided_at is None
        or entry_ts is None
        or not detected_at <= bot_decided_at <= entry_ts
        or resolved_at is None
        or timeout_at is None
        or not detected_at <= created_at <= entry_ts <= resolved_at
        or outcome not in {"target", "stop", "timeout"}
        or valid_marks is None
        or valid_marks < 2
        or not isinstance(features_raw, Mapping)
        or not isinstance(decision_batch_id, str)
        or not decision_batch_id
        or decision_score is None
        or not 0.0 <= decision_score <= 100.0
        or decision_defined_risk not in (0, 1)
        or (decision_max_loss_raw is not None and decision_max_loss is None)
        or (decision_max_loss is not None and decision_max_loss <= 0.0)
        or account_available_raw not in (0, 1)
        or (account_available_raw == 0 and account_value_raw is not None)
        or (account_available_raw == 1 and account_value is None)
        or not math.isclose(basis, abs(entry_net) * 100.0, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(
            gross_pnl,
            (entry_net - exit_net) * 100.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(net_pnl, gross_pnl - costs, rel_tol=0.0, abs_tol=1e-6)
        or (outcome == "target" and gross_pnl + 1e-6 < basis * target_pct)
        or (outcome == "stop" and gross_pnl - 1e-6 > -basis * stop_pct)
        or (outcome == "timeout" and resolved_at < timeout_at)
        or (outcome != "timeout" and resolved_at >= timeout_at)
    ):
        return None
    try:
        features = model_features(
            features_raw,
            basis_dollars=basis,
            stop_pct=stop_pct,
            target_pct=target_pct,
            commission_estimate=costs,
            direction=str(row["direction"]),
            setup_type=str(row["setup_type"]),
            strategy=str(row["strategy"]),
            context=context,
        )
        return ManagedSample(
            opportunity_id=int(row["id"]),
            opportunity_key=str(row["opportunity_key"]),
            session=str(row["session"]),
            features=features,
            outcome=cast(Outcome, outcome),
            basis_dollars=basis,
            target_gain=basis * target_pct,
            stop_loss=basis * stop_pct,
            timeout_gross_return=(gross_pnl / basis if outcome == "timeout" else 0.0),
            costs=costs,
            realized_net_pnl=net_pnl,
            signal_id=str(row["signal_id"]),
            decision_batch_id=decision_batch_id,
            detected_at=detected_at,
            decision_score=decision_score,
            decision_defined_risk=bool(decision_defined_risk),
            decision_max_loss=decision_max_loss,
            decision_account_value_available=bool(account_available_raw),
            decision_account_value_usd=account_value,
        )
    except (TypeError, ValueError, OverflowError):
        return None


def load_training_samples(
    engine: Engine,
    *,
    feature_schema_version: str = MANAGED_FEATURE_SCHEMA_VERSION,
    outcome_policy_version: str | None = None,
    include_hermes_context: bool = False,
    canonical_per_signal: bool = True,
    include_shadow_structures: bool = False,
) -> list[ManagedSample]:
    """Load unbiased managed-path labels, optionally adding causal Hermes data.

    Production-model evidence is restricted to structures the admission path
    can actually execute.  Experimental structure-grid rows are available only
    through the explicit ``include_shadow_structures`` research switch.  By
    default a signal contributes one deterministic executable structure: the
    smallest estimated friction as a fraction of debit basis, then lexical
    strategy and row ID.  Selection uses only decision-time fields and keeps
    correlated variants from pretending to be independent signals.
    """
    stmt = (
        select(managed_opportunities)
        .where(managed_opportunities.c.training_eligible == 1)
        .where(managed_opportunities.c.status == "resolved")
        .where(managed_opportunities.c.outcome.in_(["target", "stop", "timeout"]))
        .where(managed_opportunities.c.net_pnl.is_not(None))
        .where(managed_opportunities.c.basis_dollars.is_not(None))
        .order_by(
            managed_opportunities.c.session,
            managed_opportunities.c.signal_id,
            managed_opportunities.c.id,
        )
    )
    if not include_shadow_structures:
        stmt = stmt.where(managed_opportunities.c.admission_eligible == 1).where(
            managed_opportunities.c.shadow_only == 0
        )
    if outcome_policy_version is not None:
        stmt = stmt.where(managed_opportunities.c.policy_version == outcome_policy_version)
    with engine.connect() as conn:
        raw_rows = list(conn.execute(stmt).mappings())

    # The schema identity is part of the data-generating contract.  Never pool
    # legacy or fabricated payloads merely because their numeric fields happen
    # to flatten successfully.
    raw_rows = [
        row
        for row in raw_rows
        if isinstance(row["features_json"], Mapping)
        and row["features_json"].get("feature_schema_version") == feature_schema_version
    ]

    if canonical_per_signal:
        selected: dict[tuple[str, str], RowMapping] = {}
        for row in raw_rows:
            basis = float(row["basis_dollars"])
            payload = row["features_json"]
            suggestion = payload.get("suggestion") if isinstance(payload, Mapping) else None
            estimated_friction = (
                suggestion.get("estimated_round_trip_cost")
                if isinstance(suggestion, Mapping)
                else None
            )
            friction_dollars = (
                float(estimated_friction)
                if isinstance(estimated_friction, int | float)
                and not isinstance(estimated_friction, bool)
                and math.isfinite(float(estimated_friction))
                and float(estimated_friction) >= 0.0
                else float(row["commission_estimate"])
            )
            friction = friction_dollars / basis if basis > 0.0 else float("inf")
            key = (str(row["session"]), str(row["signal_id"]))
            existing = selected.get(key)
            candidate_order = (friction, str(row["strategy"]), int(row["id"]))
            if existing is None:
                selected[key] = row
                continue
            existing_basis = float(existing["basis_dollars"])
            existing_payload = existing["features_json"]
            existing_suggestion = (
                existing_payload.get("suggestion")
                if isinstance(existing_payload, Mapping)
                else None
            )
            existing_estimated_friction = (
                existing_suggestion.get("estimated_round_trip_cost")
                if isinstance(existing_suggestion, Mapping)
                else None
            )
            existing_friction_dollars = (
                float(existing_estimated_friction)
                if isinstance(existing_estimated_friction, int | float)
                and not isinstance(existing_estimated_friction, bool)
                and math.isfinite(float(existing_estimated_friction))
                and float(existing_estimated_friction) >= 0.0
                else float(existing["commission_estimate"])
            )
            existing_order = (
                existing_friction_dollars / existing_basis
                if existing_basis > 0.0
                else float("inf"),
                str(existing["strategy"]),
                int(existing["id"]),
            )
            if candidate_order < existing_order:
                selected[key] = row
        raw_rows = sorted(selected.values(), key=lambda row: (str(row["session"]), int(row["id"])))

    contexts = _context_features(engine) if include_hermes_context else {}
    samples: list[ManagedSample] = []
    for row in raw_rows:
        context_features = (
            contexts.get(
                int(row["id"]),
                {
                    "hermes.review_present": 0.0,
                    "hermes.context_probability": None,
                    "hermes.event_conflict": None,
                },
            )
            if include_hermes_context
            else None
        )
        sample = _managed_sample_from_row(row, context=context_features)
        if sample is None:
            continue
        samples.append(sample)
    return samples


def artifact_path(artifact_dir: Path, model_version: str) -> Path:
    safe = "-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if not model_version or any(character not in safe for character in model_version):
        raise ValueError("unsafe managed model version")
    return artifact_dir / f"{model_version}.json"


def _write_artifact(path: Path, artifact: ManagedModelArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(artifact.to_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "negative_infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def evaluate_challenger(
    engine: Engine,
    *,
    feature_schema_version: str,
    outcome_policy_version: str,
    policy: PromotionPolicy | None = None,
    include_hermes_context: bool = False,
    min_train_sessions: int = 15,
    embargo_sessions: int = 1,
) -> PromotionReport:
    """Rebuild deterministic out-of-fold evidence without registering a model."""
    samples = load_training_samples(
        engine,
        feature_schema_version=feature_schema_version,
        outcome_policy_version=outcome_policy_version,
        include_hermes_context=include_hermes_context,
        canonical_per_signal=False,
    )
    return walk_forward_evaluate(
        samples,
        policy=policy or PromotionPolicy(),
        min_train_sessions=min_train_sessions,
        embargo_sessions=embargo_sessions,
        feature_schema_version=feature_schema_version,
        outcome_policy_version=outcome_policy_version,
    )


def train_challenger(
    engine: Engine,
    artifact_dir: Path,
    *,
    model_version: str,
    feature_schema_version: str,
    outcome_policy_version: str,
    policy: PromotionPolicy | None = None,
    include_hermes_context: bool = False,
    min_train_sessions: int = 15,
    embargo_sessions: int = 1,
    now: datetime | None = None,
) -> TrainingRun:
    """Train, evaluate, checksum, and register one immutable challenger."""
    policy = policy or PromotionPolicy()
    samples = load_training_samples(
        engine,
        feature_schema_version=feature_schema_version,
        outcome_policy_version=outcome_policy_version,
        include_hermes_context=include_hermes_context,
        canonical_per_signal=False,
    )
    report = walk_forward_evaluate(
        samples,
        policy=policy,
        min_train_sessions=min_train_sessions,
        embargo_sessions=embargo_sessions,
        feature_schema_version=feature_schema_version,
        outcome_policy_version=outcome_policy_version,
    )
    if not samples or not report.rows:
        return TrainingRun(None, report, None, "insufficient_data")
    samples_by_id = {sample.opportunity_id: sample for sample in samples}
    residuals = [
        (row.realized_net_pnl - row.expected_value)
        / samples_by_id[row.opportunity_id].basis_dollars
        for row in report.rows
    ]
    residual_groups = [
        samples_by_id[row.opportunity_id].signal_id
        or samples_by_id[row.opportunity_id].opportunity_key
        for row in report.rows
    ]
    artifact = fit_managed_model(
        samples,
        model_version=model_version,
        feature_schema_version=feature_schema_version,
        outcome_policy_version=outcome_policy_version,
        ev_residuals=residuals,
        ev_residual_groups=residual_groups,
    )
    path = artifact_path(artifact_dir, model_version)
    if path.exists():
        existing = ManagedModelArtifact.from_json(path.read_text(encoding="utf-8"))
        if existing.artifact_hash != artifact.artifact_hash:
            raise ValueError("model version already exists with different artifact")
    else:
        _write_artifact(path, artifact)
    created_at = (now or datetime.now(UTC)).astimezone(UTC)
    metrics_payload = cast(
        dict[str, object],
        _json_safe(
            {
                "eligible": report.eligible,
                "reasons": report.reasons,
                "metrics": asdict(report.metrics),
                "include_hermes_context": include_hermes_context,
                "model_role": (CONTEXT_MODEL_ROLE if include_hermes_context else BASE_MODEL_ROLE),
                "promotion_allowed": not include_hermes_context,
                "deployment_scope": (
                    "shadow_reporting_only"
                    if include_hermes_context
                    else "paper_admission_candidate"
                ),
                "admission_selection_policy": (
                    policy.admission_selection_policy().payload()
                ),
                "artifact_file": path.name,
            }
        ),
    )
    status = "challenger" if report.eligible else "rejected"
    with engine.begin() as conn:
        result = conn.execute(
            insert(managed_models).values(
                model_version=model_version,
                artifact_hash=artifact.artifact_hash,
                feature_schema_version=feature_schema_version,
                outcome_policy_version=outcome_policy_version,
                trained_from_session=artifact.trained_from_session,
                trained_through_session=artifact.trained_through_session,
                metrics_json=metrics_payload,
                status=status,
                created_at=created_at,
            )
        )
        primary_key = result.inserted_primary_key
        if primary_key is None or primary_key[0] is None:
            raise RuntimeError("managed model registry insert returned no identity")
        model_id = int(primary_key[0])
        conn.execute(
            insert(managed_model_evaluations).values(
                model_id=model_id,
                evaluation_kind="walk_forward",
                fold_index=0,
                train_from_session=artifact.trained_from_session,
                train_through_session=artifact.trained_through_session,
                test_from_session=report.rows[0].session,
                test_through_session=report.rows[-1].session,
                metrics_json=metrics_payload,
                created_at=created_at,
            )
        )
    return TrainingRun(artifact, report, model_id, status)


def promote_challenger(
    engine: Engine,
    artifact_dir: Path,
    model_version: str,
    *,
    paper_only: bool,
    expected_admission_policy: AdmissionSelectionPolicy | None = None,
    now: datetime | None = None,
) -> ManagedModelArtifact:
    """Atomically promote a validated challenger, failing closed outside paper."""
    if not paper_only:
        raise ValueError("managed model promotion is restricted to paper execution")
    with engine.connect() as conn:
        row = (
            conn.execute(
                select(managed_models).where(managed_models.c.model_version == model_version)
            )
            .mappings()
            .one_or_none()
        )
        holdout = (
            conn.execute(
                select(managed_model_evaluations)
                .where(managed_model_evaluations.c.model_id == row["id"])
                .where(managed_model_evaluations.c.evaluation_kind == "holdout")
                .where(managed_model_evaluations.c.fold_index == 0)
            )
            .mappings()
            .one_or_none()
            if row is not None
            else None
        )
        incumbent = conn.execute(
            select(
                managed_models.c.model_version,
                managed_models.c.artifact_hash,
            ).where(managed_models.c.status == "promoted")
        ).one_or_none()
    if row is None or row["status"] != "challenger":
        raise ValueError("model is not an eligible challenger")
    try:
        _approved_base_metrics(
            row,
            expected_admission_policy=expected_admission_policy,
        )
    except ValueError as exc:
        raise ValueError(
            "only an explicitly registered causal-base challenger may be promoted"
        ) from exc
    artifact = ManagedModelArtifact.from_json(
        artifact_path(artifact_dir, model_version).read_text(encoding="utf-8")
    )
    _reject_context_artifact(artifact)
    if artifact.artifact_hash != row["artifact_hash"]:
        raise ValueError("artifact does not match immutable model registry")
    if artifact.model_version != row["model_version"]:
        raise ValueError("artifact model version differs from registry")
    if artifact.feature_schema_version != row["feature_schema_version"]:
        raise ValueError("artifact feature schema differs from registry")
    if artifact.outcome_policy_version != row["outcome_policy_version"]:
        raise ValueError("artifact outcome policy differs from registry")
    if artifact.trained_from_session != row["trained_from_session"]:
        raise ValueError("artifact training start differs from registry")
    if artifact.trained_through_session != row["trained_through_session"]:
        raise ValueError("artifact training end differs from registry")
    with engine.connect() as conn:
        evidence = _validated_prospective_holdout(
            conn,
            row,
            holdout,
            artifact,
            artifact_dir,
            require_promoted_at=False,
        )
    expected_incumbent = evidence.get("incumbent")
    if incumbent is None:
        if expected_incumbent is not None:
            raise ValueError("prospective holdout incumbent is no longer promoted")
    else:
        if not isinstance(expected_incumbent, Mapping):
            raise ValueError("replacement holdout omitted the promoted incumbent")
        if (
            expected_incumbent.get("model_version") != incumbent.model_version
            or expected_incumbent.get("artifact_hash") != incumbent.artifact_hash
            or expected_incumbent.get("incremental_eligible") is not True
        ):
            raise ValueError("replacement holdout does not validate current incumbent")
    promoted_at = (now or datetime.now(UTC)).astimezone(UTC)
    evaluated_at = _as_utc(holdout["created_at"]) if holdout is not None else None
    if evaluated_at is None or promoted_at < evaluated_at:
        raise ValueError("promotion cannot predate its prospective holdout evidence")
    with engine.begin() as conn:
        # Rebind the immutable evaluation to its actual cohort in the same
        # transaction that changes the production model pointer.  This closes
        # the gap between the preflight read and the status update.
        _validated_prospective_holdout(
            conn,
            row,
            holdout,
            artifact,
            artifact_dir,
            require_promoted_at=False,
        )
        current_incumbent = conn.execute(
            select(
                managed_models.c.model_version,
                managed_models.c.artifact_hash,
            ).where(managed_models.c.status == "promoted")
        ).one_or_none()
        expected_identity = (
            (str(incumbent.model_version), str(incumbent.artifact_hash))
            if incumbent is not None
            else None
        )
        current_identity = (
            (
                str(current_incumbent.model_version),
                str(current_incumbent.artifact_hash),
            )
            if current_incumbent is not None
            else None
        )
        if current_identity != expected_identity:
            raise RuntimeError("promoted incumbent changed during challenger promotion")
        conn.execute(
            update(managed_models)
            .where(managed_models.c.status == "promoted")
            .values(status="retired")
        )
        result = conn.execute(
            update(managed_models)
            .where(managed_models.c.id == int(row["id"]))
            .where(managed_models.c.status == "challenger")
            .values(status="promoted", promoted_at=promoted_at)
        )
        if result.rowcount != 1:
            raise RuntimeError("managed model promotion lost a concurrent race")
    return artifact


def load_promoted_model(
    engine: Engine,
    artifact_dir: Path,
    *,
    expected_admission_policy: AdmissionSelectionPolicy | None = None,
) -> ManagedModelArtifact | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(managed_models).where(managed_models.c.status == "promoted"))
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        holdout = (
            conn.execute(
                select(managed_model_evaluations)
                .where(managed_model_evaluations.c.model_id == row["id"])
                .where(managed_model_evaluations.c.evaluation_kind == "holdout")
                .where(managed_model_evaluations.c.fold_index == 0)
            )
            .mappings()
            .one_or_none()
        )
        _approved_base_metrics(
            row,
            expected_admission_policy=expected_admission_policy,
        )
        artifact = ManagedModelArtifact.from_json(
            artifact_path(artifact_dir, str(row["model_version"])).read_text(encoding="utf-8")
        )
        _reject_context_artifact(artifact)
        if artifact.artifact_hash != row["artifact_hash"]:
            raise ValueError("promoted artifact hash differs from registry")
        if artifact.model_version != row["model_version"]:
            raise ValueError("promoted artifact model version differs from registry")
        if artifact.feature_schema_version != row["feature_schema_version"]:
            raise ValueError("promoted artifact feature schema differs from registry")
        if artifact.outcome_policy_version != row["outcome_policy_version"]:
            raise ValueError("promoted artifact outcome policy differs from registry")
        if artifact.trained_from_session != row["trained_from_session"]:
            raise ValueError("promoted artifact training start differs from registry")
        if artifact.trained_through_session != row["trained_through_session"]:
            raise ValueError("promoted artifact training end differs from registry")
        _validated_prospective_holdout(
            conn,
            row,
            holdout,
            artifact,
            artifact_dir,
            require_promoted_at=True,
        )
    return artifact


def _reject_context_artifact(artifact: ManagedModelArtifact) -> None:
    """Fail closed if context features reach the live model-loading boundary."""
    if "+hermes_context" in artifact.feature_schema_version or any(
        name.startswith("hermes.") for name in artifact.encoder.names
    ):
        raise ValueError("Hermes-context artifacts are shadow-only in this release")
