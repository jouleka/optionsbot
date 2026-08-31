"""Apply a promoted causal managed-path model to persisted scan candidates."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from sqlalchemy import Engine, exists, select, update

from optionsbot.admission_policy import configured_admission_policy
from optionsbot.config import Settings
from optionsbot.learning.features import model_features
from optionsbot.learning.managed_model import predict_managed_outcome
from optionsbot.learning.repository import load_promoted_model
from optionsbot.scoring import ScoredStrategy
from optionsbot.storage.schema import managed_opportunities, strategy_scores

log = logging.getLogger(__name__)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _basis(suggestion: Mapping[str, object]) -> float | None:
    explicit = _finite(suggestion.get("managed_marketable_basis_dollars"))
    if explicit is not None and explicit > 0.0:
        return explicit
    cashflow = _finite(suggestion.get("credit_or_debit"))
    if cashflow is None or cashflow >= 0.0:
        return None
    return abs(cashflow)


def apply_promoted_managed_model(
    engine: Engine,
    settings: Settings,
    snapshot_id: int,
    scored: Sequence[ScoredStrategy],
) -> tuple[ScoredStrategy, ...]:
    """Revalue production candidates with the promoted three-event model.

    Capture has already frozen each opportunity before this function runs.
    Failures leave the managed EV unavailable, so the existing positive-edge
    gate holds the candidate. Shadow structure-grid rows are not present in
    ``scored`` and therefore cannot become executable here.
    """
    if not scored:
        return tuple(scored)
    artifact_dir = settings.managed_learning.artifact_dir
    if artifact_dir is None:
        return tuple(scored)
    try:
        artifact = load_promoted_model(
            engine,
            artifact_dir,
            expected_admission_policy=configured_admission_policy(settings),
        )
    except Exception:  # noqa: BLE001 -- corrupt/missing artifacts fail closed
        log.exception("promoted managed model could not be loaded")
        return tuple(scored)
    if artifact is None:
        return tuple(scored)
    if (
        artifact.feature_schema_version
        != settings.managed_learning.feature_schema_version
        or artifact.outcome_policy_version
        != settings.managed_learning.outcome_policy_version
    ):
        log.error(
            "promoted managed model schema/policy mismatch: model=%s",
            artifact.model_version,
        )
        return tuple(scored)

    with engine.connect() as conn:
        rows = conn.execute(
            select(
                managed_opportunities.c.id.label("opportunity_id"),
                managed_opportunities.c.session,
                managed_opportunities.c.direction,
                managed_opportunities.c.setup_type,
                managed_opportunities.c.strategy,
                managed_opportunities.c.features_json,
                managed_opportunities.c.stop_pct,
                managed_opportunities.c.target_pct,
                managed_opportunities.c.commission_estimate,
                managed_opportunities.c.admission_eligible,
                managed_opportunities.c.shadow_only,
                managed_opportunities.c.bot_decided_at,
                strategy_scores.c.id.label("score_id"),
                strategy_scores.c.suggestion_json,
            )
            .join(
                strategy_scores,
                managed_opportunities.c.strategy_score_id == strategy_scores.c.id,
            )
            .where(strategy_scores.c.snapshot_id == snapshot_id)
            .where(managed_opportunities.c.admission_eligible == 1)
            .where(managed_opportunities.c.shadow_only == 0)
            .where(managed_opportunities.c.bot_decided_at.is_(None))
        ).mappings().all()
    by_strategy = {str(row["strategy"]): row for row in rows}
    replacements: dict[str, ScoredStrategy] = {}
    persisted: list[tuple[int, str, dict[str, object], dict[str, object]]] = []
    for item in scored:
        row = by_strategy.get(item.strategy_name)
        if row is None:
            continue
        # A model may only score sessions strictly after its evidence window.
        if str(row["session"]) <= artifact.trained_through_session:
            log.error(
                "managed model temporal guard held %s: session=%s trained_through=%s",
                item.strategy_name,
                row["session"],
                artifact.trained_through_session,
            )
            continue
        stored = dict(row["suggestion_json"] or {})
        basis = _basis(stored)
        costs = _finite(row["commission_estimate"])
        stop_pct = _finite(row["stop_pct"])
        target_pct = _finite(row["target_pct"])
        if (
            basis is None
            or costs is None
            or stop_pct is None
            or target_pct is None
            or stop_pct <= 0.0
            or target_pct <= 0.0
        ):
            continue
        maximum_profit = _finite(stored.get("max_profit"))
        if maximum_profit is not None and basis * target_pct + costs > maximum_profit:
            stored.update(
                expected_value=None,
                managed_admission_reason="target_not_reachable_after_commissions",
            )
            persisted.append(
                (
                    int(row["score_id"]),
                    item.strategy_name,
                    dict(row["suggestion_json"] or {}),
                    stored,
                )
            )
            replacements[item.strategy_name] = replace(
                item,
                suggestion=replace(item.suggestion, expected_value=None),
            )
            continue
        features_raw = row["features_json"]
        features = model_features(
            features_raw if isinstance(features_raw, Mapping) else {},
            basis_dollars=basis,
            stop_pct=stop_pct,
            target_pct=target_pct,
            commission_estimate=costs,
            direction=str(row["direction"]),
            setup_type=str(row["setup_type"]),
            strategy=str(row["strategy"]),
        )
        prediction = predict_managed_outcome(
            artifact,
            features,
            basis_dollars=basis,
            target_gain=basis * target_pct,
            stop_loss=basis * stop_pct,
            costs=costs,
        )
        conservative_ev = (
            prediction.expected_value_lcb
            if math.isfinite(prediction.expected_value_lcb)
            else None
        )
        stored.update(
            expected_value=conservative_ev,
            expected_value_model=prediction.model_version,
            managed_target_hit_probability=prediction.target_probability,
            managed_target_hit_probability_lcb=prediction.target_probability_lcb,
            managed_stop_probability=prediction.stop_probability,
            managed_timeout_probability=prediction.timeout_probability,
            managed_expected_value=prediction.expected_value,
            managed_expected_value_lcb=conservative_ev,
            managed_timeout_expected_return=artifact.timeout_expected_return,
            managed_ev_residual_return_q05=artifact.ev_residual_q05,
            managed_probability_model=prediction.model_version,
            managed_model_artifact_hash=prediction.artifact_hash,
            managed_feature_schema_version=artifact.feature_schema_version,
            managed_outcome_policy_version=artifact.outcome_policy_version,
            managed_model_trained_through=artifact.trained_through_session,
            managed_admission_reason=(
                "promoted_model_positive_after_cost_lcb"
                if conservative_ev is not None and conservative_ev > 0.0
                else "promoted_model_non_positive_after_cost_lcb"
            ),
        )
        persisted.append(
            (
                int(row["score_id"]),
                item.strategy_name,
                dict(row["suggestion_json"] or {}),
                stored,
            )
        )
        replacements[item.strategy_name] = replace(
            item,
            suggestion=replace(item.suggestion, expected_value=conservative_ev),
        )
    if persisted:
        with engine.begin() as conn:
            for score_id, strategy_name, original, payload in persisted:
                bound_unconsumed = exists(
                    select(managed_opportunities.c.id)
                    .where(managed_opportunities.c.strategy_score_id == score_id)
                    .where(managed_opportunities.c.admission_eligible == 1)
                    .where(managed_opportunities.c.shadow_only == 0)
                    .where(managed_opportunities.c.bot_decided_at.is_(None))
                )
                result = conn.execute(
                    update(strategy_scores)
                    .where(strategy_scores.c.id == score_id)
                    .where(strategy_scores.c.suggestion_json == original)
                    .where(bound_unconsumed)
                    .values(suggestion_json=payload)
                )
                if result.rowcount != 1:
                    # Another disposition/update won the race. Do not return a
                    # model-valued in-memory candidate that was not frozen in
                    # the exact score row.
                    replacements.pop(strategy_name, None)
    return tuple(replacements.get(item.strategy_name, item) for item in scored)
