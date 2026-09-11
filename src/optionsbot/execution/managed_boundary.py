"""Fail-closed managed-model boundary for managed exact-0DTE entries.

Scan-time probabilities are evidence, not durable order authority.  This
module binds an executable candidate to the one currently promoted causal-base
artifact and rebuilds its prediction from the immutable decision-time feature
payload plus the latest executable option economics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, Engine, select
from sqlalchemy.engine import RowMapping

from optionsbot.admission_policy import configured_admission_policy
from optionsbot.config import Settings
from optionsbot.learning.features import model_features
from optionsbot.learning.managed_model import ManagedPrediction, predict_managed_outcome
from optionsbot.learning.repository import load_promoted_model
from optionsbot.opening_range_economics import PAPER_RULE_EXPECTED_VALUE_MODEL
from optionsbot.storage.schema import managed_opportunities, snapshots, strategy_scores


class ManagedExecutionError(ValueError):
    """The managed-model packet cannot authorize an exact-0DTE entry."""


_MANAGED_PACKET_IDENTITY_FIELDS = (
    "managed_probability_model",
    "managed_model_artifact_hash",
    "managed_feature_schema_version",
    "managed_outcome_policy_version",
    "managed_model_trained_through",
)
_ACTIVE_MANAGED_STATUSES = ("pending_entry", "active")


def carries_managed_model_packet(suggestion: Mapping[str, object]) -> bool:
    """Return whether a suggestion claims managed-model admission evidence."""
    return any(
        suggestion.get(name) is not None and suggestion.get(name) != ""
        for name in _MANAGED_PACKET_IDENTITY_FIELDS
    )


def carries_paper_rule_packet(suggestion: Mapping[str, object]) -> bool:
    """Return whether a suggestion claims deterministic paper-rule authority."""
    return suggestion.get("expected_value_model") == PAPER_RULE_EXPECTED_VALUE_MODEL


def _utc_timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ManagedExecutionError(f"managed opportunity {name} is missing")
    return (
        value.replace(tzinfo=UTC)
        if value.tzinfo is None
        else value.astimezone(UTC)
    )


def _managed_binding_rows(conn: Connection, score_id: int) -> list[RowMapping]:
    return list(
        conn.execute(
            select(
                managed_opportunities.c.strategy_score_id,
                managed_opportunities.c.signal_id,
                managed_opportunities.c.status,
                managed_opportunities.c.symbol,
                managed_opportunities.c.session,
                managed_opportunities.c.direction,
                managed_opportunities.c.setup_type,
                managed_opportunities.c.strategy,
                managed_opportunities.c.structure_hash,
                managed_opportunities.c.legs_json,
                managed_opportunities.c.features_json,
                managed_opportunities.c.policy_version,
                managed_opportunities.c.detected_at,
                managed_opportunities.c.entry_cutoff_at,
                managed_opportunities.c.admission_eligible,
                managed_opportunities.c.shadow_only,
                managed_opportunities.c.bot_action,
                managed_opportunities.c.bot_reason,
                managed_opportunities.c.bot_decided_at,
                managed_opportunities.c.stop_pct,
                managed_opportunities.c.target_pct,
                strategy_scores.c.snapshot_id.label("score_snapshot_id"),
                strategy_scores.c.strategy.label("score_strategy"),
                strategy_scores.c.legs_json.label("score_legs_json"),
                strategy_scores.c.suggestion_json.label("score_suggestion_json"),
                snapshots.c.symbol.label("score_symbol"),
            )
            .join(
                strategy_scores,
                managed_opportunities.c.strategy_score_id == strategy_scores.c.id,
            )
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(strategy_scores.c.id == score_id)
        ).mappings()
    )


def _validate_bound_row(
    row: RowMapping,
    score_id: int,
    *,
    now: datetime,
) -> None:
    """Validate immutable score identity and its first admission disposition."""
    if row["status"] not in _ACTIVE_MANAGED_STATUSES:
        raise ManagedExecutionError(
            "managed opportunity is no longer active execution evidence"
        )
    if row["admission_eligible"] != 1 or row["shadow_only"] != 0:
        raise ManagedExecutionError(
            "managed opportunity is immutable research-only evidence"
        )
    if (
        row["bot_action"] != "candidate"
        or not isinstance(row["bot_reason"], str)
        or not row["bot_reason"]
        or row["bot_decided_at"] is None
    ):
        raise ManagedExecutionError(
            "managed opportunity lacks an immutable candidate admission disposition"
        )
    detected_at = _utc_timestamp(row["detected_at"], name="detected_at")
    cutoff_at = _utc_timestamp(row["entry_cutoff_at"], name="entry_cutoff_at")
    decided_at = _utc_timestamp(row["bot_decided_at"], name="bot_decided_at")
    execution_now = _utc_timestamp(now, name="execution clock")
    if decided_at < detected_at:
        raise ManagedExecutionError("managed candidate disposition predates detection")
    if decided_at >= cutoff_at:
        raise ManagedExecutionError("managed candidate disposition is at or after cutoff")
    if decided_at > execution_now:
        raise ManagedExecutionError("managed candidate disposition is future-dated")
    if execution_now >= cutoff_at:
        raise ManagedExecutionError(
            "managed candidate execution is at or after its frozen entry cutoff"
        )
    if row["strategy_score_id"] != score_id:
        raise ManagedExecutionError("managed opportunity was rebound to another strategy score")
    if row["score_strategy"] != row["strategy"]:
        raise ManagedExecutionError("executable strategy differs from captured strategy")
    if str(row["score_symbol"]).strip().upper() != str(row["symbol"]).strip().upper():
        raise ManagedExecutionError("executable symbol differs from captured symbol")
    features_payload = row["features_json"]
    if not isinstance(features_payload, Mapping):
        raise ManagedExecutionError("managed opportunity feature payload is malformed")
    if features_payload.get("snapshot_id") != row["score_snapshot_id"]:
        raise ManagedExecutionError("executable score differs from captured snapshot identity")
    captured_legs = _canonical_option_legs(row["legs_json"])
    current_legs = _canonical_option_legs(row["score_legs_json"])
    if not captured_legs or not current_legs:
        raise ManagedExecutionError("managed opportunity option structure is malformed")
    captured_symbol = str(row["symbol"]).strip().upper()
    if any(leg["symbol"] != captured_symbol for leg in captured_legs):
        raise ManagedExecutionError(
            "managed opportunity option legs differ from the captured symbol"
        )
    captured_hash = _structure_hash(captured_legs)
    if captured_hash != row["structure_hash"]:
        raise ManagedExecutionError("captured option structure hash is invalid")
    if current_legs != captured_legs or _structure_hash(current_legs) != captured_hash:
        raise ManagedExecutionError("executable option legs differ from captured structure")


def validate_managed_stage_authorization(
    conn: Connection,
    score_id: int,
    suggestion: Mapping[str, object],
    *,
    now: datetime,
    allow_paper_rule: bool = False,
) -> bool:
    """Enforce a managed admission binding in the order-insert transaction.

    Returns ``False`` only for a genuinely legacy/non-managed score. A linked
    opportunity or a claimed managed packet always fails closed unless exactly
    one complete, timely candidate binding exists.
    """
    rows = _managed_binding_rows(conn, score_id)
    if not rows:
        if carries_managed_model_packet(suggestion) or carries_paper_rule_packet(
            suggestion
        ):
            raise ManagedExecutionError(
                "managed admission packet has no immutable opportunity binding"
            )
        return False
    if len(rows) != 1:
        raise ManagedExecutionError(
            "managed order staging requires exactly one immutable opportunity"
        )
    _validate_bound_row(rows[0], score_id, now=now)
    model_claim = carries_managed_model_packet(suggestion)
    paper_claim = carries_paper_rule_packet(suggestion)
    if model_claim and paper_claim:
        raise ManagedExecutionError(
            "managed model identity cannot also claim deterministic paper authority"
        )
    if model_claim:
        for name in _MANAGED_PACKET_IDENTITY_FIELDS:
            _required_text(suggestion, name)
        _validate_three_event_packet(suggestion)
        return True
    if allow_paper_rule and paper_claim:
        _validate_paper_rule_packet(suggestion, rows[0])
        return True
    if paper_claim:
        raise ManagedExecutionError(
            "deterministic paper candidate lacks the runtime paper authorization"
        )
    raise ManagedExecutionError(
        "managed candidate lacks a complete managed model packet"
    )


def _validate_paper_rule_packet(
    payload: Mapping[str, object],
    row: RowMapping,
) -> None:
    """Validate explicit, non-model OR/FVG paper admission evidence."""
    if carries_managed_model_packet(payload):
        raise ManagedExecutionError(
            "managed model identity cannot fall back to deterministic paper authority"
        )
    opening_plan = payload.get("opening_range_fvg")
    managed_plan = payload.get("managed_signal_plan")
    if not isinstance(opening_plan, Mapping) or not isinstance(managed_plan, Mapping):
        raise ManagedExecutionError("deterministic paper plan is missing")
    signal_id = opening_plan.get("signal_id")
    session = opening_plan.get("session")
    direction = opening_plan.get("direction")
    setup_type = opening_plan.get("setup_type")
    session_expiry = str(row["session"]).replace("-", "")
    captured_legs = _canonical_option_legs(row["legs_json"])
    allowed_strategies = {
        "bull": {"bull_call_spread", "long_call"},
        "bear": {"bear_put_spread", "long_put"},
    }
    if (
        opening_plan.get("status") != "entry_confirmed"
        or opening_plan.get("source") != "trusted_daemon"
        or managed_plan.get("schema_version") != "managed_signal_plan_v1"
        or managed_plan.get("status") != "entry_confirmed"
        or managed_plan.get("source") != "trusted_daemon"
        or managed_plan.get("generator") != "opening_range_fvg"
        or managed_plan.get("admission_enabled") is not True
        or not isinstance(signal_id, str)
        or not signal_id
        or managed_plan.get("signal_id") != signal_id
        or managed_plan.get("session") != session
        or managed_plan.get("direction") != direction
        or managed_plan.get("setup_type") != setup_type
        or signal_id != row["signal_id"]
        or session != row["session"]
        or direction != row["direction"]
        or setup_type != row["setup_type"]
        or managed_plan.get("option_expiry") != session_expiry
        or not captured_legs
        or {str(leg["expiry"]) for leg in captured_legs} != {session_expiry}
        or direction not in allowed_strategies
        or row["strategy"] not in allowed_strategies[str(direction)]
    ):
        raise ManagedExecutionError(
            "deterministic paper plan lacks trusted OR/FVG identity"
        )
    expected_value = _finite(payload.get("expected_value"))
    terminal_value = _finite(payload.get("terminal_expected_value"))
    costs = _finite(payload.get("estimated_round_trip_cost"))
    prob_profit = _finite(payload.get("prob_profit"))
    opening_stop = _finite(opening_plan.get("stop_pct"))
    opening_target = _finite(opening_plan.get("target_pct"))
    opening_target_r = _finite(opening_plan.get("target_r"))
    managed_stop = _finite(managed_plan.get("stop_pct"))
    managed_target = _finite(managed_plan.get("target_pct"))
    managed_target_r = _finite(managed_plan.get("target_r"))
    row_stop = _finite(row["stop_pct"])
    row_target = _finite(row["target_pct"])
    plan_numbers = (
        opening_stop,
        opening_target,
        opening_target_r,
        managed_stop,
        managed_target,
        managed_target_r,
        row_stop,
        row_target,
    )
    if (
        expected_value is None
        or expected_value <= 0.0
        or terminal_value is None
        or costs is None
        or costs < 0.0
        or prob_profit is None
        or not 0.0 < prob_profit < 1.0
        or any(value is None for value in plan_numbers)
        or not math.isclose(
            expected_value,
            terminal_value - costs,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ManagedExecutionError(
            "deterministic paper expectancy is missing, non-positive, or inconsistent"
        )
    assert opening_stop is not None
    assert opening_target is not None
    assert opening_target_r is not None
    assert managed_stop is not None
    assert managed_target is not None
    assert managed_target_r is not None
    assert row_stop is not None
    assert row_target is not None
    if (
        not math.isclose(opening_stop, managed_stop, rel_tol=1e-9)
        or not math.isclose(opening_stop, row_stop, rel_tol=1e-9)
        or not math.isclose(opening_target, managed_target, rel_tol=1e-9)
        or not math.isclose(opening_target, row_target, rel_tol=1e-9)
        or not math.isclose(opening_target_r, managed_target_r, rel_tol=1e-9)
        or not math.isclose(
            opening_target,
            opening_stop * opening_target_r,
            rel_tol=1e-9,
        )
    ):
        raise ManagedExecutionError(
            "deterministic paper stop/target plan differs from its immutable binding"
        )


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _canonical_option_legs(raw_legs: object) -> list[dict[str, Any]]:
    """Normalize the executable score structure to the capture contract."""
    if not isinstance(raw_legs, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw in raw_legs:
        if not isinstance(raw, Mapping):
            return []
        try:
            symbol = str(raw["symbol"]).strip().upper()
            side = str(raw["side"]).strip().lower()
            sec_type = str(raw.get("sec_type", "OPT")).strip().upper()
            expiry = str(raw["expiry"])
            strike = float(raw["strike"])
            right = str(raw["right"]).strip().upper()
            quantity = int(raw.get("quantity", 1))
        except (KeyError, TypeError, ValueError, OverflowError):
            return []
        if (
            not symbol
            or side not in {"buy", "sell"}
            or sec_type != "OPT"
            or len(expiry) != 8
            or not expiry.isdigit()
            or not math.isfinite(strike)
            or strike <= 0
            or right not in {"C", "P"}
            or quantity <= 0
        ):
            return []
        normalized.append(
            {
                "symbol": symbol,
                "side": side,
                "sec_type": sec_type,
                "expiry": expiry,
                "strike": strike,
                "right": right,
                "quantity": quantity,
            }
        )
    return sorted(
        normalized,
        key=lambda leg: (
            leg["symbol"],
            leg["expiry"],
            leg["right"],
            leg["strike"],
            leg["side"],
            leg["quantity"],
        ),
    )


def _structure_hash(legs: object) -> str:
    encoded = json.dumps(
        legs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ManagedExecutionError(f"managed model packet is missing {name}")
    return value


def _validate_three_event_packet(payload: Mapping[str, object]) -> None:
    probabilities: list[float] = []
    for name in (
        "managed_target_hit_probability",
        "managed_stop_probability",
        "managed_timeout_probability",
    ):
        value = _finite(payload.get(name))
        if value is None or not 0.0 <= value <= 1.0:
            raise ManagedExecutionError(
                "managed exact-0DTE entry requires a complete three-event "
                f"managed model packet ({name} is missing or invalid)"
            )
        probabilities.append(value)
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ManagedExecutionError(
            "managed target/stop/timeout probabilities do not sum to one"
        )


def refresh_managed_prediction(
    engine: Engine,
    settings: Settings,
    score_id: int,
    suggestion: Mapping[str, object],
    *,
    fresh_basis_dollars: float,
    fresh_costs: float,
    maximum_profit: float | None,
    now: datetime | None = None,
) -> ManagedPrediction:
    """Verify current promotion identity and re-infer with fresh economics.

    The feature payload and setup identity come from the immutable prospective
    capture row.  Only basis and the conservative round-trip cost reserve are
    replaced with values derived from the latest executable NBBO.
    """
    if settings.managed_learning.artifact_dir is None:
        raise ManagedExecutionError("managed model artifact directory is unavailable")
    if (
        not math.isfinite(fresh_basis_dollars)
        or fresh_basis_dollars <= 0.0
        or not math.isfinite(fresh_costs)
        or fresh_costs < 0.0
    ):
        raise ManagedExecutionError("fresh managed basis or costs are invalid")
    _validate_three_event_packet(suggestion)

    try:
        artifact = load_promoted_model(
            engine,
            settings.managed_learning.artifact_dir,
            expected_admission_policy=configured_admission_policy(settings),
        )
    except Exception as exc:  # corrupt/tampered files and registry mismatches fail closed
        raise ManagedExecutionError(
            f"current promoted managed artifact failed verification: {exc}"
        ) from exc
    if artifact is None:
        raise ManagedExecutionError("no causal-base managed model is currently promoted")

    candidate_identity = {
        "model version": _required_text(suggestion, "managed_probability_model"),
        "artifact hash": _required_text(suggestion, "managed_model_artifact_hash"),
        "feature schema": _required_text(
            suggestion, "managed_feature_schema_version"
        ),
        "outcome policy": _required_text(
            suggestion, "managed_outcome_policy_version"
        ),
        "trained-through session": _required_text(
            suggestion, "managed_model_trained_through"
        ),
    }
    current_identity = {
        "model version": artifact.model_version,
        "artifact hash": artifact.artifact_hash,
        "feature schema": artifact.feature_schema_version,
        "outcome policy": artifact.outcome_policy_version,
        "trained-through session": artifact.trained_through_session,
    }
    for name, expected in current_identity.items():
        if candidate_identity[name] != expected:
            raise ManagedExecutionError(
                f"candidate managed {name} is no longer the promoted artifact"
            )
    if suggestion.get("expected_value_model") != artifact.model_version:
        raise ManagedExecutionError(
            "candidate expected-value model does not match the promoted artifact"
        )
    if (
        artifact.feature_schema_version
        != settings.managed_learning.feature_schema_version
        or artifact.outcome_policy_version
        != settings.managed_learning.outcome_policy_version
    ):
        raise ManagedExecutionError(
            "promoted managed artifact does not match configured schema/policy"
        )

    execution_now = now or datetime.now(UTC)
    with engine.connect() as conn:
        rows = _managed_binding_rows(conn, score_id)
    if len(rows) != 1:
        raise ManagedExecutionError(
            "managed exact-0DTE entry requires one immutable managed opportunity"
        )
    row = rows[0]
    _validate_bound_row(row, score_id, now=execution_now)
    features_payload = row["features_json"]
    stored_suggestion = row["score_suggestion_json"]
    if not isinstance(stored_suggestion, Mapping):
        raise ManagedExecutionError("persisted managed model packet is malformed")
    for name in _MANAGED_PACKET_IDENTITY_FIELDS:
        if stored_suggestion.get(name) != suggestion.get(name):
            raise ManagedExecutionError(
                f"managed model packet changed after the score was read ({name})"
            )
    if features_payload.get("feature_schema_version") != artifact.feature_schema_version:
        raise ManagedExecutionError(
            "managed opportunity feature schema differs from promoted artifact"
        )
    if row["policy_version"] != artifact.outcome_policy_version:
        raise ManagedExecutionError(
            "managed opportunity outcome policy differs from promoted artifact"
        )
    if str(row["session"]) <= artifact.trained_through_session:
        raise ManagedExecutionError(
            "promoted artifact evidence window overlaps the candidate session"
        )

    stop_pct = _finite(row["stop_pct"])
    target_pct = _finite(row["target_pct"])
    if (
        stop_pct is None
        or target_pct is None
        or stop_pct <= 0.0
        or target_pct <= 0.0
    ):
        raise ManagedExecutionError("managed opportunity payoff policy is invalid")
    target_gain = fresh_basis_dollars * target_pct
    if (
        maximum_profit is not None
        and (
            not math.isfinite(maximum_profit)
            or maximum_profit <= 0.0
            or target_gain + fresh_costs > maximum_profit
        )
    ):
        raise ManagedExecutionError(
            "fresh managed target is not attainable after round-trip costs"
        )

    try:
        features = model_features(
            features_payload,
            basis_dollars=fresh_basis_dollars,
            stop_pct=stop_pct,
            target_pct=target_pct,
            commission_estimate=fresh_costs,
            direction=str(row["direction"]),
            setup_type=str(row["setup_type"]),
            strategy=str(row["strategy"]),
        )
        prediction = predict_managed_outcome(
            artifact,
            features,
            basis_dollars=fresh_basis_dollars,
            target_gain=target_gain,
            stop_loss=fresh_basis_dollars * stop_pct,
            costs=fresh_costs,
        )
    except Exception as exc:
        raise ManagedExecutionError(
            f"fresh managed prediction failed verification: {exc}"
        ) from exc
    if (
        prediction.model_version != artifact.model_version
        or prediction.artifact_hash != artifact.artifact_hash
    ):
        raise ManagedExecutionError("fresh prediction artifact identity changed")
    if (
        not math.isfinite(prediction.expected_value_lcb)
        or prediction.expected_value_lcb <= 0.0
    ):
        raise ManagedExecutionError(
            "fresh managed after-cost expected-value lower bound is not positive"
        )
    return prediction
