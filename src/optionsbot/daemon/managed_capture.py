"""Prospective executable-path capture for managed 0DTE opportunities.

This module is deliberately observational.  It creates no order, does not
write a production probability, and never reads or changes execution state.
Confirmed candidates are registered before alert/EV/Hermes admission, then a
bounded scheduler pass samples marketable synthetic-combo NBBO marks.  Missing
or ambiguous observations are persisted and censored rather than guessed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, Engine, literal, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from optionsbot.config import Settings
from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.risk_structure import structural_max_profit_dollars
from optionsbot.execution.walk import combo_bid_ask
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.types import OptionQuote
from optionsbot.learning.features import build_capture_feature_payload
from optionsbot.managed_contract import (
    ManagedOutcomePolicySpec,
    validate_managed_contract,
)
from optionsbot.market_hours import nyse_session_close_utc
from optionsbot.storage.schema import (
    managed_opportunities,
    managed_opportunity_marks,
    snapshots,
    strategy_scores,
)
from optionsbot.strategies import all_strategies

log = logging.getLogger(__name__)

_ACTIVE_STATUSES = ("pending_entry", "active")
_CAPACITY_REALLOCATION_REASON = "managed_capture_capacity_reallocated_for_independent_signal"
_MAX_CONCURRENT_QUOTE_REQUESTS = 4
_PER_QUOTE_TIMEOUT_SECONDS = 1.0
_MANAGED_PLAN_SCHEMA_VERSION = "managed_signal_plan_v1"
_OPENING_RANGE_GENERATOR = "opening_range_fvg"
_SHADOW_GRID_PREFIX = "shadow_grid_v1:"
_PRIMARY_STRATEGIES = frozenset(strategy.name for strategy in all_strategies())
LegSpec = tuple[str, str, float, str]


@dataclass(frozen=True, slots=True)
class ManagedCaptureSummary:
    opportunities_seen: int
    usable_marks: int
    unusable_marks: int
    resolved: int
    censored: int
    skipped_for_trading: bool
    quote_errors: int


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _managed_plan_for_row(
    suggestion: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return only the daemon-authored row-local managed identity.

    Snapshot-wide and legacy ``opening_range_fvg`` payloads are descriptive
    evidence, not an admission binding.  Reusing one of those shared payloads
    for every score row would let a later or research-only score inherit a
    production signal identity it never received.
    """
    row_plan = suggestion.get("managed_signal_plan")
    return row_plan if isinstance(row_plan, Mapping) else None


def _immutable_admission_classification(
    *,
    plan: Mapping[str, Any],
    suggestion: Mapping[str, Any],
    strategy: str,
) -> tuple[bool, bool]:
    """Classify the frozen row without trusting mutable model output.

    Only the established production OR/FVG generator on a registered primary
    strategy can be admission eligible.  Every research provenance marker is
    one-way: it can force shadow status but can never be counteracted by a
    second flag that claims production authority.
    """
    generator = plan.get("generator")
    shadow_marker = (
        plan.get("status") == "shadow_confirmed"
        or generator != _OPENING_RANGE_GENERATOR
        or strategy.startswith(_SHADOW_GRID_PREFIX)
        or strategy not in _PRIMARY_STRATEGIES
        or suggestion.get("shadow_only") is True
        or suggestion.get("admission_enabled") is False
        or suggestion.get("shadow_schema_version") is not None
        or suggestion.get("shadow_candidate_id") is not None
        or suggestion.get("shadow_strategy") is not None
        or suggestion.get("risk_tier") == "research_only"
    )
    eligible = (
        not shadow_marker
        and plan.get("schema_version") == _MANAGED_PLAN_SCHEMA_VERSION
        and plan.get("source") == "trusted_daemon"
        and plan.get("status") == "entry_confirmed"
        and plan.get("admission_enabled") is True
        and generator == _OPENING_RANGE_GENERATOR
        and strategy in _PRIMARY_STRATEGIES
    )
    return eligible, not eligible


def canonical_legs(raw_legs: object) -> list[dict[str, Any]]:
    """Return a stable semantic ordering for a persisted option structure."""
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


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _managed_policy_version(settings: Settings) -> str:
    """Return the validated identity of the currently configured label policy."""
    validate_managed_contract(
        feature_schema_version=settings.managed_learning.feature_schema_version,
        outcome_policy_version=settings.managed_learning.outcome_policy_version,
        outcome_policy_spec=ManagedOutcomePolicySpec(
            capture_interval_seconds=settings.validation.managed_capture_interval_seconds,
            capture_offset_seconds=settings.validation.managed_capture_offset_seconds,
            quote_max_age_seconds=settings.validation.managed_capture_quote_max_age_seconds,
            quote_span_seconds=settings.validation.managed_capture_quote_span_seconds,
            max_mark_gap_seconds=settings.validation.managed_capture_max_mark_gap_seconds,
        ),
    )
    return settings.managed_learning.outcome_policy_version


def opportunity_key(signal_id: str, strategy: str, *, policy_version: str) -> str:
    """One frozen structure per signal/strategy, independent of rescans."""
    return _json_hash(
        {
            "policy_version": policy_version,
            "signal_id": signal_id,
            "strategy": strategy,
        }
    )


def _baseline(score: float, suggestion: Mapping[str, Any], settings: Settings) -> tuple[str, str]:
    if suggestion.get("shadow_only") is True:
        shadow_reason = suggestion.get("shadow_reason")
        return (
            "hold",
            shadow_reason
            if isinstance(shadow_reason, str) and shadow_reason
            else "shadow_structure_not_admission_eligible",
        )
    reasons: list[str] = []
    expected_value = _finite(suggestion.get("expected_value"))
    if score < settings.scan.score_threshold:
        reasons.append("score_below_floor")
    if expected_value is None:
        reasons.append("managed_expected_value_unavailable")
    elif expected_value <= 0:
        reasons.append("managed_expected_value_non_positive")
    if suggestion.get("defined_risk") is not True or _finite(suggestion.get("max_loss")) is None:
        reasons.append("defined_risk_not_proven")
    if reasons:
        return "hold", ";".join(reasons)
    return "candidate", "scan_score_edge_and_defined_risk_passed"


def _capacity_reservation(
    conn: Connection,
    *,
    signal_id: str,
    policy_version: str,
    max_active: int,
) -> tuple[bool, Any | None]:
    """Reserve one row while preferring independent signals over structures.

    Alternative structures may use otherwise idle capacity, but a newly seen
    signal can reclaim the newest surplus row from a signal that already has
    another active representative.  The oldest active row for every signal is
    retained, so capacity is deterministic and independent-signal coverage is
    maximal up to ``max_active``.
    """
    rows = conn.execute(
        select(
            managed_opportunities.c.id,
            managed_opportunities.c.signal_id,
            managed_opportunities.c.status,
        )
        .where(managed_opportunities.c.status.in_(_ACTIVE_STATUSES))
        .where(managed_opportunities.c.policy_version == policy_version)
        .order_by(managed_opportunities.c.id)
    ).fetchall()
    if len(rows) < max_active:
        return True, None

    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[str(row.signal_id)].append(row)
    if signal_id in grouped:
        # The signal is already represented. At capacity, another structure
        # must not displace an independent signal.
        return False, None

    surplus: list[Any] = []
    for group in grouped.values():
        # Retain the first/oldest representative; later rows are alternatives.
        surplus.extend(group[1:])
    if not surplus:
        return False, None
    victim = min(
        surplus,
        key=lambda row: (
            0 if row.status == "pending_entry" else 1,
            -int(row.id),
        ),
    )
    return True, victim


def _terminalize_capacity_victim(
    conn: Connection,
    victim: Any,
    *,
    now: datetime,
) -> None:
    """Truthfully end a surplus observational path reclaimed for a signal."""
    if victim.status == "pending_entry":
        values: dict[str, Any] = {
            "status": "unobservable",
            "training_eligible": 0,
            "resolution_reason": _CAPACITY_REALLOCATION_REASON,
        }
    else:
        values = {
            "status": "censored",
            "outcome": "censored",
            "resolved_at": now,
            "training_eligible": 0,
            "resolution_reason": _CAPACITY_REALLOCATION_REASON,
        }
    result = conn.execute(
        update(managed_opportunities)
        .where(managed_opportunities.c.id == victim.id)
        .where(managed_opportunities.c.status == victim.status)
        .values(**values)
    )
    if result.rowcount != 1:
        raise RuntimeError("managed capture capacity reservation lost its victim")


def register_snapshot_opportunities(
    engine: Engine,
    settings: Settings,
    snapshot_id: int,
    *,
    now: datetime | None = None,
    decision_batch_id: str | None = None,
) -> int:
    """Register every confirmed managed candidate on ``snapshot_id``.

    The call is DB-only and belongs directly after ``scan_symbol`` persistence,
    before ranking, account affordability, alert dedup, Hermes, or execution.
    A uniqueness key keeps repeated scans from changing the first structure.
    """
    if not settings.validation.managed_capture_enabled:
        return 0
    policy_version = _managed_policy_version(settings)
    created_at = _aware(now or datetime.now(UTC)).astimezone(UTC)
    batch_id = decision_batch_id or f"snapshot:{snapshot_id}"
    if not batch_id.strip():
        raise ValueError("managed decision batch identity cannot be empty")
    with engine.connect() as conn:
        snapshot = conn.execute(
            select(
                snapshots.c.symbol,
                snapshots.c.ts,
                snapshots.c.raw_json,
                snapshots.c.spot,
                snapshots.c.iv_rank,
                snapshots.c.hv20,
                snapshots.c.iv_hv_ratio,
                snapshots.c.expected_move,
                snapshots.c.regime_dir,
                snapshots.c.regime_iv,
            ).where(snapshots.c.id == snapshot_id)
        ).one_or_none()
        score_rows = conn.execute(
            select(
                strategy_scores.c.id,
                strategy_scores.c.strategy,
                strategy_scores.c.score,
                strategy_scores.c.rationale,
                strategy_scores.c.legs_json,
                strategy_scores.c.suggestion_json,
            )
            .where(strategy_scores.c.snapshot_id == snapshot_id)
            .order_by(strategy_scores.c.id)
        ).fetchall()
    if snapshot is None or not isinstance(snapshot.raw_json, Mapping):
        return 0
    snapshot_raw = dict(snapshot.raw_json)
    detected_at = _aware(snapshot.ts).astimezone(UTC)
    session_close = nyse_session_close_utc(detected_at)
    if session_close is None:
        return 0
    entry_cutoff = session_close - timedelta(
        minutes=settings.execution.zero_dte_entry_cutoff_minutes
    )
    force_exit_at = session_close - timedelta(
        minutes=settings.execution.zero_dte_force_exit_minutes
    )
    inserted = 0
    snapshot_symbol = str(snapshot.symbol).strip().upper()
    for row in score_rows:
        raw_suggestion = row.suggestion_json
        suggestion = dict(raw_suggestion) if isinstance(raw_suggestion, Mapping) else {}
        plan = _managed_plan_for_row(suggestion)
        if plan is None:
            continue
        plan_status = plan.get("status")
        source = plan.get("source")
        signal_id = plan.get("signal_id")
        session = plan.get("session")
        direction = plan.get("direction")
        generator = plan.get("generator")
        setup_type = plan.get("setup_type")
        stop_pct = _finite(plan.get("stop_pct"))
        target_r = _finite(plan.get("target_r"))
        target_pct = _finite(plan.get("target_pct"))
        strategy = str(row.strategy)
        decision_score = _finite(row.score)
        decision_defined_risk = suggestion.get("defined_risk") is True
        decision_max_loss = _finite(suggestion.get("max_loss"))
        if decision_score is None or not 0.0 <= decision_score <= 100.0:
            continue
        if decision_max_loss is not None and decision_max_loss <= 0.0:
            decision_max_loss = None
        is_independent_shadow = generator != _OPENING_RANGE_GENERATOR
        admission_eligible, shadow_only = _immutable_admission_classification(
            plan=plan,
            suggestion=suggestion,
            strategy=strategy,
        )
        # Only daemon-authored row plans can establish label identity.  Unknown
        # strategies/generators are discarded rather than admitted as a new,
        # unreviewed production family.
        if (
            plan.get("schema_version") != _MANAGED_PLAN_SCHEMA_VERSION
            or source != "trusted_daemon"
            or plan_status not in {"entry_confirmed", "shadow_confirmed"}
            or not isinstance(signal_id, str)
            or not signal_id
            or not isinstance(session, str)
            or len(session) != 10
            or direction not in {"bull", "bear"}
            or not isinstance(generator, str)
            or not generator
            or (
                generator != _OPENING_RANGE_GENERATOR
                and generator
                not in {
                    "opening_momentum_continuation",
                    "failed_breakout_reversal",
                    "late_session_momentum",
                }
            )
            or not (strategy in _PRIMARY_STRATEGIES or strategy.startswith(_SHADOW_GRID_PREFIX))
            or not isinstance(setup_type, str)
            or not setup_type
            or stop_pct is None
            or not 0 < stop_pct < 1
            or target_pct is None
            or target_pct <= 0
        ):
            continue
        if is_independent_shadow and (
            plan_status != "shadow_confirmed"
            or plan.get("admission_enabled") is not False
            or setup_type != generator
            or target_r is None
            or not math.isclose(
                stop_pct,
                settings.execution.opening_range_stop_pct,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                target_r,
                settings.execution.opening_range_target_r_min,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                target_pct,
                settings.execution.opening_range_stop_pct
                * settings.execution.opening_range_target_r_min,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            continue
        compact_session = session.replace("-", "")
        option_expiry = plan.get("option_expiry", compact_session)
        if not isinstance(option_expiry, str):
            continue
        thesis_expires_at_raw = plan.get("thesis_expires_at")
        thesis_expires_at = (
            _timestamp(thesis_expires_at_raw)
            if thesis_expires_at_raw is not None
            else force_exit_at
        )
        if thesis_expires_at is None:
            continue
        timeout_at = min(force_exit_at, thesis_expires_at)
        key = opportunity_key(signal_id, strategy, policy_version=policy_version)
        legs = canonical_legs(row.legs_json)
        structure_hash = _json_hash(legs)
        baseline_action, baseline_reason = _baseline(float(row.score), suggestion, settings)
        bot_action = None if admission_eligible else "hold"
        bot_reason = None if admission_eligible else "immutable_research_only_opportunity"
        bot_decided_at = None if admission_eligible else created_at
        reasons: list[str] = []
        if not legs:
            reasons.append("invalid_or_non_option_structure")
        elif any(leg["symbol"] != snapshot_symbol for leg in legs):
            reasons.append("structure_symbol_mismatch")
        elif any(leg["expiry"] != compact_session for leg in legs):
            reasons.append("structure_is_not_exact_0dte")
        if option_expiry != compact_session:
            reasons.append("managed_plan_expiry_is_not_exact_0dte")
        if session != detected_at.date().isoformat():
            reasons.append("managed_plan_session_mismatch")
        if detected_at >= session_close:
            reasons.append("detected_after_session_close")
        if detected_at >= timeout_at:
            reasons.append("thesis_expired_before_detection")
        if detected_at >= entry_cutoff:
            baseline_action = "hold"
            baseline_reason = baseline_reason + ";entry_cutoff_passed"
        option_contracts = sum(int(leg["quantity"]) for leg in legs)
        persisted_commission = _finite(suggestion.get("managed_commission_estimate"))
        commission_estimate = (
            persisted_commission
            if persisted_commission is not None and persisted_commission >= 0.0
            else option_contracts * 2.0 * settings.execution.opening_range_commission_per_contract
        )
        features = build_capture_feature_payload(
            feature_schema_version=settings.managed_learning.feature_schema_version,
            snapshot_id=snapshot_id,
            spot=snapshot.spot,
            iv_rank=snapshot.iv_rank,
            hv20=snapshot.hv20,
            iv_hv_ratio=snapshot.iv_hv_ratio,
            expected_move=snapshot.expected_move,
            regime_dir=snapshot.regime_dir,
            regime_iv=snapshot.regime_iv,
            raw_snapshot=snapshot_raw,
            score=float(row.score),
            suggestion=suggestion,
            registered_before_entry_cutoff=detected_at < entry_cutoff,
        )
        values: dict[str, Any] = dict(
            opportunity_key=key,
            signal_id=signal_id,
            session=session,
            symbol=snapshot_symbol,
            direction=direction,
            setup_type=setup_type,
            strategy=strategy,
            strategy_score_id=int(row.id),
            structure_hash=structure_hash,
            legs_json=legs,
            features_json=features,
            policy_version=policy_version,
            created_at=created_at,
            detected_at=detected_at,
            decision_batch_id=batch_id,
            decision_score=decision_score,
            decision_defined_risk=int(decision_defined_risk),
            decision_max_loss=decision_max_loss,
            decision_account_value_available=(0 if bot_decided_at is not None else None),
            decision_account_value_usd=None,
            baseline_action=baseline_action,
            baseline_reason=baseline_reason,
            admission_eligible=int(admission_eligible),
            shadow_only=int(shadow_only),
            bot_action=bot_action,
            bot_reason=bot_reason,
            bot_decided_at=bot_decided_at,
            session_close_at=session_close,
            entry_cutoff_at=entry_cutoff,
            timeout_at=timeout_at,
            stop_pct=stop_pct,
            target_pct=target_pct,
            commission_estimate=commission_estimate,
            training_eligible=0,
        )
        with engine.begin() as conn:
            if (
                conn.execute(
                    select(managed_opportunities.c.id).where(
                        managed_opportunities.c.opportunity_key == key
                    )
                ).first()
                is not None
            ):
                continue
            victim: Any | None = None
            if not reasons:
                admitted, victim = _capacity_reservation(
                    conn,
                    signal_id=signal_id,
                    policy_version=policy_version,
                    max_active=settings.validation.managed_capture_max_active,
                )
                if not admitted:
                    reasons.append("managed_capture_capacity_reached")
            status = "unobservable" if reasons else "pending_entry"
            statement = (
                sqlite_insert(managed_opportunities)
                .values(
                    **values,
                    status=status,
                    resolution_reason=";".join(reasons) if reasons else None,
                )
                .on_conflict_do_nothing(index_elements=["opportunity_key"])
            )
            result = conn.execute(statement)
            if result.rowcount and victim is not None:
                _terminalize_capacity_victim(conn, victim, now=created_at)
        inserted += int(result.rowcount or 0)
    return inserted


def record_snapshot_bot_dispositions(
    engine: Engine,
    snapshot_id: int,
    dispositions: Mapping[str, tuple[str, str]],
    *,
    policy_version: str,
    decided_at: datetime | None = None,
    account_value_usd: float | None = None,
) -> int:
    """Freeze OptionsBot's deterministic scan-admission result before Hermes.

    This is intentionally the *scan admission* policy (score, managed EV,
    defined risk, and live-equity affordability), not a claim that every later
    execution/liquidity gate passed or an order filled.  The first complete
    disposition wins; repeated scans cannot rewrite it.
    """
    if not dispositions:
        return 0
    at = _aware(decided_at or datetime.now(UTC)).astimezone(UTC)
    account_value = _finite(account_value_usd)
    if account_value_usd is not None and account_value is None:
        raise ValueError("managed decision account value must be finite when available")
    if account_value is None and any(
        action == "candidate" for action, _reason in dispositions.values()
    ):
        raise ValueError("managed candidate disposition requires decision-time account value")
    recorded = 0
    for strategy, (action, reason) in dispositions.items():
        if action not in {"candidate", "hold"} or not reason:
            continue
        with engine.begin() as conn:
            score_id = conn.execute(
                select(strategy_scores.c.id)
                .where(strategy_scores.c.snapshot_id == snapshot_id)
                .where(strategy_scores.c.strategy == strategy)
            ).scalar_one_or_none()
            if score_id is None:
                continue
            statement = (
                update(managed_opportunities)
                .where(managed_opportunities.c.strategy_score_id == int(score_id))
                .where(managed_opportunities.c.policy_version == policy_version)
                .where(managed_opportunities.c.admission_eligible == 1)
                .where(managed_opportunities.c.shadow_only == 0)
                .where(managed_opportunities.c.bot_action.is_(None))
                .where(managed_opportunities.c.bot_reason.is_(None))
                .where(managed_opportunities.c.bot_decided_at.is_(None))
                .where(managed_opportunities.c.detected_at <= at)
                .values(
                    bot_action=action,
                    bot_reason=reason,
                    bot_decided_at=at,
                    decision_account_value_available=int(account_value_usd is not None),
                    decision_account_value_usd=account_value,
                )
            )
            if action == "candidate":
                statement = statement.where(managed_opportunities.c.entry_cutoff_at > at)
            result = conn.execute(statement)
        recorded += int(result.rowcount or 0)
    return recorded


def _leg_specs(legs: Sequence[Mapping[str, Any]]) -> list[LegSpec]:
    return [
        (
            str(leg["symbol"]),
            str(leg["expiry"]),
            float(leg["strike"]),
            str(leg["right"]),
        )
        for leg in legs
    ]


def _compact_quote(quote: OptionQuote) -> dict[str, Any]:
    return {
        "symbol": quote.symbol,
        "expiry": quote.expiry,
        "strike": quote.strike,
        "right": quote.right,
        "bid": quote.bid,
        "ask": quote.ask,
        "iv": quote.iv,
        "delta": quote.delta,
        "gamma": quote.gamma,
        "theta": quote.theta,
        "vega": quote.vega,
        "open_interest": quote.open_interest,
        "volume": quote.volume,
        "ts": quote.ts.isoformat() if quote.ts is not None else None,
        "delayed": quote.delayed,
    }


def _usable_combo(
    legs: list[dict[str, Any]],
    quote_cache: Mapping[LegSpec, OptionQuote],
    *,
    now: datetime,
    settings: Settings,
) -> tuple[
    tuple[float, float] | None,
    datetime | None,
    datetime | None,
    list[dict[str, Any]],
    str | None,
]:
    local: dict[tuple[str, float, str], OptionQuote] = {}
    quotes: list[OptionQuote] = []
    for spec in _leg_specs(legs):
        quote = quote_cache.get(spec)
        if quote is None:
            return None, None, None, [], f"missing_quote:{spec}"
        quotes.append(quote)
        local[(spec[1], spec[2], spec[3])] = quote
    compact = [_compact_quote(quote) for quote in quotes]
    if any(quote.delayed is not False for quote in quotes):
        return None, None, None, compact, "delayed_or_unknown_quote"
    timestamps: list[datetime] = []
    for quote in quotes:
        if quote.bid is None or quote.ask is None:
            return None, None, None, compact, "partial_nbbo"
        if (
            not math.isfinite(float(quote.bid))
            or not math.isfinite(float(quote.ask))
            or float(quote.bid) < 0
            or float(quote.ask) < float(quote.bid)
        ):
            return None, None, None, compact, "invalid_nbbo"
        if quote.ts is None:
            return None, None, None, compact, "missing_quote_timestamp"
        quote_ts = _aware(quote.ts).astimezone(UTC)
        age = (now - quote_ts).total_seconds()
        if age < 0 or age > settings.validation.managed_capture_quote_max_age_seconds:
            return None, None, None, compact, f"quote_age_outside_window:{age:.3f}s"
        timestamps.append(quote_ts)
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    span = (max_ts - min_ts).total_seconds()
    if span > settings.validation.managed_capture_quote_span_seconds:
        return None, min_ts, max_ts, compact, f"asynchronous_leg_quotes:{span:.3f}s"
    combo = combo_bid_ask(legs, local)
    if combo is None or not all(math.isfinite(value) for value in combo):
        return None, min_ts, max_ts, compact, "combo_nbbo_unavailable"
    if combo[1] < combo[0]:
        return None, min_ts, max_ts, compact, "crossed_combo_nbbo"
    return combo, min_ts, max_ts, compact, None


def _poll_bucket(now: datetime, interval_seconds: int) -> int:
    return int(now.timestamp()) // interval_seconds


def _signal_round_robin_rows(rows: Sequence[Any], *, bucket: int) -> list[Any]:
    """Interleave structures by signal and rotate which signal starts a poll."""
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        signal_id = str(row.signal_id) if row.signal_id else f"opportunity:{row.id}"
        grouped.setdefault(signal_id, []).append(row)
    signals = list(grouped)
    if not signals:
        return []
    offset = bucket % len(signals)
    signals = signals[offset:] + signals[:offset]
    ordered: list[Any] = []
    for depth in range(max(len(grouped[signal]) for signal in signals)):
        ordered.extend(grouped[signal][depth] for signal in signals if depth < len(grouped[signal]))
    return ordered


def _requested_specs_for_rows(
    rows: Sequence[Any],
    *,
    bucket: int,
    line_budget: int,
) -> list[LegSpec]:
    """Spend the line budget on complete opportunity bundles, signal-first."""
    ordered_specs: list[LegSpec] = []
    seen_specs: set[LegSpec] = set()
    for row in _signal_round_robin_rows(rows, bucket=bucket):
        row_specs = list(dict.fromkeys(_leg_specs(list(row.legs_json or []))))
        new_specs = [spec for spec in row_specs if spec not in seen_specs]
        if len(ordered_specs) + len(new_specs) > line_budget:
            # A partial combo can never produce an executable mark. Skip it
            # and let a later smaller/shared bundle use the remaining lines.
            continue
        ordered_specs.extend(new_specs)
        seen_specs.update(new_specs)
    return ordered_specs


async def _fetch_quotes_bounded(
    market_data: MarketDataClient,
    specs: Sequence[LegSpec],
    *,
    deadline: float,
    line_budget: int,
) -> tuple[dict[LegSpec, OptionQuote], int]:
    """Fetch live snapshots without letting one contract consume the tick."""
    if not specs:
        return {}, 0
    loop = asyncio.get_running_loop()
    concurrency = min(_MAX_CONCURRENT_QUOTE_REQUESTS, line_budget, len(specs))
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(spec: LegSpec) -> tuple[LegSpec, OptionQuote | None, bool]:
        async with semaphore:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return spec, None, True
            try:
                quote = await asyncio.wait_for(
                    market_data.get_option_snapshot(
                        spec[0],
                        spec[1],
                        spec[2],
                        spec[3],  # type: ignore[arg-type]
                    ),
                    timeout=min(_PER_QUOTE_TIMEOUT_SECONDS, remaining),
                )
            except Exception:  # noqa: BLE001 -- shadow quote failures are observations
                log.debug("managed capture quote failed for %s", spec, exc_info=True)
                return spec, None, True
            return spec, quote, False

    results = await asyncio.gather(*(_one(spec) for spec in specs))
    quotes = {spec: quote for spec, quote, _failed in results if quote is not None}
    return quotes, sum(1 for _spec, _quote, failed in results if failed)


def _insert_mark(
    conn: Connection,
    *,
    opportunity_id: int,
    bucket: int,
    now: datetime,
    min_ts: datetime | None,
    max_ts: datetime | None,
    combo: tuple[float, float] | None,
    liquidation_net: float | None,
    gross_pnl: float | None,
    net_pnl: float | None,
    usable: bool,
    issue: str | None,
    legs_json: list[dict[str, Any]],
) -> bool:
    statement = (
        sqlite_insert(managed_opportunity_marks)
        .values(
            opportunity_id=opportunity_id,
            poll_bucket=bucket,
            observed_at=now,
            leg_quote_min_ts=min_ts,
            leg_quote_max_ts=max_ts,
            combo_bid=combo[0] if combo is not None else None,
            combo_ask=combo[1] if combo is not None else None,
            combo_mid=(combo[0] + combo[1]) / 2.0 if combo is not None else None,
            liquidation_net=liquidation_net,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            usable=1 if usable else 0,
            issue=issue,
            legs_json=legs_json,
        )
        .on_conflict_do_nothing(index_elements=["opportunity_id", "poll_bucket"])
    )
    result = conn.execute(statement)
    return bool(result.rowcount)


def _persist_mark(
    engine: Engine,
    *,
    opportunity_id: int,
    bucket: int,
    now: datetime,
    min_ts: datetime | None,
    max_ts: datetime | None,
    combo: tuple[float, float] | None,
    liquidation_net: float | None,
    gross_pnl: float | None,
    net_pnl: float | None,
    usable: bool,
    issue: str | None,
    legs_json: list[dict[str, Any]],
) -> bool:
    """Persist an unusable mark only while its opportunity is still active.

    Quote collection operates on a bounded snapshot of active opportunities.
    Another reducer may terminalize one of those rows before an unusable quote
    reaches this boundary.  Make the parent-state predicate part of the INSERT
    itself so that stale work cannot append an immutable post-terminal mark.
    """
    with engine.begin() as conn:
        columns = (
            managed_opportunity_marks.c.opportunity_id,
            managed_opportunity_marks.c.poll_bucket,
            managed_opportunity_marks.c.observed_at,
            managed_opportunity_marks.c.leg_quote_min_ts,
            managed_opportunity_marks.c.leg_quote_max_ts,
            managed_opportunity_marks.c.combo_bid,
            managed_opportunity_marks.c.combo_ask,
            managed_opportunity_marks.c.combo_mid,
            managed_opportunity_marks.c.liquidation_net,
            managed_opportunity_marks.c.gross_pnl,
            managed_opportunity_marks.c.net_pnl,
            managed_opportunity_marks.c.usable,
            managed_opportunity_marks.c.issue,
            managed_opportunity_marks.c.legs_json,
        )
        values = (
            opportunity_id,
            bucket,
            now,
            min_ts,
            max_ts,
            combo[0] if combo is not None else None,
            combo[1] if combo is not None else None,
            (combo[0] + combo[1]) / 2.0 if combo is not None else None,
            liquidation_net,
            gross_pnl,
            net_pnl,
            1 if usable else 0,
            issue,
            legs_json,
        )
        source = select(
            *(
                literal(value, type_=column.type)
                for column, value in zip(columns, values, strict=True)
            )
        ).where(
            managed_opportunities.c.id == opportunity_id,
            managed_opportunities.c.status.in_(_ACTIVE_STATUSES),
        )
        statement = (
            sqlite_insert(managed_opportunity_marks)
            .from_select([column.name for column in columns], source)
            .on_conflict_do_nothing(index_elements=["opportunity_id", "poll_bucket"])
        )
        return bool(conn.execute(statement).rowcount)


def _update_reduced_opportunity(
    conn: Connection,
    row: Any,
    values: Mapping[str, Any],
) -> None:
    """CAS one lifecycle generation or abort the mark transaction.

    ``valid_marks`` advances for every usable observation once the row is
    active, so it is also a compact reducer generation.  Matching it together
    with the lifecycle state prevents two stale ticks from overwriting extrema
    or terminal ordering.  Raising is intentional: the caller's transaction
    must roll the immutable mark back when its paired reduction loses a race.
    """
    statement = (
        update(managed_opportunities)
        .where(managed_opportunities.c.id == int(row.id))
        .where(managed_opportunities.c.status == str(row.status))
        .where(managed_opportunities.c.valid_marks == int(row.valid_marks))
    )
    if row.last_valid_mark_at is None:
        statement = statement.where(managed_opportunities.c.last_valid_mark_at.is_(None))
    else:
        statement = statement.where(
            managed_opportunities.c.last_valid_mark_at == row.last_valid_mark_at
        )
    result = conn.execute(statement.values(**values))
    if result.rowcount != 1:
        raise RuntimeError("managed opportunity reduction lost a concurrent race")


def _process_usable_mark(
    engine: Engine,
    row: Any,
    *,
    now: datetime,
    bucket: int,
    combo: tuple[float, float],
    min_ts: datetime,
    max_ts: datetime,
    legs_json: list[dict[str, Any]],
    settings: Settings,
) -> tuple[str | None, bool]:
    """Persist a usable mark and its deterministic reduction atomically."""
    with engine.begin() as conn:
        current = conn.execute(
            select(managed_opportunities)
            .where(managed_opportunities.c.id == int(row.id))
            .with_for_update()
        ).one_or_none()
        if current is None or current.status not in _ACTIVE_STATUSES:
            return None, False

        entry_net = _finite(current.entry_net)
        if current.status == "pending_entry":
            if entry_net is None:
                entry_net = combo[0]  # marketable open: buy asks / sell bids
        elif (
            entry_net is None
            or _finite(current.basis_dollars) is None
            or float(current.basis_dollars) <= 0.0
            or not isinstance(current.entry_ts, datetime)
            or not isinstance(current.last_valid_mark_at, datetime)
            or int(current.valid_marks) < 1
        ):
            raise RuntimeError("active managed opportunity has an incoherent entry state")
        if entry_net is None:  # defensive: statuses are closed above
            raise RuntimeError("managed opportunity has no executable entry basis")

        liquidation_net = combo[1]  # marketable close side in signed-net space
        gross_pnl = (entry_net - liquidation_net) * 100.0
        commission = float(current.commission_estimate)
        net_pnl = gross_pnl - commission

        outcome: str | None = None
        values: dict[str, Any]
        if current.status == "pending_entry":
            basis = abs(entry_net) * 100.0
            if basis <= 0 or not math.isfinite(basis):
                outcome = "unobservable"
                values = {
                    "status": "unobservable",
                    "resolution_reason": "non_positive_marketable_entry_basis",
                }
            else:
                max_profit = structural_max_profit_dollars(
                    list(current.legs_json or []), entry_net_per_share=entry_net
                )
                target_dollars = basis * float(current.target_pct)
                values = {
                    "entry_ts": now,
                    "entry_combo_bid": combo[0],
                    "entry_combo_ask": combo[1],
                    "entry_net": entry_net,
                    "basis_dollars": basis,
                    "last_valid_mark_at": now,
                    "valid_marks": 1,
                    "mfe_dollars": gross_pnl,
                    "mae_dollars": gross_pnl,
                }
                if max_profit is not None and target_dollars + commission > max_profit:
                    outcome = "unobservable"
                    values.update(
                        status="unobservable",
                        resolution_reason="target_not_reachable_after_commissions",
                    )
                else:
                    values["status"] = "active"
        else:
            active_basis = float(current.basis_dollars)
            previous_ts = _aware(current.last_valid_mark_at).astimezone(UTC)
            gap = (now - previous_ts).total_seconds()
            prior_max_gap = _finite(current.max_mark_gap_seconds) or 0.0
            max_gap = max(prior_max_gap, gap)
            prior_mfe = _finite(current.mfe_dollars)
            prior_mae = _finite(current.mae_dollars)
            mfe = max(gross_pnl, prior_mfe if prior_mfe is not None else gross_pnl)
            mae = min(gross_pnl, prior_mae if prior_mae is not None else gross_pnl)
            too_large_gap = max_gap > settings.validation.managed_capture_max_mark_gap_seconds
            reason: str | None = None
            if now >= _aware(current.timeout_at).astimezone(UTC):
                if too_large_gap:
                    outcome = "censored"
                    reason = "observation_gap_before_timeout"
                else:
                    outcome = "timeout"
                    reason = "scheduled_zero_dte_force_exit"
            elif gross_pnl >= active_basis * float(current.target_pct):
                if too_large_gap:
                    outcome = "censored"
                    reason = "ambiguous_gap_before_target_observation"
                else:
                    outcome = "target"
                    reason = "first_observed_target_boundary"
            elif gross_pnl <= -active_basis * float(current.stop_pct):
                if too_large_gap:
                    outcome = "censored"
                    reason = "ambiguous_gap_before_stop_observation"
                else:
                    outcome = "stop"
                    reason = "first_observed_stop_boundary"
            values = {
                "last_valid_mark_at": now,
                "valid_marks": int(current.valid_marks) + 1,
                "max_mark_gap_seconds": max_gap,
                "mfe_dollars": mfe,
                "mae_dollars": mae,
            }
            if outcome is not None:
                admission_ts = _aware(current.entry_ts).astimezone(UTC)
                entry_cutoff = _aware(current.entry_cutoff_at).astimezone(UTC)
                decision_ts = (
                    _aware(current.bot_decided_at).astimezone(UTC)
                    if isinstance(current.bot_decided_at, datetime)
                    else None
                )
                eligible = (
                    outcome != "censored"
                    and decision_ts is not None
                    and decision_ts <= admission_ts < entry_cutoff
                )
                values.update(
                    status="censored" if outcome == "censored" else "resolved",
                    outcome=outcome,
                    resolved_at=now,
                    exit_net=liquidation_net,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    training_eligible=1 if eligible else 0,
                    resolution_reason=reason,
                )

        inserted = _insert_mark(
            conn,
            opportunity_id=int(current.id),
            bucket=bucket,
            now=now,
            min_ts=min_ts,
            max_ts=max_ts,
            combo=combo,
            liquidation_net=liquidation_net,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            usable=True,
            issue=None,
            legs_json=legs_json,
        )
        if not inserted:
            return None, False
        _update_reduced_opportunity(conn, current, values)
        # The entry quote establishes the executable basis. It is not also a
        # threshold decision; the first subsequent policy tick performs that.
        return outcome, True


def _finalize_after_close(
    engine: Engine,
    now: datetime,
    *,
    policy_version: str,
) -> tuple[int, int]:
    """Terminalize rows that can no longer obtain a same-session quote."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                managed_opportunities.c.id,
                managed_opportunities.c.status,
            )
            .where(managed_opportunities.c.status.in_(_ACTIVE_STATUSES))
            .where(managed_opportunities.c.policy_version == policy_version)
            .where(managed_opportunities.c.session_close_at < now)
        ).fetchall()
    censored = 0
    unobservable = 0
    for row in rows:
        if row.status == "pending_entry":
            values: dict[str, Any] = {
                "status": "unobservable",
                "resolution_reason": "no_usable_entry_quote_before_session_close",
            }
        else:
            values = {
                "status": "censored",
                "outcome": "censored",
                "resolved_at": now,
                "training_eligible": 0,
                "resolution_reason": "no_usable_timeout_quote_before_session_close",
            }
        if not _terminalize_after_close_row(engine, row, values):
            continue
        if row.status == "pending_entry":
            unobservable += 1
        else:
            censored += 1
    return censored, unobservable


def _terminalize_after_close_row(
    engine: Engine,
    row: Any,
    values: Mapping[str, Any],
) -> bool:
    """CAS one close-time terminal transition and report whether it won."""
    with engine.begin() as conn:
        result = conn.execute(
            update(managed_opportunities)
            .where(managed_opportunities.c.id == row.id)
            .where(managed_opportunities.c.status == row.status)
            .values(**values)
        )
    return result.rowcount == 1


async def run_managed_capture_tick(
    context: DaemonContext,
    *,
    now: datetime | None = None,
    md: MarketDataClient | None = None,
) -> ManagedCaptureSummary:
    """Capture one bounded shadow mark without delaying protective exits."""
    settings = context.settings
    if not settings.validation.managed_capture_enabled:
        return ManagedCaptureSummary(0, 0, 0, 0, 0, False, 0)
    policy_version = _managed_policy_version(settings)
    fixed_test_clock = now is not None
    observed_at = _aware(now or datetime.now(UTC)).astimezone(UTC)
    pre_censored, _pre_unobservable = _finalize_after_close(
        context.engine,
        observed_at,
        policy_version=policy_version,
    )
    with context.engine.connect() as conn:
        rows = conn.execute(
            select(managed_opportunities)
            .where(managed_opportunities.c.status.in_(_ACTIVE_STATUSES))
            .where(managed_opportunities.c.policy_version == policy_version)
            .where(managed_opportunities.c.session_close_at >= observed_at)
            .order_by(managed_opportunities.c.created_at, managed_opportunities.c.id)
            .limit(settings.validation.managed_capture_max_active)
        ).fetchall()
    if not rows:
        return ManagedCaptureSummary(0, 0, 0, 0, pre_censored, False, 0)

    bucket = _poll_bucket(
        observed_at,
        settings.validation.managed_capture_interval_seconds,
    )
    line_budget = min(
        settings.validation.managed_capture_max_unique_legs,
        settings.ibkr.max_market_data_lines,
    )
    requested_specs = _requested_specs_for_rows(
        rows,
        bucket=bucket,
        line_budget=line_budget,
    )
    market_data = md or MarketDataClient(context.ibkr, resolver=context.resolver)
    try:
        await asyncio.wait_for(
            context.ibkr_lock.acquire(),
            timeout=settings.validation.managed_capture_lock_timeout_seconds,
        )
    except TimeoutError:
        unusable = 0
        for row in rows:
            if _persist_mark(
                context.engine,
                opportunity_id=int(row.id),
                bucket=bucket,
                now=observed_at,
                min_ts=None,
                max_ts=None,
                combo=None,
                liquidation_net=None,
                gross_pnl=None,
                net_pnl=None,
                usable=False,
                issue="trading_market_data_lock_busy",
                legs_json=[],
            ):
                unusable += 1
        return ManagedCaptureSummary(len(rows), 0, unusable, 0, pre_censored, True, 0)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.validation.managed_capture_tick_timeout_seconds
    try:
        quote_cache, quote_errors = await _fetch_quotes_bounded(
            market_data,
            requested_specs,
            deadline=deadline,
            line_budget=line_budget,
        )
    finally:
        context.ibkr_lock.release()

    # IBKR timestamps a snapshot when the quote arrives, not when this job was
    # scheduled.  Use completion time for real freshness/threshold decisions;
    # otherwise a perfectly current ticker can appear a few seconds "future"
    # relative to ``observed_at`` and be discarded.  An explicit ``now`` keeps
    # deterministic tests/replays on their supplied clock.
    mark_at = observed_at if fixed_test_clock else datetime.now(UTC)
    usable = 0
    unusable = 0
    resolved = 0
    censored = pre_censored
    for row in rows:
        if row.status == "pending_entry" and row.bot_decided_at is None:
            if _persist_mark(
                context.engine,
                opportunity_id=int(row.id),
                bucket=bucket,
                now=mark_at,
                min_ts=None,
                max_ts=None,
                combo=None,
                liquidation_net=None,
                gross_pnl=None,
                net_pnl=None,
                usable=False,
                issue="scan_disposition_pending",
                legs_json=[],
            ):
                unusable += 1
            continue
        combo, min_ts, max_ts, leg_quotes, issue = _usable_combo(
            list(row.legs_json or []),
            quote_cache,
            now=mark_at,
            settings=settings,
        )
        if issue is not None or combo is None or min_ts is None or max_ts is None:
            if _persist_mark(
                context.engine,
                opportunity_id=int(row.id),
                bucket=bucket,
                now=mark_at,
                min_ts=min_ts,
                max_ts=max_ts,
                combo=None,
                liquidation_net=None,
                gross_pnl=None,
                net_pnl=None,
                usable=False,
                issue=issue or "quote_budget_or_tick_deadline_exhausted",
                legs_json=leg_quotes,
            ):
                unusable += 1
            continue
        outcome, inserted = _process_usable_mark(
            context.engine,
            row,
            now=mark_at,
            bucket=bucket,
            combo=combo,
            min_ts=min_ts,
            max_ts=max_ts,
            legs_json=leg_quotes,
            settings=settings,
        )
        if inserted:
            usable += 1
        if outcome == "censored":
            censored += 1
        elif outcome is not None:
            resolved += 1
    return ManagedCaptureSummary(
        opportunities_seen=len(rows),
        usable_marks=usable,
        unusable_marks=unusable,
        resolved=resolved,
        censored=censored,
        skipped_for_trading=False,
        quote_errors=quote_errors,
    )
