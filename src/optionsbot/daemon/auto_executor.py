"""Full-auto entry hook (IBK-130): alerted candidates → execute_pick.

Runs the SAME pipeline as Telegram /execute — every gate (freshness, caps,
liquidity, margin, dedup, plus the auto-only earnings and buying-power
gates) applies per pick. Confirm mode never enters here.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import exists, or_, select, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.risk_structure import has_structurally_defined_option_risk
from optionsbot.execution.state import trip_kill
from optionsbot.hermes_overlay import hold_pending_reviews, load_overlay_state
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.review_evidence import snapshot_ready_for_auto
from optionsbot.scoring import ScoredStrategy
from optionsbot.storage.schema import (
    alerts,
    entry_intent_consumptions,
    entry_reviews,
    snapshots,
    strategy_scores,
)

log = logging.getLogger(__name__)

_REQUIRED_ENTRY_CHECKS = {
    "bot_health",
    "candidate",
    "microstructure",
    "greeks",
    "regime_history",
    "catalysts",
    "account_risk",
}


def _score_id_for(context: DaemonContext, snapshot_id: int, strategy: str) -> int | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(strategy_scores.c.id)
            .where(strategy_scores.c.snapshot_id == snapshot_id)
            .where(strategy_scores.c.strategy == strategy)
        ).one_or_none()
    return int(row.id) if row is not None else None


def _requested_review_id_for(context: DaemonContext, score_id: int) -> int | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(entry_reviews.c.id)
            .where(entry_reviews.c.strategy_score_id == score_id)
            .where(entry_reviews.c.verdict == "vetted_paper_candidate")
            .where(entry_reviews.c.status == "requested")
            .limit(1)
        ).first()
    return int(row.id) if row is not None else None


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _positive_defined_risk(row: Any) -> bool:
    legs = row.legs_json
    suggestion = row.suggestion_json
    if not has_structurally_defined_option_risk(legs):
        return False
    if not isinstance(suggestion, dict) or suggestion.get("defined_risk") is not True:
        return False
    try:
        premium = float(suggestion["credit_or_debit"])
        max_loss = float(suggestion["max_loss"])
        max_profit = float(suggestion["max_profit"])
        prob_profit = float(suggestion["prob_profit"])
        expected_value = float(suggestion["expected_value"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    values = (premium, max_loss, max_profit, prob_profit, expected_value)
    return (
        all(math.isfinite(value) for value in values)
        and premium != 0
        and max_loss > 0
        and max_profit > 0
        and 0 < prob_profit < 1
        and expected_value > 0
    )


def _review_authorization_error(
    context: DaemonContext,
    review_id: int,
    score_id: int,
    *,
    now: datetime | None = None,
) -> str | None:
    """Independently validate every field that can authorize an entry."""
    check_now = now if now is not None else datetime.now(UTC)
    with context.engine.connect() as conn:
        row = conn.execute(
            select(
                entry_reviews.c.id,
                entry_reviews.c.strategy_score_id,
                entry_reviews.c.alert_id,
                entry_reviews.c.reviewed_at,
                entry_reviews.c.verdict,
                entry_reviews.c.confidence,
                entry_reviews.c.sources_json,
                entry_reviews.c.reason,
                entry_reviews.c.checks_json,
                alerts.c.strategy_score_id.label("alert_score_id"),
                alerts.c.status.label("alert_status"),
                alerts.c.ts.label("alert_ts"),
                alerts.c.sent_ts,
                alerts.c.telegram_msg_id,
                alerts.c.symbol.label("alert_symbol"),
                alerts.c.strategy.label("alert_strategy"),
                alerts.c.score.label("alert_score"),
                strategy_scores.c.strategy,
                strategy_scores.c.score,
                strategy_scores.c.legs_json,
                strategy_scores.c.suggestion_json,
                snapshots.c.symbol,
                snapshots.c.ts.label("snapshot_ts"),
                snapshots.c.raw_json,
            )
            .select_from(
                entry_reviews.join(
                    strategy_scores,
                    entry_reviews.c.strategy_score_id == strategy_scores.c.id,
                )
                .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
                .outerjoin(alerts, entry_reviews.c.alert_id == alerts.c.id)
            )
            .where(entry_reviews.c.id == review_id)
        ).first()
    if row is None:
        return "review or exact candidate is missing"
    if int(row.strategy_score_id) != score_id:
        return "review strategy-score identity changed"
    if row.alert_id is None or row.alert_score_id is None:
        return "exact alert identity is missing"
    if int(row.alert_score_id) != score_id:
        return "alert does not authorize this strategy score"
    if row.alert_status != "sent" or row.sent_ts is None or row.telegram_msg_id is None:
        return "alert delivery is not proven"
    if row.alert_symbol != row.symbol or row.alert_strategy != row.strategy:
        return "alert candidate metadata does not match persisted score"
    try:
        alert_score = float(row.alert_score)
        candidate_score = float(row.score)
    except (TypeError, ValueError, OverflowError):
        return "alert score is malformed"
    if not math.isfinite(alert_score) or alert_score != candidate_score:
        return "alert score does not match persisted candidate"
    if row.verdict != "vetted_paper_candidate":
        return "review verdict is not executable"
    try:
        confidence = float(row.confidence)
    except (TypeError, ValueError, OverflowError):
        return "review confidence is malformed"
    if not math.isfinite(confidence) or not 0.80 <= confidence <= 1.0:
        return "review confidence is below the execution threshold"
    sources = row.sources_json
    if not isinstance(sources, list):
        return "review sources are malformed"
    clean_sources = [
        source.strip() for source in sources if isinstance(source, str) and source.strip()
    ]
    if len({source.casefold() for source in clean_sources}) < 2:
        return "review lacks two distinct corroborating sources"
    checks = row.checks_json
    if not isinstance(checks, dict) or set(checks) != _REQUIRED_ENTRY_CHECKS:
        return "review check set is incomplete"
    if any(checks.get(name) is not True for name in _REQUIRED_ENTRY_CHECKS):
        return "one or more mandatory review checks failed"
    if not isinstance(row.reason, str) or not row.reason.strip():
        return "review reason is missing"
    reviewed_at = _aware(row.reviewed_at)
    alert_ts = _aware(row.alert_ts)
    sent_ts = _aware(row.sent_ts)
    snapshot_ts = _aware(row.snapshot_ts)
    if reviewed_at is None or alert_ts is None or sent_ts is None or snapshot_ts is None:
        return "review, alert delivery, or snapshot timestamp is missing"
    if (
        reviewed_at < sent_ts
        or sent_ts < alert_ts
        or reviewed_at > check_now
        or sent_ts > check_now
        or alert_ts > check_now
        or snapshot_ts > check_now
    ):
        return "review timestamp is inconsistent with proven alert delivery"
    max_age = timedelta(minutes=context.settings.execution.max_pick_age_minutes)
    age = check_now - snapshot_ts
    if age > max_age or age < timedelta(0):
        return "candidate is stale or timestamped in the future"
    raw = row.raw_json
    if not isinstance(raw, dict):
        return "candidate readiness evidence is missing"
    if not snapshot_ready_for_auto(raw):
        return "candidate market data is delayed, unknown, or warming"
    if not _positive_defined_risk(row):
        return "candidate is not positive-expectancy defined risk"
    return None


def _hold_invalid_review(context: DaemonContext, review_id: int, reason: str) -> None:
    with context.engine.begin() as conn:
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .where(entry_reviews.c.status.in_(("requested", "processing")))
            .values(
                status="held",
                decision_reason="invalid authorization: " + reason,
                claimed_at=None,
                processed_at=datetime.now(UTC),
            )
        )


def _claim_review(context: DaemonContext, review_id: int) -> bool:
    with context.engine.begin() as conn:
        result = conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .where(entry_reviews.c.status == "requested")
            .values(
                status="processing",
                decision_reason=None,
                claimed_at=datetime.now(UTC),
            )
        )
    return result.rowcount == 1


def _finish_review(
    context: DaemonContext,
    review_id: int,
    *,
    ok: bool,
    message: str,
    order_id: int | None,
    failure_status: str = "held",
) -> bool:
    values: dict[str, object] = {
        "decision_reason": message,
        "claimed_at": None,
    }
    if ok:
        values.update(
            status="submitted",
            processed_at=datetime.now(UTC),
            order_id=order_id,
        )
    else:
        values.update(
            status=failure_status,
            order_id=None,
            processed_at=datetime.now(UTC),
        )
    with context.engine.begin() as conn:
        result = conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .where(entry_reviews.c.status == "processing")
            .values(**values)
        )
    return result.rowcount == 1


def _prior_order_id(context: DaemonContext, score_id: int) -> int | None:
    with context.engine.connect() as conn:
        value = conn.execute(
            select(entry_intent_consumptions.c.first_order_id).where(
                entry_intent_consumptions.c.strategy_score_id == score_id
            )
        ).scalar_one_or_none()
    return int(value) if value is not None else None


def _trip_review_state_race(context: DaemonContext, review_id: int) -> str:
    reason = f"entry review #{review_id} changed state during execution handoff"
    trip_kill(context.engine, reason)
    log.critical("%s; execution kill switch tripped", reason)
    return reason


async def _execute_reviewed_score(
    context: DaemonContext,
    *,
    score_id: int,
    symbol: str,
    strategy: str,
) -> bool:
    """Claim and execute one exact reviewed score through ``execute_pick``."""
    if context.order_client is None:
        log.warning("auto-execute hold score_id=%s: order client unavailable", score_id)
        return False
    overlay = load_overlay_state(context.engine)
    if not overlay.enabled:
        held = hold_pending_reviews(context.engine, overlay)
        log.warning("Hermes overlay disabled; held %d pending review(s)", held)
        return False
    review_id = _requested_review_id_for(context, score_id)
    if review_id is None:
        log.info(
            "auto-execute hold %s/%s score_id=%s: awaiting exact Hermes review",
            symbol,
            strategy,
            score_id,
        )
        return False
    authorization_error = _review_authorization_error(context, review_id, score_id)
    if authorization_error is not None:
        _hold_invalid_review(context, review_id, authorization_error)
        log.error(
            "auto-execute held review_id=%s score_id=%s: %s",
            review_id,
            score_id,
            authorization_error,
        )
        return False
    prior_order_id = _prior_order_id(context, score_id)
    if prior_order_id is not None:
        _hold_invalid_review(
            context,
            review_id,
            f"prior order intent #{prior_order_id} already consumed this candidate",
        )
        return False
    if not _claim_review(context, review_id):
        log.info("auto-execute hold review_id=%s: already claimed", review_id)
        return False
    authorization_error = _review_authorization_error(context, review_id, score_id)
    if authorization_error is not None:
        _hold_invalid_review(context, review_id, authorization_error)
        return False
    # Close the race with an outcomes tick tripping the persisted breaker after
    # this review was claimed but before any broker-facing execution begins.
    overlay = load_overlay_state(context.engine)
    if not overlay.enabled:
        hold_pending_reviews(context.engine, overlay)
        log.warning("Hermes overlay disabled during execution handoff")
        return False

    try:
        # Imported lazily because the engine pulls daemon.market_hours; a
        # module-level import would close a cycle.
        from optionsbot.execution import engine as execution_engine

        walk_md = (
            MarketDataClient(context.exec_ibkr, context.resolver)
            if context.exec_ibkr is not None
            else None
        )
        deps = execution_engine.ExecutionDeps(
            engine=context.engine,
            settings=context.settings,
            order_client=context.order_client,
            md=MarketDataClient(context.ibkr, context.resolver),
            positions=PositionsClient(context.ibkr),
            ibkr_lock=context.ibkr_lock,
            walk_md=walk_md,
            walk_tasks=context.walk_tasks,
        )
        # Outcome accrual + breaker evaluation uses this same lock. Therefore
        # the persisted state cannot trip between this final check and the
        # broker-facing execution pipeline.
        async with context.hermes_overlay_lock:
            overlay = load_overlay_state(context.engine)
            if not overlay.enabled:
                hold_pending_reviews(context.engine, overlay)
                log.warning("Hermes overlay disabled during broker handoff")
                return False
            outcome = await execution_engine.execute_pick(deps, score_id)
        try:
            finished = _finish_review(
                context,
                review_id,
                ok=outcome.ok,
                message=outcome.message,
                order_id=outcome.order_id,
            )
        except Exception:  # noqa: BLE001 -- outcome may already represent a broker side effect
            reason = _trip_review_state_race(context, review_id)
            log.exception("review completion failed after execution outcome")
            await _send(context, f"🚨 {reason}; execution halted")
            return False
        if not finished:
            reason = _trip_review_state_race(context, review_id)
            await _send(context, f"🚨 {reason}; execution halted")
            return False
        log.info(
            "auto-execute %s/%s score_id=%s -> ok=%s | %s",
            symbol,
            strategy,
            score_id,
            outcome.ok,
            outcome.message.replace("\n", " "),
        )
        await _send(
            context,
            f"🤖 auto-execute {symbol} {strategy}:\n{outcome.message}",
        )
        return bool(outcome.ok)
    except Exception as exc:  # noqa: BLE001 -- safety boundary
        reason = (
            f"entry review #{review_id} execution raised after authorization; "
            "broker side effects are uncertain"
        )
        trip_kill(context.engine, reason)
        try:
            _finish_review(
                context,
                review_id,
                ok=False,
                message=f"execution error: {exc}",
                order_id=None,
                failure_status="failed",
            )
        except Exception:  # noqa: BLE001 -- kill switch is already persisted
            log.exception("could not persist failed review after execution exception")
        log.exception("reviewed auto-execution failed for score %s", score_id)
        await _send(context, f"🚨 {reason}; execution halted")
        return False


async def auto_execute_candidates(
    context: DaemonContext,
    candidates: list[tuple[str, ScoredStrategy, int]],
) -> int:
    """Execute only alerted candidates with an exact requested Hermes review."""
    if context.settings.execution.mode != "auto" or context.order_client is None:
        return 0
    overlay = load_overlay_state(context.engine)
    if not overlay.enabled:
        hold_pending_reviews(context.engine, overlay)
        return 0
    submitted = 0
    log.info("auto-execute pass: %d candidate(s)", len(candidates))
    for symbol, scored, snapshot_id in candidates:
        score_id = _score_id_for(context, snapshot_id, scored.strategy_name)
        if score_id is None:
            log.warning(
                "auto-execute skip %s/%s: no score_id for snapshot %s",
                symbol,
                scored.strategy_name,
                snapshot_id,
            )
            continue
        if await _execute_reviewed_score(
            context,
            score_id=score_id,
            symbol=symbol,
            strategy=scored.strategy_name,
        ):
            submitted += 1
    return submitted


async def run_entry_reviews_tick(context: DaemonContext) -> int:
    """Consume delayed Hermes reviews through the normal auto-execution path."""
    if context.settings.execution.mode != "auto" or context.order_client is None:
        return 0
    overlay = load_overlay_state(context.engine)
    if not overlay.enabled:
        held = hold_pending_reviews(context.engine, overlay)
        if held:
            log.warning("Hermes overlay disabled; held %d pending review(s)", held)
        return 0
    now = datetime.now(UTC)
    cutoff = now - timedelta(minutes=context.settings.execution.max_pick_age_minutes)
    stale_score_ids = (
        select(strategy_scores.c.id)
        .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
        .where(snapshots.c.ts < cutoff)
    )
    abandoned = or_(
        entry_reviews.c.claimed_at.is_(None),
        entry_reviews.c.claimed_at < now - timedelta(minutes=10),
    )
    prior_order = exists(
        select(entry_intent_consumptions.c.strategy_score_id).where(
            entry_intent_consumptions.c.strategy_score_id == entry_reviews.c.strategy_score_id
        )
    )
    with context.engine.begin() as conn:
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.status == "processing")
            .where(abandoned)
            .where(prior_order)
            .values(
                status="held",
                claimed_at=None,
                processed_at=now,
                decision_reason="prior order intent consumed abandoned review lease",
            )
        )
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.status == "processing")
            .where(abandoned)
            .where(~prior_order)
            .values(
                status="requested",
                claimed_at=None,
                decision_reason="recovered unconsumed abandoned processing lease",
            )
        )
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.verdict == "vetted_paper_candidate")
            .where(entry_reviews.c.status == "requested")
            .where(entry_reviews.c.strategy_score_id.in_(stale_score_ids))
            .values(
                status="expired",
                decision_reason="original candidate exceeded max_pick_age_minutes",
                processed_at=now,
            )
        )
    with context.engine.connect() as conn:
        rows = conn.execute(
            select(
                entry_reviews.c.id,
                strategy_scores.c.id.label("score_id"),
                strategy_scores.c.strategy,
                snapshots.c.symbol,
            )
            .join(
                strategy_scores,
                entry_reviews.c.strategy_score_id == strategy_scores.c.id,
            )
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(entry_reviews.c.verdict == "vetted_paper_candidate")
            .where(entry_reviews.c.status == "requested")
            .order_by(entry_reviews.c.reviewed_at)
            .limit(20)
        ).fetchall()
    submitted = 0
    for row in rows:
        if await _execute_reviewed_score(
            context,
            score_id=int(row.score_id),
            symbol=str(row.symbol),
            strategy=str(row.strategy),
        ):
            submitted += 1
    return submitted


async def _send(context: DaemonContext, text: str) -> None:
    try:
        await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001
        log.exception("auto-execute notification failed")
