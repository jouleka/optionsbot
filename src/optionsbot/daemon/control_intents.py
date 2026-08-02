"""Trusted consumer for intents emitted by the restricted Hermes MCP process."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.exit_requests import ALLOWED_CATALYST_TYPES
from optionsbot.execution.state import trip_kill
from optionsbot.hermes_overlay import load_overlay_state
from optionsbot.mcp_server.intent_queue import control_intents, create_intent_engine
from optionsbot.storage.schema import alerts, entry_reviews, exit_requests, orders, strategy_scores

_REQUIRED_PROPOSAL_CHECKS = {
    "bot_health",
    "regime_history",
    "catalysts",
}

log = logging.getLogger(__name__)


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _clean_sources(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("sources must be a list")
    clean = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(clean) != len(value) or len({item.casefold() for item in clean}) != len(clean):
        raise ValueError("sources must be distinct non-empty strings")
    return clean


def _consume_entry_review(context: DaemonContext, payload: dict[str, Any]) -> str:
    pick_id = int(payload["pick_id"])
    alert_id = int(payload["alert_id"])
    reviewed_at = _timestamp(payload["reviewed_at"], "reviewed_at")
    verdict = str(payload["verdict"])
    if verdict not in {"vetted_paper_candidate", "watch_only", "no_trade"}:
        raise ValueError("unknown review verdict")
    confidence = float(payload["confidence"])
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("review confidence must be finite within [0, 1]")
    sources = _clean_sources(payload["sources"])
    reason = str(payload["reason"]).strip()
    checks = payload["checks"]
    if not reason or not isinstance(checks, dict):
        raise ValueError("review reason/checks are required")
    status = {
        "vetted_paper_candidate": "requested",
        "watch_only": "held",
        "no_trade": "refused",
    }[verdict]
    decision_reason = None
    if verdict == "vetted_paper_candidate":
        overlay = load_overlay_state(context.engine)
        if not overlay.enabled:
            status = "held"
            decision_reason = "overlay breaker: " + (
                overlay.reason or "Hermes overlay correctness breaker is disabled"
            )
    try:
        with context.engine.begin() as conn:
            pk = conn.execute(
                insert(entry_reviews).values(
                    strategy_score_id=pick_id,
                    alert_id=alert_id,
                    reviewed_at=reviewed_at,
                    verdict=verdict,
                    confidence=confidence,
                    sources_json=sources,
                    reason=reason,
                    checks_json=checks,
                    status=status,
                    decision_reason=decision_reason,
                    processed_at=datetime.now(UTC) if status == "held" else None,
                )
            ).inserted_primary_key
    except IntegrityError:
        with context.engine.connect() as conn:
            existing = conn.execute(
                select(entry_reviews.c.id).where(entry_reviews.c.strategy_score_id == pick_id)
            ).scalar_one_or_none()
        if existing is None:
            raise
        return f"entry review already existed as #{int(existing)}"
    assert pk is not None
    if verdict == "watch_only":
        suffix = " (watch-only; no order authority)"
    elif status == "held":
        suffix = " and held by the overlay breaker"
    else:
        suffix = ""
    return f"entry review imported as #{int(pk[0])}{suffix}"


def _consume_exit_request(context: DaemonContext, payload: dict[str, Any]) -> str:
    position_id = int(payload["position_id"])
    requested_at = _timestamp(payload["requested_at"], "requested_at")
    catalyst = str(payload["catalyst_type"]).strip().lower()
    if catalyst not in ALLOWED_CATALYST_TYPES:
        raise ValueError("unknown catalyst_type")
    confidence = float(payload["confidence"])
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("exit confidence must be finite within [0, 1]")
    sources = _clean_sources(payload["sources"])
    reason = str(payload["reason"]).strip()
    if not reason:
        raise ValueError("exit reason is required")
    with context.engine.connect() as conn:
        existing = conn.execute(
            select(exit_requests.c.id)
            .where(exit_requests.c.position_id == position_id)
            .where(exit_requests.c.requested_at == requested_at)
        ).scalar_one_or_none()
    if existing is not None:
        return f"exit request already existed as #{int(existing)}"
    with context.engine.begin() as conn:
        pk = conn.execute(
            insert(exit_requests).values(
                position_id=position_id,
                requested_at=requested_at,
                catalyst_type=catalyst,
                confidence=confidence,
                sources_json=sources,
                reason=reason,
                status="requested",
            )
        ).inserted_primary_key
    assert pk is not None
    return f"exit request imported as #{int(pk[0])} for daemon-side gating"


async def _consume_entry_proposal(context: DaemonContext, payload: dict[str, Any]) -> str:
    """Rebuild a Hermes idea from live data; OptionsBot remains authoritative."""
    from optionsbot.analysis.opening_range_fvg import detect_opening_range_fvg
    from optionsbot.daemon.alert_pipeline import enqueue_alert
    from optionsbot.daemon.auto_executor import auto_execute_candidates
    from optionsbot.daemon.candidate_evidence import (
        capture_candidate_evidence,
        with_reconciled_economics,
    )
    from optionsbot.daemon.market_hours import (
        is_market_open,
        minutes_to_nyse_close,
        nyse_session_start_utc,
    )
    from optionsbot.daemon.scan_runner import (
        candidate_admission_blockers,
        rank_alert_candidates,
    )
    from optionsbot.ibkr.history import HistoryClient
    from optionsbot.ibkr.positions import PositionsClient
    from optionsbot.scan import scan_symbol
    from optionsbot.strategies import get_strategy

    proposed_at = _timestamp(payload["proposed_at"], "proposed_at")
    now = datetime.now(UTC)
    if proposed_at > now + timedelta(minutes=1) or now - proposed_at > timedelta(minutes=10):
        raise ValueError("proposal is stale or future-dated")
    symbol = str(payload["symbol"]).strip().upper()
    direction = str(payload["direction"]).strip().lower()
    iv_regime = str(payload["iv_regime"]).strip().lower()
    strategy = str(payload.get("strategy") or "auto").strip().lower()
    if direction not in {"bull", "neutral", "bear"}:
        raise ValueError("invalid direction")
    if iv_regime not in {"high", "neutral", "low"}:
        raise ValueError("invalid IV regime")
    universe = context.settings.screener.universe or []
    if symbol not in universe:
        raise ValueError("symbol is outside the configured paper proposal universe")
    if strategy != "auto":
        try:
            get_strategy(strategy)
        except KeyError as exc:
            raise ValueError("unknown strategy") from exc
    confidence = float(payload["confidence"])
    if not math.isfinite(confidence) or not 0.65 <= confidence <= 1.0:
        raise ValueError("proposal confidence must be within [0.65, 1]")
    sources = _clean_sources(payload["sources"])
    if len(sources) < 2:
        raise ValueError("proposal requires two distinct sources")
    thesis = str(payload["thesis"]).strip()
    checks = payload["checks"]
    if not thesis or not isinstance(checks, dict):
        raise ValueError("proposal thesis/checks are required")
    if set(checks) != _REQUIRED_PROPOSAL_CHECKS or any(
        checks.get(name) is not True for name in _REQUIRED_PROPOSAL_CHECKS
    ):
        raise ValueError("all proposal research checks must pass")
    if not (
        context.settings.execution.paper_only
        and context.settings.ibkr.paper
        and context.settings.execution.zero_dte_only
    ):
        raise ValueError("Hermes proposals require exact-0DTE paper mode")
    if not is_market_open(now):
        raise ValueError("market is closed")
    minutes_left = minutes_to_nyse_close(now)
    if (
        minutes_left is None
        or minutes_left <= context.settings.execution.zero_dte_entry_cutoff_minutes
    ):
        raise ValueError("proposal arrived after the 0DTE entry cutoff")

    opening_range_enabled = bool(
        getattr(context.settings.scan, "opening_range_fvg_enabled", False)
    )
    opening_signal = None
    if opening_range_enabled:
        if direction not in {"bull", "bear"}:
            return "proposal declined: opening-range/FVG mode requires bull or bear direction"
        market_open_at = nyse_session_start_utc(now) + timedelta(hours=9, minutes=30)
        range_end = market_open_at + timedelta(
            minutes=context.settings.scan.opening_range_minutes
        )
        entry_end = market_open_at + timedelta(
            minutes=context.settings.scan.opening_range_entry_window_minutes
        )
        if not range_end <= now <= entry_end:
            return "proposal declined: outside the 09:40–11:00 ET opening-range window"

    async with context.ibkr_lock:
        if opening_range_enabled:
            intraday = await asyncio.wait_for(
                HistoryClient(context.ibkr, context.resolver).get_intraday_history(
                    symbol,
                    timeframe_minutes=(
                        context.settings.scan.opening_range_timeframe_minutes
                    ),
                ),
                timeout=context.settings.scan.scan_symbol_timeout_s,
            )
            opening_signal = detect_opening_range_fvg(
                intraday,
                symbol=symbol,
                now=now,
                timeframe_minutes=context.settings.scan.opening_range_timeframe_minutes,
                opening_range_minutes=context.settings.scan.opening_range_minutes,
                entry_window_minutes=(
                    context.settings.scan.opening_range_entry_window_minutes
                ),
                stop_pct=context.settings.execution.opening_range_stop_pct,
                target_r_min=context.settings.execution.opening_range_target_r_min,
                target_r_max=context.settings.execution.opening_range_target_r_max,
            )
            if opening_signal is None:
                return "proposal declined: no confirmed opening-range/FVG retest"
            signal_completed = opening_signal.respected_ts.astimezone(UTC) + timedelta(
                minutes=opening_signal.timeframe_minutes
            )
            signal_age = now - signal_completed
            if signal_age < timedelta(0) or signal_age > timedelta(
                minutes=context.settings.scan.opening_range_signal_max_age_minutes
            ):
                return "proposal declined: opening-range/FVG confirmation is stale"
            if opening_signal.direction != direction:
                return (
                    "proposal declined: direction conflicts with confirmed "
                    f"opening-range/FVG {opening_signal.direction} breakout"
                )
        result = await asyncio.wait_for(
            scan_symbol(
                symbol,
                context.ibkr,
                context.engine,
                context.settings,
                resolver=context.resolver,
                view_override=(direction, iv_regime),  # type: ignore[arg-type]
                opening_range_signal=opening_signal,
            ),
            timeout=context.settings.scan.scan_symbol_timeout_s,
        )
        summary = await asyncio.wait_for(
            PositionsClient(context.ibkr).get_account_summary(),
            timeout=context.settings.scan.scan_symbol_timeout_s,
        )
    matching = [
        scored for scored in result.scored if strategy == "auto" or scored.strategy_name == strategy
    ]
    if not matching:
        return (
            f"proposal declined: live {symbol} scan produced no {strategy} structure "
            f"for {direction}/{iv_regime}"
        )
    matching.sort(key=lambda scored: scored.score, reverse=True)
    account_value = (
        float(summary.net_liquidation_usd) if summary.net_liquidation_usd is not None else None
    )
    preliminary_eligible = rank_alert_candidates(
        [(symbol, scored, result.snapshot_id) for scored in matching],
        context.settings.scan.score_threshold,
        account_value,
        context.settings.execution.max_single_trade_risk_pct,
    )
    selected = preliminary_eligible[0][1] if preliminary_eligible else matching[0]
    with context.engine.connect() as conn:
        score_id = conn.execute(
            select(strategy_scores.c.id)
            .where(strategy_scores.c.snapshot_id == result.snapshot_id)
            .where(strategy_scores.c.strategy == selected.strategy_name)
        ).scalar_one()
    legs = [
        {
            "symbol": leg.symbol,
            "side": leg.side,
            "sec_type": leg.sec_type,
            "expiry": leg.expiry,
            "strike": leg.strike,
            "right": leg.right,
            "quantity": leg.quantity,
        }
        for leg in selected.suggestion.legs
    ]
    evidence = await capture_candidate_evidence(
        context,
        score_id=int(score_id),
        symbol=symbol,
        legs=legs,
    )
    selected = replace(
        selected,
        suggestion=with_reconciled_economics(selected.suggestion, evidence),
    )

    admission_blockers = candidate_admission_blockers(
        selected,
        context.settings.scan.score_threshold,
        account_value,
        context.settings.execution.max_single_trade_risk_pct,
    )
    if evidence.get("ready") is not True:
        reasons = evidence.get("reasons")
        detail = ",".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
        admission_blockers.append(f"execution_evidence_not_ready({detail or 'unspecified'})")

    alert_id: int | None = None
    bot_eligible = not admission_blockers
    if bot_eligible:
        if await enqueue_alert(context, symbol, selected, result.snapshot_id):
            with context.engine.connect() as conn:
                alert_id = conn.execute(
                    select(alerts.c.id)
                    .where(alerts.c.strategy_score_id == int(score_id))
                    .order_by(alerts.c.id.desc())
                    .limit(1)
                ).scalar_one()
        else:
            bot_eligible = False
            admission_blockers.append("alert_admission_dedup_or_delivery_failed")

    if not bot_eligible:
        decision_reason = "OptionsBot independently declined: " + "; ".join(admission_blockers)
        # The control-intent row is the immutable audit record for a proposal
        # that never passed OptionsBot's own admission gates.  entry_reviews
        # deliberately requires an exact, proven-delivered alert identity, so
        # do not manufacture a nullable review merely to shadow-record a
        # declined hypothesis.
        return (
            f"Hermes proposal declined as score #{int(score_id)} for "
            f"{symbol}/{selected.strategy_name}: "
            f"{decision_reason}"
        )

    assert alert_id is not None
    with context.engine.begin() as conn:
        pk = conn.execute(
            insert(entry_reviews).values(
                strategy_score_id=int(score_id),
                alert_id=alert_id,
                reviewed_at=proposed_at,
                verdict="vetted_paper_candidate",
                confidence=confidence,
                sources_json=sources,
                reason=f"Hermes-originated proposal: {thesis}",
                checks_json=checks,
                status="requested",
                decision_reason=None,
                processed_at=None,
            )
        )
        assert pk.inserted_primary_key is not None
        review_id = int(pk.inserted_primary_key[0])

    submitted = await auto_execute_candidates(
        context,
        [(symbol, selected, result.snapshot_id)],
    )
    with context.engine.begin() as conn:
        order_id = conn.execute(
            select(orders.c.id)
            .where(orders.c.strategy_score_id == int(score_id))
            .order_by(orders.c.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .values(
                status="submitted" if submitted else "held",
                order_id=order_id,
                processed_at=datetime.now(UTC),
                decision_reason=(
                    "OptionsBot independently accepted and submitted"
                    if submitted
                    else "OptionsBot execution gates declined after proposal admission"
                ),
            )
        )
    return (
        f"Hermes proposal #{review_id} rebuilt as score #{int(score_id)}; "
        f"OptionsBot submitted={bool(submitted)}"
    )


def _consume_one(context: DaemonContext, kind: str, payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("intent payload must be an object")
    if kind == "entry_review":
        return _consume_entry_review(context, payload)
    if kind == "request_exit":
        return _consume_exit_request(context, payload)
    if kind == "halt":
        reason = str(payload.get("reason") or "Hermes restricted MCP halt")
        state = trip_kill(context.engine, reason)
        return f"kill switch set: {state.reason}"
    raise ValueError(f"unknown intent kind: {kind}")


def consume_control_intents(
    context: DaemonContext,
    intent_db_path: Path | str,
    *,
    limit: int = 50,
) -> int:
    """Validate/import pending intents; return the number successfully consumed."""
    intent_engine = create_intent_engine(intent_db_path)
    consumed = 0
    try:
        with intent_engine.connect() as conn:
            rows = conn.execute(
                select(control_intents)
                .where(control_intents.c.status == "pending")
                .order_by(control_intents.c.id)
                .limit(max(1, min(int(limit), 100)))
            ).fetchall()
        for row in rows:
            now = datetime.now(UTC)
            try:
                result = _consume_one(context, str(row.kind), row.payload_json)
                status = "processed"
                consumed += 1
            except Exception as exc:  # noqa: BLE001 -- reject malformed/untrusted intent
                result = f"rejected: {type(exc).__name__}: {exc}"
                status = "rejected"
            with intent_engine.begin() as conn:
                conn.execute(
                    update(control_intents)
                    .where(control_intents.c.id == row.id)
                    .where(control_intents.c.status == "pending")
                    .values(status=status, processed_at=now, result_text=result)
                )
            if status == "rejected":
                log.warning(
                    "restricted intent rejected id=%s kind=%s result=%s",
                    row.id,
                    row.kind,
                    result,
                )
            else:
                log.info(
                    "restricted intent processed id=%s kind=%s result=%s",
                    row.id,
                    row.kind,
                    result,
                )
    finally:
        intent_engine.dispose()
    return consumed


async def consume_control_intents_async(
    context: DaemonContext,
    intent_db_path: Path | str,
    *,
    limit: int = 50,
) -> int:
    """Async consumer variant that can rebuild Hermes entry proposals live."""
    intent_engine = create_intent_engine(intent_db_path)
    consumed = 0
    try:
        with intent_engine.connect() as conn:
            rows = conn.execute(
                select(control_intents)
                .where(control_intents.c.status == "pending")
                .order_by(control_intents.c.id)
                .limit(max(1, min(int(limit), 100)))
            ).fetchall()
        # Never let a live proposal scan delay a halt or close request queued
        # behind it. Process control actions first and at most one expensive
        # proposal per scheduler pass; remaining proposals stay pending.
        ordered = sorted(rows, key=lambda row: str(row.kind) == "entry_proposal")
        proposal_started = False
        for row in ordered:
            if str(row.kind) == "entry_proposal":
                if proposal_started:
                    continue
                proposal_started = True
            now = datetime.now(UTC)
            try:
                payload = row.payload_json
                if not isinstance(payload, dict):
                    raise ValueError("intent payload must be an object")
                if str(row.kind) == "entry_proposal":
                    result = await _consume_entry_proposal(context, payload)
                else:
                    result = _consume_one(context, str(row.kind), payload)
                status = "processed"
                consumed += 1
            except Exception as exc:  # noqa: BLE001 -- untrusted proposal fails closed
                result = f"rejected: {type(exc).__name__}: {exc}"
                status = "rejected"
            with intent_engine.begin() as conn:
                conn.execute(
                    update(control_intents)
                    .where(control_intents.c.id == row.id)
                    .where(control_intents.c.status == "pending")
                    .values(status=status, processed_at=now, result_text=result)
                )
            if status == "rejected":
                log.warning(
                    "restricted intent rejected id=%s kind=%s result=%s",
                    row.id,
                    row.kind,
                    result,
                )
            else:
                log.info(
                    "restricted intent processed id=%s kind=%s result=%s",
                    row.id,
                    row.kind,
                    result,
                )
    finally:
        intent_engine.dispose()
    return consumed
