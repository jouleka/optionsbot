"""Trusted consumer for intents emitted by the restricted Hermes MCP process."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.exit_requests import ALLOWED_CATALYST_TYPES
from optionsbot.execution.state import trip_kill
from optionsbot.hermes_overlay import load_overlay_state
from optionsbot.mcp_server.intent_queue import control_intents, create_intent_engine
from optionsbot.storage.schema import entry_reviews, exit_requests


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
    suffix = " and held by the overlay breaker" if status == "held" else ""
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
    finally:
        intent_engine.dispose()
    return consumed
