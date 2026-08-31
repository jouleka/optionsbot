"""Read and submit non-authoritative Hermes context-critic observations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import ValidationError
from sqlalchemy import func, select

from optionsbot.hermes_context import (
    CONTEXT_CONTRACT_VERSION,
    ContextAnomalyCode,
    HermesContextSubmissionV1,
    classify_context_timing,
    earliest_context_entry,
)
from optionsbot.hermes_context_metrics import context_shadow_report
from optionsbot.mcp_server.intent_queue import control_intents, enqueue_intent
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.storage.schema import (
    managed_context_reviews,
    managed_opportunities,
    orders,
)

CONTEXT_PROBABILITY_MEANING = (
    "Probability that independent external context supports the signal's stated "
    "direction; never probability of profit, target-first, or order authority."
)

_TIMING_PRIORITY = {
    "pretrade": 0,
    "post_cutoff": 1,
    "post_entry": 2,
    "post_outcome": 3,
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _first_order_at_subquery() -> Any:
    return (
        select(
            orders.c.strategy_score_id,
            func.min(orders.c.staged_ts).label("first_order_at"),
        )
        .where(orders.c.intent == "open")
        .where(orders.c.strategy_score_id.is_not(None))
        .group_by(orders.c.strategy_score_id)
        .subquery()
    )


def _opportunity_projection() -> Any:
    first_orders = _first_order_at_subquery()
    return select(
        managed_opportunities,
        first_orders.c.first_order_at,
    ).outerjoin(
        first_orders,
        first_orders.c.strategy_score_id == managed_opportunities.c.strategy_score_id,
    )


def _pending_context_intent_ids(
    lifespan: Any,
    *,
    model_version: str,
    prompt_version: str,
) -> set[int]:
    """Return pending opportunity IDs for exactly one critic version."""
    with lifespan.intent_engine.connect() as conn:
        rows = conn.execute(
            select(control_intents.c.payload_json)
            .where(control_intents.c.kind == "context_review")
            .where(control_intents.c.status == "pending")
        ).fetchall()
    pending: set[int] = set()
    for row in rows:
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        submission = payload.get("submission")
        if not isinstance(submission, dict):
            continue
        if (
            submission.get("model_version") != model_version
            or submission.get("prompt_version") != prompt_version
        ):
            continue
        opportunity_id = submission.get("opportunity_id")
        if type(opportunity_id) is int and opportunity_id > 0:
            pending.add(opportunity_id)
    return pending


def _db_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _timing_for_row(row: Any, received_at: datetime) -> str:
    return classify_context_timing(
        received_at=received_at,
        cutoff_at=_db_utc(row.entry_cutoff_at),
        first_entry_at=earliest_context_entry(
            _db_utc(row.entry_ts),
            _db_utc(row.first_order_at),
        ),
        outcome_available_at=_db_utc(row.resolved_at),
    ).value


def _group_key(row: Any) -> tuple[str, str]:
    return str(row.session), str(row.signal_id)


def _row_rank(row: Any, timing: str) -> tuple[int, datetime, int]:
    detected_at = _db_utc(row.detected_at)
    if detected_at is None:
        raise ValueError("managed opportunity detected_at is missing")
    return _TIMING_PRIORITY[timing], detected_at, int(row.id)


def register(server: FastMCP) -> None:
    @server.tool()
    def context_critic_metrics(
        ctx: Context[ServerSession, Any],
    ) -> dict[str, Any]:
        """Evaluate only causally pretrade shadow-policy disagreements."""
        lifespan = ctx.request_context.lifespan_context
        return context_shadow_report(lifespan.engine)

    @server.tool()
    def pending_context_opportunities(
        limit: int,
        max_age_minutes: int,
        model_version: str,
        prompt_version: str,
        ctx: Context[ServerSession, Any],
    ) -> dict[str, Any]:
        """Return managed opportunities lacking this shadow-critic version.

        The queue is research-only. Returning an opportunity does not indicate
        that OptionsBot considers it executable, and no response can authorize
        an order.
        """
        lifespan = ctx.request_context.lifespan_context
        lim = max(1, min(int(limit or 10), 25))
        age_minutes = max(1, min(int(max_age_minutes or 20), 390))
        now = _utc_now()
        cutoff = now - timedelta(minutes=age_minutes)
        pending_ids = _pending_context_intent_ids(
            lifespan,
            model_version=model_version,
            prompt_version=prompt_version,
        )
        with lifespan.engine.connect() as conn:
            rows = conn.execute(
                _opportunity_projection()
                .where(managed_opportunities.c.detected_at >= cutoff)
                .where(managed_opportunities.c.detected_at <= now)
                .where(managed_opportunities.c.created_at <= now)
                .where(managed_opportunities.c.bot_decided_at.is_not(None))
                .where(managed_opportunities.c.bot_decided_at <= now)
                .order_by(managed_opportunities.c.detected_at, managed_opportunities.c.id)
            ).fetchall()
            reviewed_groups = {
                (str(row.session), str(row.signal_id))
                for row in conn.execute(
                    select(
                        managed_opportunities.c.session,
                        managed_opportunities.c.signal_id,
                    )
                    .join(
                        managed_context_reviews,
                        managed_context_reviews.c.opportunity_id == managed_opportunities.c.id,
                    )
                    .where(managed_context_reviews.c.model_version == model_version)
                    .where(managed_context_reviews.c.prompt_version == prompt_version)
                    .distinct()
                )
            }
            pending_groups: set[tuple[str, str]] = set()
            pending_list = sorted(pending_ids)
            for start in range(0, len(pending_list), 500):
                chunk = pending_list[start : start + 500]
                pending_groups.update(
                    (str(row.session), str(row.signal_id))
                    for row in conn.execute(
                        select(
                            managed_opportunities.c.session,
                            managed_opportunities.c.signal_id,
                        ).where(managed_opportunities.c.id.in_(chunk))
                    )
                )

        # A context response describes the shared signal, not the option
        # structure used to transport its packet. Pick one deterministic row
        # per immutable signal identity, preferring a still-causal structure
        # when sibling structures have already crossed an entry/outcome bound.
        representatives: dict[tuple[str, str], tuple[Any, str]] = {}
        excluded_groups = reviewed_groups | pending_groups
        for row in rows:
            key = _group_key(row)
            if key in excluded_groups:
                continue
            timing = _timing_for_row(row, now)
            incumbent = representatives.get(key)
            if incumbent is None or _row_rank(row, timing) < _row_rank(incumbent[0], incumbent[1]):
                representatives[key] = (row, timing)

        ranked = sorted(
            representatives.values(),
            key=lambda item: _row_rank(item[0], item[1]),
        )
        items: list[dict[str, Any]] = []
        for row, timing in ranked[:lim]:
            items.append(
                {
                    "opportunity_id": int(row.id),
                    "opportunity_key": row.opportunity_key,
                    "signal_id": row.signal_id,
                    "session": row.session,
                    "symbol": row.symbol,
                    "direction": row.direction,
                    "setup_type": row.setup_type,
                    "strategy": row.strategy,
                    "detected_at": iso_utc(row.detected_at),
                    "entry_cutoff_at": iso_utc(row.entry_cutoff_at),
                    "managed_entry_at": iso_utc(row.entry_ts),
                    "first_broker_order_at": iso_utc(row.first_order_at),
                    "capture_baseline_action": row.baseline_action,
                    "capture_baseline_reason": row.baseline_reason,
                    "scan_admission_action": row.bot_action,
                    "scan_admission_reason": row.bot_reason,
                    "scan_admission_decided_at": iso_utc(row.bot_decided_at),
                    "admission_eligible": bool(row.admission_eligible),
                    "shadow_only": bool(row.shadow_only),
                    "managed_status": row.status,
                    "timing_now": timing,
                }
            )
        return {
            "ok": True,
            "count": len(items),
            "opportunities": items,
            "packet_tool": "context_opportunity_packet",
            "contract_version": CONTEXT_CONTRACT_VERSION,
            "authority": "shadow_only_no_order_or_halt_authority",
        }

    @server.tool()
    def context_opportunity_packet(
        opportunity_id: int,
        signal_id: str,
        ctx: Context[ServerSession, Any],
    ) -> dict[str, Any]:
        """Return one immutable managed-opportunity packet by exact identity."""
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.connect() as conn:
            row = conn.execute(
                _opportunity_projection()
                .where(managed_opportunities.c.id == int(opportunity_id))
                .where(managed_opportunities.c.signal_id == signal_id)
            ).one_or_none()
        if row is None:
            return {
                "ok": False,
                "error": "managed_opportunity_identity_not_found",
                "opportunity_id": int(opportunity_id),
                "signal_id": signal_id,
            }
        if row.bot_decided_at is None:
            return {
                "ok": False,
                "error": "bot_scan_admission_not_frozen",
                "opportunity_id": int(opportunity_id),
                "signal_id": signal_id,
            }
        now = datetime.now(UTC)
        return {
            "ok": True,
            "opportunity": {
                "opportunity_id": int(row.id),
                "opportunity_key": row.opportunity_key,
                "signal_id": row.signal_id,
                "session": row.session,
                "symbol": row.symbol,
                "direction": row.direction,
                "setup_type": row.setup_type,
                "strategy": row.strategy,
                "structure_hash": row.structure_hash,
                "legs": list(row.legs_json or []),
                "features": dict(row.features_json or {}),
                "policy_version": row.policy_version,
                "detected_at": iso_utc(row.detected_at),
                "entry_cutoff_at": iso_utc(row.entry_cutoff_at),
                "managed_entry_at": iso_utc(row.entry_ts),
                "first_broker_order_at": iso_utc(row.first_order_at),
                "timeout_at": iso_utc(row.timeout_at),
                "capture_baseline_action": row.baseline_action,
                "capture_baseline_reason": row.baseline_reason,
                "scan_admission_action": row.bot_action,
                "scan_admission_reason": row.bot_reason,
                "scan_admission_decided_at": iso_utc(row.bot_decided_at),
                "admission_eligible": bool(row.admission_eligible),
                "shadow_only": bool(row.shadow_only),
                "scan_admission_scope": (
                    "score, managed EV, defined risk, and live-equity "
                    "affordability; excludes later liquidity, margin, and fill gates"
                ),
                "managed_status": row.status,
                "timing_now": _timing_for_row(row, now),
            },
            "response_contract": {
                "contract_version": CONTEXT_CONTRACT_VERSION,
                "context_probability": CONTEXT_PROBABILITY_MEANING,
                "context_probability_may_be_null": True,
                "allowed_anomaly_codes": [item.value for item in ContextAnomalyCode],
                "evidence_ids": ("stable provider-qualified IDs, not source names or prose"),
                "timing": "assigned from trusted ledger timestamps",
                "authority": "shadow_only_no_order_or_halt_authority",
            },
        }

    @server.tool()
    def submit_context_review(
        contract_version: str,
        opportunity_id: int,
        signal_id: str,
        context_probability: float | None,
        event_conflict: bool,
        anomaly_codes: list[str],
        evidence_ids: list[str],
        model_version: str,
        prompt_version: str,
        ctx: Context[ServerSession, Any],
    ) -> dict[str, Any]:
        """Queue one structured shadow observation; never authorize an action."""
        lifespan = ctx.request_context.lifespan_context
        try:
            submission = HermesContextSubmissionV1.model_validate(
                {
                    "contract_version": contract_version,
                    "opportunity_id": opportunity_id,
                    "signal_id": signal_id,
                    "context_probability": context_probability,
                    "event_conflict": event_conflict,
                    "anomaly_codes": anomaly_codes,
                    "evidence_ids": evidence_ids,
                    "model_version": model_version,
                    "prompt_version": prompt_version,
                }
            )
        except ValidationError as exc:
            return {
                "ok": False,
                "error": "invalid_context_response",
                "details": [
                    {"location": list(item["loc"]), "message": item["msg"]}
                    for item in exc.errors(include_url=False, include_input=False)
                ],
            }
        received_at = datetime.now(UTC)
        with lifespan.engine.connect() as conn:
            row = conn.execute(
                _opportunity_projection().where(
                    managed_opportunities.c.id == submission.opportunity_id
                )
            ).one_or_none()
            existing = conn.execute(
                select(managed_context_reviews.c.id)
                .where(managed_context_reviews.c.opportunity_id == submission.opportunity_id)
                .where(managed_context_reviews.c.model_version == submission.model_version)
                .where(managed_context_reviews.c.prompt_version == submission.prompt_version)
                .order_by(managed_context_reviews.c.id)
                .limit(1)
            ).scalar_one_or_none()
        if row is None:
            return {"ok": False, "error": "unknown_managed_opportunity"}
        if row.signal_id != submission.signal_id:
            return {"ok": False, "error": "managed_opportunity_signal_mismatch"}
        if row.bot_decided_at is None:
            return {"ok": False, "error": "bot_scan_admission_not_frozen"}
        created_at = _db_utc(row.created_at)
        detected_at = _db_utc(row.detected_at)
        decided_at = _db_utc(row.bot_decided_at)
        if (
            created_at is None
            or detected_at is None
            or decided_at is None
            or received_at < created_at
            or received_at < detected_at
            or received_at < decided_at
        ):
            return {"ok": False, "error": "context_receipt_predates_managed_evidence"}
        if existing is not None:
            return {
                "ok": True,
                "already_recorded": True,
                "review_id": int(existing),
                "opportunity_id": submission.opportunity_id,
                "authority": "shadow_only_no_order_or_halt_authority",
            }
        with lifespan.intent_engine.connect() as conn:
            pending_rows = conn.execute(
                select(control_intents.c.id, control_intents.c.payload_json)
                .where(control_intents.c.kind == "context_review")
                .where(control_intents.c.status == "pending")
                .order_by(control_intents.c.id.desc())
                .limit(100)
            ).fetchall()
        for pending in pending_rows:
            payload = pending.payload_json if isinstance(pending.payload_json, dict) else {}
            candidate = payload.get("submission")
            if not isinstance(candidate, dict):
                continue
            if (
                candidate.get("opportunity_id") == submission.opportunity_id
                and candidate.get("model_version") == submission.model_version
                and candidate.get("prompt_version") == submission.prompt_version
            ):
                return {
                    "ok": True,
                    "already_queued": True,
                    "intent_id": int(pending.id),
                    "opportunity_id": submission.opportunity_id,
                    "authority": "shadow_only_no_order_or_halt_authority",
                }
        intent_id, intent_uid = enqueue_intent(
            lifespan.intent_engine,
            "context_review",
            {
                "received_at": received_at.isoformat(),
                "submission": submission.model_dump(mode="json"),
            },
            now=received_at,
        )
        return {
            "ok": True,
            "status": "queued_for_immutable_shadow_audit",
            "intent_id": intent_id,
            "intent_uid": intent_uid,
            "opportunity_id": submission.opportunity_id,
            "timing_at_receipt": _timing_for_row(row, received_at),
            "context_probability_meaning": CONTEXT_PROBABILITY_MEANING,
            "authority": "shadow_only_no_order_or_halt_authority",
        }
