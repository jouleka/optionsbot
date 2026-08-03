"""IBK-138 MCP tools for Hermes nightwatch supervision.

These tools deliberately split read-only analyst packets from write-gated actions:
``pending_picks`` is a compact read-only queue, ``pick_review_packet`` returns
one complete candidate without transport truncation, ``request_exit`` only
queues an audited request for the daemon to evaluate, and ``halt`` trips the
existing persisted kill switch with an exact confirmation token.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import desc, insert, select
from sqlalchemy.exc import IntegrityError

from optionsbot.hermes_overlay import learning_feedback
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.intent_queue import (
    control_intents,
    enqueue_intent,
    recent_proposal_decisions,
)
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.review_evidence import review_evidence_ready, snapshot_ready_for_auto
from optionsbot.risk_structure import has_structurally_defined_option_risk
from optionsbot.storage.schema import (
    alerts,
    entry_reviews,
    exit_requests,
    orders,
    snapshots,
    strategy_scores,
)

RUBRIC: dict[str, list[str]] = {
    "must_check": [
        "fresh bid/ask/mid for every option leg and combo NBBO/slippage",
        "delta/gamma/theta/vega exposure and whether the structure is defined-risk",
        "open interest, volume, spread width, and stale/delayed quote flags",
        "implied-volatility rank versus historical volatility and expected move",
        "news/catalyst corroboration",
        "earnings/events, SEC filings, analyst actions, and material headlines",
        "relative strength, trend context, prior bot history, and recent realized P&L",
        "portfolio heat, buying-power usage, existing correlated positions, and beta",
        "paper-only interlock, kill-switch state, and market-hours/fillability",
    ],
    "never": [
        "never treat one headline as sufficient for a trade or exit",
        "never place a trade from Hermes; use the bot's existing execution gates",
        "never queue request_exit for winners unless deterministic bot risk rules also want out",
    ],
    "opening_range_fvg_policy": [
        (
            "Use opening_range_fvg_v1 playbook-specific session history for this "
            "setup; legacy or broad strategy/symbol history is context and cannot "
            "veto an otherwise ready positive-EV candidate."
        ),
        (
            "An absent playbook-specific tuple is a paper-learning cold start, "
            "not a failed regime/history gate."
        ),
        (
            "candidate_scope.review_authorization_units=1 authorizes only the "
            "proven one-unit candidate; suggested_quantity is a non-authoritative "
            "scan hint and the daemon independently resizes and reruns aggregate "
            "risk gates."
        ),
        (
            "For a 0DTE price-action setup, absence of a positive headline is not "
            "a blocker; fail catalysts only for a known material conflict, an "
            "earnings/event-date contradiction, or explicitly required event data "
            "that is unavailable."
        ),
    ],
}

REQUIRED_ENTRY_CHECKS = {
    "bot_health",
    "candidate",
    "microstructure",
    "greeks",
    "regime_history",
    "catalysts",
    "account_risk",
}
REQUIRED_PROPOSAL_CHECKS = {"bot_health", "regime_history", "catalysts"}
ENTRY_VERDICT_STATUS = {
    "vetted_paper_candidate": "requested",
    "watch_only": "held",
    "no_trade": "refused",
}
ALLOWED_CATALYST_TYPES = frozenset(
    {
        "headline_news",
        "downgrade_upgrade",
        "earnings_guidance",
        "sec_filing",
        "macro_rate",
        "volatility_shock",
        "price_action",
        "risk_management",
        "broker_reconcile",
    }
)

_LESSON_SUMMARY_KEYS = (
    "review_id",
    "pick_id",
    "symbol",
    "strategy",
    "playbook",
    "score",
    "verdict",
    "review_context",
    "evidence_ready",
    "call_pnl",
    "call_won",
    "actual_trade_pnl",
    "execution_won",
    "diagnosis",
    "outcome_basis",
    "forecast_useful",
    "decision_value",
    "lesson",
    "exit_reason",
    "max_profit_at_entry",
    "realized_profit_capture_pct",
)


def _raw(row: Any) -> dict[str, Any]:
    return dict(row.raw_json or {})


def _open_position_ids(lifespan: ServerContext) -> set[int]:
    with lifespan.engine.connect() as conn:
        entries = conn.execute(
            select(orders.c.id).where(orders.c.intent == "open").where(orders.c.status == "filled")
        ).fetchall()
        closed = conn.execute(
            select(orders.c.closes_order_id)
            .where(orders.c.intent == "close")
            .where(orders.c.status == "filled")
            .where(orders.c.closes_order_id.is_not(None))
        ).fetchall()
    closed_ids = {int(row.closes_order_id) for row in closed}
    return {int(row.id) for row in entries if int(row.id) not in closed_ids}


def _pending_open_close(lifespan: ServerContext, position_id: int) -> bool:
    with lifespan.engine.connect() as conn:
        row = conn.execute(
            select(orders.c.id)
            .where(orders.c.intent == "close")
            .where(orders.c.closes_order_id == position_id)
            .where(orders.c.status.in_(["staged", "submitting", "submitted", "partial"]))
            .limit(1)
        ).first()
    return row is not None


def _pick_dict(row: Any) -> dict[str, Any]:
    raw = _raw(row)
    suggestion = dict(row.suggestion_json or {})
    return {
        "pick_id": row.score_id,
        "alert_id": row.alert_id,
        "alert_status": row.alert_status,
        "symbol": row.symbol,
        "strategy": row.strategy,
        "score": row.score,
        "snapshot_id": row.snapshot_id,
        "snapshot_ts": iso_utc(row.ts),
        "age_minutes": row.age_minutes,
        "market": {
            "spot": row.spot,
            "iv_rank": row.iv_rank,
            "hv20": row.hv20,
            "iv_hv_ratio": row.iv_hv_ratio,
            "expected_move": row.expected_move,
            "direction": row.regime_dir,
            "iv_regime": row.regime_iv,
            "earnings_in_window": raw.get("earnings_in_window"),
            "relative_strength": raw.get("relative_strength"),
            "volume": raw.get("volume"),
            "average_volume": raw.get("average_volume"),
            "iv_rank_is_proxy": raw.get("iv_rank_is_proxy"),
            "beta_to_benchmark": raw.get("beta_to_benchmark"),
            "beta_benchmark": raw.get("beta_benchmark"),
            "recent_price_history": raw.get("recent_price_history"),
            "opening_range_fvg": raw.get("opening_range_fvg"),
        },
        "suggestion": {
            "defined_risk": suggestion.get("defined_risk"),
            "credit_or_debit": suggestion.get("credit_or_debit"),
            "max_loss": suggestion.get("max_loss"),
            "max_profit": suggestion.get("max_profit"),
            "prob_profit": suggestion.get("prob_profit"),
            "expected_value": suggestion.get("expected_value"),
            "reward_risk": suggestion.get("reward_risk"),
            "suggested_quantity": suggestion.get("suggested_quantity"),
            "risk_tier": suggestion.get("risk_tier"),
            "opening_range_fvg": suggestion.get("opening_range_fvg"),
        },
        "review_evidence": suggestion.get("review_evidence"),
        "legs": list(row.legs_json or []),
        "rationale": row.rationale,
    }


def _pick_summary(row: Any) -> dict[str, Any]:
    suggestion = dict(row.suggestion_json or {})
    evidence = suggestion.get("review_evidence")
    evidence_dict = evidence if isinstance(evidence, dict) else {}
    return {
        "pick_id": row.score_id,
        "alert_id": row.alert_id,
        "symbol": row.symbol,
        "strategy": row.strategy,
        "score": row.score,
        "snapshot_ts": iso_utc(row.ts),
        "age_minutes": row.age_minutes,
        "suggestion": {
            "defined_risk": suggestion.get("defined_risk"),
            "credit_or_debit": suggestion.get("credit_or_debit"),
            "max_loss": suggestion.get("max_loss"),
            "max_profit": suggestion.get("max_profit"),
            "prob_profit": suggestion.get("prob_profit"),
            "expected_value": suggestion.get("expected_value"),
            "suggested_quantity": suggestion.get("suggested_quantity"),
        },
        "evidence_ready": evidence_dict.get("ready") is True,
        "evidence_captured_at": evidence_dict.get("captured_at"),
        "leg_count": len(list(row.legs_json or [])),
        "full_packet_embedded": False,
        "next_tool": "pick_review_packet",
    }


def _compact_learning_feedback(
    feedback: dict[str, Any],
    *,
    relevant_pairs: set[str],
    independent_symbols: set[str],
) -> dict[str, Any]:
    """Bound verbose analyst prose while preserving every learning signal.

    Review reasons can be several thousand characters each. Returning ten of
    them beside ten full option packets exceeded Hermes's 50 KiB tool budget
    and cut a candidate off in the middle of ``review_evidence``. The compact
    queue retains all aggregate statistics and structured lesson fields; the
    exact candidate is fetched separately through ``pick_review_packet``.
    """
    compact = dict(feedback)
    # Exact tuple priors are substantially more useful than conflicting broad
    # marginals, but the complete cross-product can grow without bound. Keep
    # exact rows for queued candidates plus every strategy observed for the
    # three independent-origination symbols used by the analyst pass.
    for summary_name in (
        "forecast_call_summary",
        "terminal_call_summary",
        "actual_trade_summary",
        "guarded_call_summary",
    ):
        raw_summary = feedback.get(summary_name)
        if not isinstance(raw_summary, dict):
            continue
        summary = dict(raw_summary)
        relevant_symbols = independent_symbols | {pair.partition("|")[0] for pair in relevant_pairs}
        raw_symbols = raw_summary.get("by_symbol")
        if isinstance(raw_symbols, dict):
            summary["by_symbol"] = {
                key: value for key, value in raw_symbols.items() if key in relevant_symbols
            }
        raw_pairs = raw_summary.get("by_strategy_symbol")
        if isinstance(raw_pairs, dict):
            summary["by_strategy_symbol"] = {
                key: value
                for key, value in raw_pairs.items()
                if key in relevant_pairs or key.partition("|")[0] in independent_symbols
            }
        raw_pair_sessions = raw_summary.get("by_strategy_symbol_sessions")
        if isinstance(raw_pair_sessions, dict):
            summary["by_strategy_symbol_sessions"] = {
                key: value
                for key, value in raw_pair_sessions.items()
                if key in relevant_pairs or key.partition("|")[0] in independent_symbols
            }
        for playbook_key in (
            "by_playbook_strategy_symbol",
            "by_playbook_strategy_symbol_sessions",
        ):
            raw_playbook_pairs = raw_summary.get(playbook_key)
            if not isinstance(raw_playbook_pairs, dict):
                continue
            filtered: dict[str, Any] = {}
            for key, value in raw_playbook_pairs.items():
                parts = key.split("|", 2)
                if len(parts) != 3:
                    continue
                pair = f"{parts[1]}|{parts[2]}"
                if pair in relevant_pairs or parts[1] in independent_symbols:
                    filtered[key] = value
            summary[playbook_key] = filtered
        compact[summary_name] = summary
    lessons: list[dict[str, Any]] = []
    raw_lessons = feedback.get("recent_lessons")
    if isinstance(raw_lessons, list):
        for raw_lesson in raw_lessons[:5]:
            if not isinstance(raw_lesson, dict):
                continue
            lesson = {key: raw_lesson.get(key) for key in _LESSON_SUMMARY_KEYS}
            review_reason = raw_lesson.get("review_reason")
            if isinstance(review_reason, str) and review_reason:
                lesson["review_reason_summary"] = review_reason[:240]
            lessons.append(lesson)
    compact["recent_lessons"] = lessons
    proposals: list[dict[str, Any]] = []
    raw_proposals = feedback.get("recent_proposal_decisions")
    if isinstance(raw_proposals, list):
        for raw_proposal in raw_proposals[:5]:
            if not isinstance(raw_proposal, dict):
                continue
            proposal = {
                key: raw_proposal.get(key)
                for key in (
                    "intent_id",
                    "symbol",
                    "direction",
                    "iv_regime",
                    "strategy",
                    "confidence",
                    "status",
                )
            }
            decision = raw_proposal.get("decision")
            if isinstance(decision, str) and decision:
                proposal["decision_summary"] = decision[:300]
            proposals.append(proposal)
    compact["recent_proposal_decisions"] = proposals
    return compact


def _candidate_select() -> Any:
    return (
        select(
            strategy_scores.c.id.label("score_id"),
            alerts.c.id.label("alert_id"),
            alerts.c.status.label("alert_status"),
            strategy_scores.c.snapshot_id,
            strategy_scores.c.strategy,
            strategy_scores.c.score,
            strategy_scores.c.rationale,
            strategy_scores.c.legs_json,
            strategy_scores.c.suggestion_json,
            snapshots.c.symbol,
            snapshots.c.ts,
            snapshots.c.spot,
            snapshots.c.iv_rank,
            snapshots.c.hv20,
            snapshots.c.iv_hv_ratio,
            snapshots.c.expected_move,
            snapshots.c.regime_dir,
            snapshots.c.regime_iv,
            snapshots.c.raw_json,
        )
        .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
        .join(alerts, alerts.c.strategy_score_id == strategy_scores.c.id)
        .outerjoin(
            entry_reviews,
            entry_reviews.c.strategy_score_id == strategy_scores.c.id,
        )
    )


def _is_positive_defined_risk_candidate(row: Any) -> bool:
    legs = row.legs_json
    suggestion = row.suggestion_json
    if not has_structurally_defined_option_risk(legs) or not isinstance(suggestion, dict):
        return False
    if suggestion.get("defined_risk") is not True:
        return False
    try:
        premium = float(suggestion["credit_or_debit"])
        max_loss = float(suggestion["max_loss"])
        prob_profit = float(suggestion["prob_profit"])
        expected_value = float(suggestion["expected_value"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    max_profit_raw = suggestion.get("max_profit")
    unbounded_long_option = (
        max_profit_raw is None
        and len(legs) == 1
        and str(legs[0].get("sec_type", "OPT")).upper() == "OPT"
        and str(legs[0].get("side", "")).upper() == "BUY"
    )
    if unbounded_long_option:
        max_profit = None
    else:
        if max_profit_raw is None:
            return False
        try:
            max_profit = float(max_profit_raw)
        except (TypeError, ValueError, OverflowError):
            return False
    values = (premium, max_loss, prob_profit, expected_value)
    return (
        all(math.isfinite(value) for value in values)
        and (unbounded_long_option or (max_profit is not None and math.isfinite(max_profit)))
        and premium != 0
        and max_loss > 0
        and (unbounded_long_option or (max_profit is not None and max_profit > 0))
        and 0 < prob_profit < 1
        and expected_value > 0
    )


def register(server: FastMCP) -> None:
    @server.tool()
    def pending_picks(
        limit: int,
        min_score: float,
        max_age_minutes: int,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Return a compact queue of candidates for Hermes pre-trade review.

        Read-only. Call ``pick_review_packet`` for each queue item before
        deciding it. Splitting the queue from exact packets keeps transport
        output bounded and prevents option evidence from being truncated.
        """
        lifespan = ctx.request_context.lifespan_context
        lim = max(1, min(int(limit or 10), 50))
        cutoff = datetime.now(UTC) - timedelta(minutes=max(1, int(max_age_minutes or 60)))
        with lifespan.engine.connect() as conn:
            rows = conn.execute(
                _candidate_select()
                .where(strategy_scores.c.score >= float(min_score or 0.0))
                .where(snapshots.c.ts >= cutoff)
                .where(alerts.c.status == "sent")
                .where(entry_reviews.c.id.is_(None))
                .order_by(desc(strategy_scores.c.score), desc(snapshots.c.ts))
                .limit(lim)
            ).fetchall()
        now = datetime.now(UTC)
        picks = []
        for row in rows:
            ts = row.ts.replace(tzinfo=UTC) if row.ts.tzinfo is None else row.ts
            data = dict(row._mapping)
            data["age_minutes"] = round((now - ts).total_seconds() / 60, 1)
            picks.append(_pick_summary(SimpleNamespace(**data)))
        feedback = learning_feedback(lifespan.engine, recent_limit=10)
        intent_engine = getattr(lifespan, "intent_engine", None)
        feedback["recent_proposal_decisions"] = (
            recent_proposal_decisions(intent_engine, limit=10) if intent_engine is not None else []
        )
        compact_feedback = _compact_learning_feedback(
            feedback,
            relevant_pairs={f"{pick['symbol']}|{pick['strategy']}" for pick in picks},
            independent_symbols={"SPY", "QQQ", "IWM"},
        )
        return {
            "ok": True,
            "count": len(picks),
            "picks": picks,
            "packet_tool": "pick_review_packet",
            "instructions": (
                "Call pick_review_packet once for every pick_id/alert_id before "
                "submitting its review."
            ),
            # Aggregate statistics retain the full history. A compact recent
            # lesson window keeps the five-minute analyst pass from spending
            # its whole context budget before it reaches independent ideas.
            "learning_feedback": compact_feedback,
        }

    @server.tool()
    def pick_review_packet(
        pick_id: int,
        alert_id: int,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Return one complete, exact candidate packet for a Hermes verdict.

        Read-only. Identity is bound to a delivered alert and an unreviewed
        score. A single packet remains safely below the Hermes transport limit.
        """
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.connect() as conn:
            row = conn.execute(
                _candidate_select()
                .where(strategy_scores.c.id == int(pick_id))
                .where(alerts.c.id == int(alert_id))
                .where(alerts.c.status == "sent")
                .where(entry_reviews.c.id.is_(None))
            ).one_or_none()
        if row is None:
            return {
                "ok": False,
                "error": "pending_candidate_not_found",
                "pick_id": int(pick_id),
                "alert_id": int(alert_id),
            }
        ts = row.ts.replace(tzinfo=UTC) if row.ts.tzinfo is None else row.ts
        data = dict(row._mapping)
        data["age_minutes"] = round(
            (datetime.now(UTC) - ts).total_seconds() / 60,
            1,
        )
        return {
            "ok": True,
            "pick": _pick_dict(SimpleNamespace(**data)),
            "rubric": RUBRIC,
        }

    @server.tool()
    def propose_entry(
        symbol: str,
        direction: Literal["bull", "neutral", "bear"],
        iv_regime: Literal["high", "neutral", "low"],
        strategy: str,
        confidence: float,
        sources: list[str],
        thesis: str,
        checks: dict[str, bool],
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Ask OptionsBot to independently scan and gate a Hermes trade idea.

        This never creates an order. It appends a bounded proposal to the
        unprivileged intent queue; the trusted daemon must reconstruct an exact
        current candidate and pass every normal paper-execution gate.
        """
        lifespan = ctx.request_context.lifespan_context
        clean_symbol = symbol.strip().upper()
        clean_direction = direction.strip().lower()
        clean_iv = iv_regime.strip().lower()
        clean_strategy = strategy.strip().lower().replace(" ", "_") or "auto"
        clean_thesis = thesis.strip()
        if not clean_symbol or not clean_thesis:
            return {"ok": False, "error": "symbol_and_thesis_required"}
        if clean_direction not in {"bull", "neutral", "bear"}:
            return {"ok": False, "error": "direction_must_be_bull_neutral_or_bear"}
        if clean_iv not in {"high", "neutral", "low"}:
            return {"ok": False, "error": "iv_regime_must_be_high_neutral_or_low"}
        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            return {"ok": False, "error": "confidence_must_be_finite_0_to_1"}
        if (
            not math.isfinite(normalized_confidence)
            or normalized_confidence < 0.65
            or normalized_confidence > 1.0
        ):
            return {"ok": False, "error": "confidence_below_0_65_or_invalid"}
        if set(checks) != REQUIRED_PROPOSAL_CHECKS or any(
            checks.get(name) is not True for name in REQUIRED_PROPOSAL_CHECKS
        ):
            return {"ok": False, "error": "all_seven_checks_must_pass"}
        raw_sources = [str(source).strip() for source in sources if str(source).strip()]
        clean_sources: list[str] = []
        seen_sources: set[str] = set()
        for source in raw_sources:
            key = source.casefold()
            if key not in seen_sources:
                seen_sources.add(key)
                clean_sources.append(source)
        if len(clean_sources) < 2:
            return {"ok": False, "error": "two_distinct_sources_required"}
        intent_engine = getattr(lifespan, "intent_engine", None)
        if intent_engine is None:
            return {"ok": False, "error": "proposal_queue_unavailable"}

        # Keep one thesis from being resubmitted every analyst interval. The
        # daemon may still decline it; Hermes can propose a materially different
        # direction/strategy immediately or retry this one after 30 minutes.
        cutoff = datetime.now(UTC) - timedelta(minutes=30)
        with intent_engine.connect() as conn:
            recent = conn.execute(
                select(control_intents.c.id, control_intents.c.payload_json)
                .where(control_intents.c.kind == "entry_proposal")
                .where(control_intents.c.created_at >= cutoff)
                .order_by(control_intents.c.id.desc())
                .limit(50)
            ).fetchall()
        for row in recent:
            payload = row.payload_json if isinstance(row.payload_json, dict) else {}
            if (
                payload.get("symbol") == clean_symbol
                and payload.get("direction") == clean_direction
                and payload.get("strategy") == clean_strategy
            ):
                return {
                    "ok": True,
                    "already_proposed": True,
                    "intent_id": int(row.id),
                    "note": "same symbol/direction/strategy is inside the 30-minute window",
                }
        now = datetime.now(UTC)
        intent_id, intent_uid = enqueue_intent(
            intent_engine,
            "entry_proposal",
            {
                "proposed_at": now.isoformat(),
                "symbol": clean_symbol,
                "direction": clean_direction,
                "iv_regime": clean_iv,
                "strategy": clean_strategy,
                "confidence": normalized_confidence,
                "sources": clean_sources,
                "thesis": clean_thesis,
                "checks": dict(checks),
            },
            now=now,
        )
        return {
            "ok": True,
            "status": "queued_for_optionsbot_validation",
            "intent_id": intent_id,
            "intent_uid": intent_uid,
            "symbol": clean_symbol,
            "note": "OptionsBot will rescan and may decline; Hermes cannot place orders",
        }

    @server.tool()
    def submit_entry_review(
        pick_id: int,
        alert_id: int,
        verdict: str,
        confidence: float,
        sources: list[str],
        reason: str,
        checks: dict[str, bool],
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Persist a Hermes pre-trade review; never place or reserve an order."""
        lifespan = ctx.request_context.lifespan_context
        normalized = verdict.strip().lower().replace(" ", "_")
        status = ENTRY_VERDICT_STATUS.get(normalized)
        if status is None:
            return {
                "ok": False,
                "error": "unknown_verdict",
                "allowed": sorted(ENTRY_VERDICT_STATUS),
            }
        clean_reason = reason.strip()
        if not clean_reason:
            return {"ok": False, "error": "reason_required"}
        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            return {"ok": False, "error": "confidence_must_be_finite_0_to_1"}
        if not math.isfinite(normalized_confidence) or not 0.0 <= normalized_confidence <= 1.0:
            return {"ok": False, "error": "confidence_must_be_finite_0_to_1"}
        if normalized == "vetted_paper_candidate" and normalized_confidence < 0.80:
            return {"ok": False, "error": "confidence_below_threshold"}
        if normalized == "vetted_paper_candidate" and (
            set(checks) != REQUIRED_ENTRY_CHECKS
            or any(checks.get(name) is not True for name in REQUIRED_ENTRY_CHECKS)
        ):
            return {"ok": False, "error": "all_seven_checks_must_pass"}
        now = datetime.now(UTC)
        with lifespan.engine.connect() as conn:
            pick = conn.execute(
                select(
                    strategy_scores.c.id,
                    strategy_scores.c.legs_json,
                    strategy_scores.c.suggestion_json,
                    snapshots.c.ts,
                    snapshots.c.raw_json,
                )
                .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
                .where(strategy_scores.c.id == int(pick_id))
            ).first()
            existing = conn.execute(
                select(entry_reviews).where(entry_reviews.c.strategy_score_id == int(pick_id))
            ).first()
            alert = conn.execute(
                select(
                    alerts.c.id,
                    alerts.c.strategy_score_id,
                    alerts.c.status,
                    alerts.c.ts,
                    alerts.c.sent_ts,
                    alerts.c.telegram_msg_id,
                ).where(alerts.c.id == int(alert_id))
            ).first()
            any_alert = conn.execute(
                select(alerts.c.id).where(alerts.c.strategy_score_id == int(pick_id)).limit(1)
            ).first()
        if pick is None:
            return {"ok": False, "error": "unknown_pick"}
        if existing is not None:
            return {
                "ok": True,
                "already_reviewed": True,
                "review_id": int(existing.id),
                "pick_id": int(pick_id),
                "verdict": existing.verdict,
                "status": existing.status,
            }
        if alert is None and any_alert is None:
            return {"ok": False, "error": "pick_not_alerted"}
        if alert is None:
            return {"ok": False, "error": "unknown_alert"}
        if int(alert.strategy_score_id) != int(pick_id):
            return {"ok": False, "error": "alert_pick_mismatch"}
        if alert.status != "sent":
            return {"ok": False, "error": "alert_not_sent"}
        if alert.sent_ts is None or alert.telegram_msg_id is None:
            return {"ok": False, "error": "alert_delivery_unproven"}
        alert_ts = alert.ts.replace(tzinfo=UTC) if alert.ts.tzinfo is None else alert.ts
        sent_ts = (
            alert.sent_ts.replace(tzinfo=UTC) if alert.sent_ts.tzinfo is None else alert.sent_ts
        )
        if sent_ts < alert_ts or sent_ts > now + timedelta(minutes=1):
            return {"ok": False, "error": "alert_delivery_time_invalid"}
        if normalized == "vetted_paper_candidate":
            pick_ts = pick.ts
            if pick_ts.tzinfo is None:
                pick_ts = pick_ts.replace(tzinfo=UTC)
            age = now - pick_ts.astimezone(UTC)
            max_age = timedelta(minutes=lifespan.settings.execution.max_pick_age_minutes)
            if age > max_age or age < -timedelta(minutes=1):
                return {"ok": False, "error": "stale_pick"}
            raw = dict(pick.raw_json or {})
            if not snapshot_ready_for_auto(raw):
                return {"ok": False, "error": "candidate_data_unready"}
            if not _is_positive_defined_risk_candidate(pick):
                return {"ok": False, "error": "candidate_not_positive_defined_risk"}
        raw_sources = [str(source).strip() for source in sources if str(source).strip()]
        if normalized == "vetted_paper_candidate" and len(raw_sources) < 2:
            return {"ok": False, "error": "two_sources_required"}
        clean_sources: list[str] = []
        seen_sources: set[str] = set()
        for source in raw_sources:
            source_key = source.casefold()
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            clean_sources.append(source)
        if normalized == "vetted_paper_candidate" and len(clean_sources) < 2:
            return {"ok": False, "error": "two_distinct_sources_required"}
        if normalized == "vetted_paper_candidate" and not review_evidence_ready(
            dict(pick.suggestion_json or {}).get("review_evidence"),
            score_id=int(pick_id),
            now=now,
            max_age_minutes=lifespan.settings.execution.max_pick_age_minutes,
        ):
            return {"ok": False, "error": "candidate_evidence_unready"}
        intent_engine = getattr(lifespan, "intent_engine", None)
        if intent_engine is not None:
            intent_id, intent_uid = enqueue_intent(
                intent_engine,
                "entry_review",
                {
                    "pick_id": int(pick_id),
                    "alert_id": int(alert_id),
                    "reviewed_at": now.isoformat(),
                    "verdict": normalized,
                    "confidence": normalized_confidence,
                    "sources": clean_sources,
                    "reason": clean_reason,
                    "checks": dict(checks),
                    "status": status,
                },
                now=now,
            )
            return {
                "ok": True,
                "status": "queued_for_daemon_validation",
                "intent_id": intent_id,
                "intent_uid": intent_uid,
                "pick_id": int(pick_id),
                "alert_id": int(alert_id),
                "note": "restricted MCP cannot write the trading ledger",
            }
        try:
            with lifespan.engine.begin() as conn:
                pk = conn.execute(
                    insert(entry_reviews).values(
                        strategy_score_id=int(pick_id),
                        alert_id=int(alert_id),
                        reviewed_at=now,
                        verdict=normalized,
                        confidence=normalized_confidence,
                        sources_json=clean_sources,
                        reason=clean_reason,
                        checks_json=dict(checks),
                        status=status,
                    )
                ).inserted_primary_key
        except IntegrityError:
            with lifespan.engine.connect() as conn:
                raced = conn.execute(
                    select(entry_reviews).where(entry_reviews.c.strategy_score_id == int(pick_id))
                ).first()
            if raced is None:
                raise
            return {
                "ok": True,
                "already_reviewed": True,
                "review_id": int(raced.id),
                "pick_id": int(pick_id),
                "verdict": raced.verdict,
                "status": raced.status,
            }
        assert pk is not None
        return {
            "ok": True,
            "status": status,
            "review_id": int(pk[0]),
            "pick_id": int(pick_id),
            "alert_id": int(alert_id),
            "note": "queued only; daemon must independently rerun every execution gate",
        }

    @server.tool()
    def request_exit(
        position_id: int,
        catalyst_type: str,
        confidence: float,
        sources: list[str],
        reason: str,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Queue a close-only exit request for the daemon's safety gate.

        This is a write, but it does NOT submit an order. The daemon later checks
        confidence, corroborating sources, deterministic exit state, quote/P&L,
        daily caps, market hours, kill switch, and atomic-close safety before it
        may convert this row into a close order.
        """
        lifespan = ctx.request_context.lifespan_context
        catalyst = catalyst_type.strip().lower()
        if catalyst not in ALLOWED_CATALYST_TYPES:
            return {
                "ok": False,
                "error": "unknown_catalyst_type",
                "allowed": sorted(ALLOWED_CATALYST_TYPES),
            }
        if not 0.0 <= confidence <= 1.0:
            return {"ok": False, "error": "confidence_must_be_0_to_1"}
        if not sources or len([s for s in sources if str(s).strip()]) < 2:
            return {"ok": False, "error": "two_sources_required"}
        if not reason.strip():
            return {"ok": False, "error": "reason_required"}
        open_ids = _open_position_ids(lifespan)
        if int(position_id) not in open_ids:
            return {"ok": False, "error": "position_not_open"}
        if _pending_open_close(lifespan, int(position_id)):
            return {"ok": False, "error": "position_already_closing"}
        with lifespan.engine.connect() as conn:
            position = conn.execute(
                select(
                    orders.c.id,
                    orders.c.symbol,
                    orders.c.strategy,
                    orders.c.quantity,
                ).where(orders.c.id == int(position_id))
            ).first()
        now = datetime.now(UTC)
        clean_sources = [str(s).strip() for s in sources if str(s).strip()]
        intent_engine = getattr(lifespan, "intent_engine", None)
        if intent_engine is not None:
            intent_id, intent_uid = enqueue_intent(
                intent_engine,
                "request_exit",
                {
                    "position_id": int(position_id),
                    "requested_at": now.isoformat(),
                    "catalyst_type": catalyst,
                    "confidence": float(confidence),
                    "sources": clean_sources,
                    "reason": reason.strip(),
                },
                now=now,
            )
            return {
                "ok": True,
                "status": "queued_for_daemon_validation",
                "intent_id": intent_id,
                "intent_uid": intent_uid,
                "position": None
                if position is None
                else {
                    "id": int(position.id),
                    "symbol": position.symbol,
                    "strategy": position.strategy,
                    "quantity": position.quantity,
                },
                "note": "restricted MCP cannot write the trading ledger or submit orders",
            }
        with lifespan.engine.begin() as conn:
            pk = conn.execute(
                insert(exit_requests).values(
                    position_id=int(position_id),
                    requested_at=now,
                    catalyst_type=catalyst,
                    confidence=float(confidence),
                    sources_json=clean_sources,
                    reason=reason.strip(),
                    status="requested",
                )
            ).inserted_primary_key
        assert pk is not None
        request_id = int(pk[0])
        return {
            "ok": True,
            "status": "requested",
            "request_id": request_id,
            "position": None
            if position is None
            else {
                "id": position.id,
                "symbol": position.symbol,
                "strategy": position.strategy,
                "quantity": position.quantity,
            },
            "note": "queued only; daemon must pass request_exit gates before any close order",
        }

    @server.tool()
    def halt(
        reason: str,
        confirm: str,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Trip the persisted execution kill switch.

        Requires exact confirmation ``HALT_OPTIONSBOT``. This is intentionally
        one-way from MCP: re-arming remains the existing human Telegram/CLI path.
        """
        if confirm != "HALT_OPTIONSBOT":
            return {
                "ok": False,
                "error": "confirmation_required",
                "required_confirm": "HALT_OPTIONSBOT",
            }
        lifespan = ctx.request_context.lifespan_context
        msg = reason.strip() or "Hermes MCP halt"
        intent_engine = getattr(lifespan, "intent_engine", None)
        if intent_engine is not None:
            intent_id, intent_uid = enqueue_intent(
                intent_engine,
                "halt",
                {"reason": msg},
            )
            return {
                "ok": True,
                "killed": "pending_daemon_consumption",
                "intent_id": intent_id,
                "intent_uid": intent_uid,
                "reason": msg,
            }
        from optionsbot.execution.state import trip_kill

        state = trip_kill(lifespan.engine, msg)
        return {
            "ok": True,
            "killed": state.killed,
            "reason": state.reason,
            "ts": iso_utc(state.ts),
        }
