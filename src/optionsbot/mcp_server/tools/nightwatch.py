"""IBK-138 MCP tools for Hermes nightwatch supervision.

These tools deliberately split read-only analyst packets from write-gated actions:
``pending_picks`` is read-only, ``request_exit`` only queues an audited request
for the daemon to evaluate, and ``halt`` trips the existing persisted kill
switch with an exact confirmation token.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import desc, insert, select
from sqlalchemy.exc import IntegrityError

from optionsbot.execution.exit_requests import ALLOWED_CATALYST_TYPES
from optionsbot.execution.orders import get_order
from optionsbot.execution.risk_structure import has_structurally_defined_option_risk
from optionsbot.execution.state import trip_kill
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.serialization import iso_utc
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
ENTRY_VERDICT_STATUS = {
    "vetted_paper_candidate": "requested",
    "watch_only": "held",
    "no_trade": "refused",
}


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
        },
        "legs": list(row.legs_json or []),
        "rationale": row.rationale,
    }


def _is_positive_defined_risk_candidate(row: Any) -> bool:
    legs = row.legs_json
    suggestion = row.suggestion_json
    if (
        not has_structurally_defined_option_risk(legs)
        or not isinstance(suggestion, dict)
    ):
        return False
    if suggestion.get("defined_risk") is not True:
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


def register(server: FastMCP) -> None:
    @server.tool()
    def pending_picks(
        limit: int,
        min_score: float,
        max_age_minutes: int,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Return recent persisted option candidates for Hermes pre-trade review.

        Read-only. This does not scan IBKR, place orders, or reserve picks. Use
        the returned rubric to check bid/ask, greeks, volume, news/catalysts,
        history, and portfolio context before any human-triggered execution.
        """
        lifespan = ctx.request_context.lifespan_context
        lim = max(1, min(int(limit or 10), 50))
        cutoff = datetime.now(UTC) - timedelta(minutes=max(1, int(max_age_minutes or 60)))
        with lifespan.engine.connect() as conn:
            rows = conn.execute(
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
            picks.append(_pick_dict(SimpleNamespace(**data)))
        return {
            "ok": True,
            "count": len(picks),
            "picks": picks,
            "rubric": RUBRIC,
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
                )
                .where(alerts.c.id == int(alert_id))
            ).first()
            any_alert = conn.execute(
                select(alerts.c.id)
                .where(alerts.c.strategy_score_id == int(pick_id))
                .limit(1)
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
            alert.sent_ts.replace(tzinfo=UTC)
            if alert.sent_ts.tzinfo is None
            else alert.sent_ts
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
            if raw.get("delayed") is not False or raw.get("warming_up") is not False:
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
                    select(entry_reviews).where(
                        entry_reviews.c.strategy_score_id == int(pick_id)
                    )
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
        position = get_order(lifespan.engine, int(position_id))
        now = datetime.now(UTC)
        clean_sources = [str(s).strip() for s in sources if str(s).strip()]
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
        state = trip_kill(lifespan.engine, msg)
        return {
            "ok": True,
            "killed": state.killed,
            "reason": state.reason,
            "ts": iso_utc(state.ts),
        }
