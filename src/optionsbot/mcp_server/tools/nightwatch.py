"""IBK-138 MCP tools for Hermes nightwatch supervision.

These tools deliberately split read-only analyst packets from write-gated actions:
``pending_picks`` is read-only, ``request_exit`` only queues an audited request
for the daemon to evaluate, and ``halt`` trips the existing persisted kill
switch with an exact confirmation token.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import desc, insert, select

from optionsbot.execution.exit_requests import ALLOWED_CATALYST_TYPES
from optionsbot.execution.orders import get_order
from optionsbot.execution.state import trip_kill
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.storage.schema import exit_requests, orders, snapshots, strategy_scores

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
                .where(strategy_scores.c.score >= float(min_score or 0.0))
                .where(snapshots.c.ts >= cutoff)
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
