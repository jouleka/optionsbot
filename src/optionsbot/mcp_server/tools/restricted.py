"""Extra read/audit tools exposed only by the restricted Hermes endpoint."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import desc, func, select

from optionsbot.mcp_server.intent_queue import control_intents, recent_intents
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.storage.schema import (
    execution_state,
    pick_outcomes,
    scan_runs,
    snapshots,
    strategy_scores,
    symbol_news,
    watchlist,
)


def register(server: FastMCP) -> None:
    @server.tool()
    def analyze(
        symbol: str,
        fresh: bool,
        ctx: Context[ServerSession, Any],
    ) -> dict[str, Any]:
        """Read the latest persisted analysis; live broker scans are unavailable."""
        normalized = symbol.upper().strip()
        if fresh:
            return {
                "ok": False,
                "error": "broker_access_disabled",
                "message": "restricted MCP can only read persisted snapshots",
                "hint": "the trusted daemon owns all live IBKR scans",
                "symbol": normalized,
            }
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.connect() as conn:
            snap = conn.execute(
                select(snapshots)
                .where(snapshots.c.symbol == normalized)
                .order_by(desc(snapshots.c.ts))
                .limit(1)
            ).first()
            if snap is None:
                return {"ok": False, "error": "no_snapshot", "symbol": normalized}
            scores = conn.execute(
                select(strategy_scores)
                .where(strategy_scores.c.snapshot_id == snap.id)
                .order_by(desc(strategy_scores.c.score))
                .limit(3)
            ).fetchall()
        return {
            "ok": True,
            "symbol": normalized,
            "snapshot_id": int(snap.id),
            "snapshot_ts": iso_utc(snap.ts),
            "view": {
                "direction": snap.regime_dir,
                "iv_regime": snap.regime_iv,
                "iv_rank_value": snap.iv_rank,
                "earnings_in_window": (snap.raw_json or {}).get("earnings_in_window"),
                "warming_up": (snap.raw_json or {}).get("warming_up"),
            },
            "top_strategies": [
                {
                    "strategy_name": row.strategy,
                    "score": row.score,
                    "rationale": row.rationale,
                    "legs": list(row.legs_json or []),
                    "suggestion": dict(row.suggestion_json or {}),
                }
                for row in scores
            ],
            "source": "persisted_snapshot",
        }

    @server.tool()
    def list_watchlist(ctx: Context[ServerSession, Any]) -> dict[str, Any]:
        """Read the persisted watchlist; mutations belong to the trusted daemon/CLI."""
        lifespan = ctx.request_context.lifespan_context
        last_scanned = (
            select(snapshots.c.symbol, func.max(snapshots.c.ts).label("last_scanned"))
            .group_by(snapshots.c.symbol)
            .subquery()
        )
        with lifespan.engine.connect() as conn:
            rows = conn.execute(
                select(
                    watchlist.c.symbol,
                    watchlist.c.view_override_dir,
                    watchlist.c.view_override_iv,
                    watchlist.c.notes,
                    watchlist.c.added_at,
                    last_scanned.c.last_scanned,
                )
                .select_from(
                    watchlist.outerjoin(
                        last_scanned, watchlist.c.symbol == last_scanned.c.symbol
                    )
                )
                .order_by(watchlist.c.symbol)
            ).fetchall()
        return {
            "ok": True,
            "entries": [
                {
                    "symbol": row.symbol,
                    "view_override": {
                        "direction": row.view_override_dir,
                        "iv_regime": row.view_override_iv,
                    },
                    "notes": row.notes,
                    "added_at": iso_utc(row.added_at),
                    "last_scanned": iso_utc(row.last_scanned),
                }
                for row in rows
            ],
        }

    @server.tool()
    def daily_brief(
        symbols: list[str] | None,
        ctx: Context[ServerSession, Any],
    ) -> dict[str, Any]:
        """Read-only cross-symbol packet built entirely from persisted snapshots."""
        lifespan = ctx.request_context.lifespan_context
        if symbols:
            resolved = list(dict.fromkeys(item.upper().strip() for item in symbols if item.strip()))
        else:
            with lifespan.engine.connect() as conn:
                resolved = [
                    str(row.symbol)
                    for row in conn.execute(
                        select(watchlist.c.symbol).order_by(watchlist.c.symbol)
                    ).fetchall()
                ]
        ranked: list[tuple[float, dict[str, Any]]] = []
        notes: list[str] = []
        with lifespan.engine.connect() as conn:
            for symbol in resolved:
                snap = conn.execute(
                    select(snapshots)
                    .where(snapshots.c.symbol == symbol)
                    .order_by(desc(snapshots.c.ts))
                    .limit(1)
                ).first()
                if snap is None:
                    notes.append(f"no persisted snapshot for {symbol}")
                    continue
                rows = conn.execute(
                    select(strategy_scores)
                    .where(strategy_scores.c.snapshot_id == snap.id)
                    .order_by(desc(strategy_scores.c.score))
                ).fetchall()
                news = conn.execute(
                    select(symbol_news.c.headlines_json).where(
                        symbol_news.c.symbol == symbol
                    )
                ).scalar_one_or_none()
                setups: list[tuple[float, dict[str, Any]]] = []
                for row in rows:
                    suggestion = dict(row.suggestion_json or {})
                    expected = suggestion.get("expected_value")
                    max_loss = suggestion.get("max_loss")
                    if (
                        isinstance(expected, (int, float))
                        and not isinstance(expected, bool)
                        and isinstance(max_loss, (int, float))
                        and not isinstance(max_loss, bool)
                        and max_loss != 0
                    ):
                        edge = float(expected) / float(max_loss)
                    else:
                        edge = float("-inf")
                    if not math.isfinite(edge):
                        edge = float("-inf")
                    setups.append(
                        (
                            edge,
                            {
                                "strategy": row.strategy,
                                "score": row.score,
                                "expected_value": expected,
                                "max_loss": max_loss,
                                "prob_profit": suggestion.get("prob_profit"),
                                "defined_risk": suggestion.get("defined_risk"),
                                "legs": list(row.legs_json or []),
                            },
                        )
                    )
                setups.sort(key=lambda item: item[0], reverse=True)
                best_edge = setups[0][0] if setups else float("-inf")
                ranked.append(
                    (
                        best_edge,
                        {
                            "symbol": symbol,
                            "snapshot_ts": iso_utc(snap.ts),
                            "view": {
                                "direction": snap.regime_dir,
                                "iv_regime": snap.regime_iv,
                                "iv_rank_value": snap.iv_rank,
                            },
                            "earnings_in_window": (snap.raw_json or {}).get(
                                "earnings_in_window"
                            ),
                            "relative_strength": (snap.raw_json or {}).get(
                                "relative_strength"
                            ),
                            "headlines": list(news or []),
                            "top_setups": [item[1] for item in setups[:3]],
                        },
                    )
                )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return {
            "ok": True,
            "generated_for": resolved,
            "ranked": [item[1] for item in ranked],
            "notes": notes,
            "source": "persisted_snapshots",
        }

    @server.tool()
    def track_record(ctx: Context[ServerSession, Any]) -> dict[str, Any]:
        """Read realized paper outcomes without loading scoring or broker code."""
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.connect() as conn:
            rows = conn.execute(select(pick_outcomes)).fetchall()
        count = len(rows)
        total_pnl = sum(float(row.realized_pnl) for row in rows)
        wins = sum(int(row.win) for row in rows)
        predicted = [
            float(row.predicted_prob_profit)
            for row in rows
            if row.predicted_prob_profit is not None
        ]
        return {
            "ok": True,
            "overall": {
                "count": count,
                "win_rate": wins / count if count else 0.0,
                "mean_predicted_prob_profit": (
                    sum(predicted) / len(predicted) if predicted else 0.0
                ),
                "realized_pnl": total_pnl,
            },
            "source": "persisted_outcomes",
        }

    @server.tool()
    def health(ctx: Context[ServerSession, Any]) -> dict[str, Any]:
        """Read-only persisted daemon health suitable for zero-LLM watchdogs."""
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.connect() as conn:
            last_scan = conn.execute(
                select(scan_runs).order_by(desc(scan_runs.c.id)).limit(1)
            ).first()
            last_snapshot = conn.execute(select(func.max(snapshots.c.ts))).scalar_one()
            state = conn.execute(select(execution_state).limit(1)).first()
        return {
            "ok": True,
            "as_of": datetime.now(UTC).isoformat(),
            "last_scan": None
            if last_scan is None
            else {
                "finished": iso_utc(last_scan.finished),
                "tickers_scanned": last_scan.tickers_scanned,
                "alerts_fired": last_scan.alerts_fired,
            },
            "last_snapshot_ts": iso_utc(last_snapshot),
            "execution_killed": bool(state.killed) if state is not None else False,
            "execution_kill_reason": state.reason if state is not None else None,
            "broker_access": False,
        }

    @server.tool()
    def control_intent_status(
        limit: int,
        ctx: Context[ServerSession, Any],
    ) -> dict[str, Any]:
        """Audit recent bounded intents and their trusted-daemon disposition."""
        lifespan = ctx.request_context.lifespan_context
        rows = recent_intents(lifespan.intent_engine, limit)
        with lifespan.intent_engine.connect() as conn:
            pending = conn.execute(
                select(func.count())
                .select_from(control_intents)
                .where(control_intents.c.status == "pending")
            ).scalar_one()
        return {
            "ok": True,
            "pending": int(pending),
            "intents": [
                {
                    "id": row["id"],
                    "intent_uid": row["intent_uid"],
                    "kind": row["kind"],
                    "created_at": iso_utc(row["created_at"]),
                    "status": row["status"],
                    "processed_at": iso_utc(row["processed_at"]),
                    "result": row["result_text"],
                }
                for row in rows
            ],
        }
