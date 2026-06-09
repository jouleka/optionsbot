"""Tests for the track_record MCP tool (IBK-117)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import insert

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.track_record import register
from optionsbot.storage.schema import pick_outcomes, snapshots, strategy_scores
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


def _seed(engine) -> None:
    with engine.begin() as conn:
        snap_id = conn.execute(insert(snapshots).values(
            symbol="SPY", ts=datetime.now(UTC), spot=400.0)).inserted_primary_key[0]
        score_id = conn.execute(insert(strategy_scores).values(
            snapshot_id=snap_id, strategy="bull_put_spread", score=70.0)).inserted_primary_key[0]
        conn.execute(insert(pick_outcomes).values(
            strategy_score_id=score_id, symbol="SPY", strategy="bull_put_spread", expiry="20260101",
            entry_spot=400.0, predicted_prob_profit=0.6, score=70.0, credit_or_debit=80.0,
            max_profit=80.0, max_loss=420.0, risk_tier="balanced", terminal_spot=410.0,
            realized_pnl=80.0, win=1, evaluated_at=datetime.now(UTC),
        ))


async def test_track_record_tool_reports(server_context: ServerContext) -> None:
    _seed(server_context.engine)
    result = await get_tools(register)["track_record"](ctx=FakeCtx(server_context))
    assert result["ok"] is True
    assert result["overall"]["count"] == 1 and result["overall"]["win_rate"] == 1.0
    assert "bull_put_spread" in result["by_strategy"]


async def test_track_record_tool_empty(server_context: ServerContext) -> None:
    result = await get_tools(register)["track_record"](ctx=FakeCtx(server_context))
    assert result["ok"] is True and result["overall"]["count"] == 0
