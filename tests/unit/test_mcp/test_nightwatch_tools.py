"""Tests for IBK-138 nightwatch MCP tools."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import insert, select

from optionsbot.execution.state import load_state
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.nightwatch import register
from optionsbot.storage.schema import exit_requests, orders, snapshots, strategy_scores
from tests.unit.test_mcp.conftest import FakeCtx, get_tools

NOW = datetime.now(UTC)

LEGS = [
    {
        "symbol": "SPY",
        "side": "sell",
        "sec_type": "OPT",
        "expiry": "20260717",
        "strike": 580.0,
        "right": "P",
        "quantity": 1,
    },
    {
        "symbol": "SPY",
        "side": "buy",
        "sec_type": "OPT",
        "expiry": "20260717",
        "strike": 575.0,
        "right": "P",
        "quantity": 1,
    },
]


def _snapshot_with_score(server_context: ServerContext, *, score: float = 82.0) -> int:
    with server_context.engine.begin() as conn:
        snap_pk = conn.execute(
            insert(snapshots).values(
                symbol="SPY",
                ts=NOW,
                spot=550.0,
                iv_rank=0.62,
                hv20=0.18,
                iv_hv_ratio=1.4,
                expected_move=8.5,
                regime_dir="neutral",
                regime_iv="high",
                raw_json={"earnings_in_window": False, "relative_strength": 0.03},
            )
        ).inserted_primary_key
        snap_id = int(snap_pk[0])
        score_pk = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snap_id,
                strategy="bull_put_spread",
                score=score,
                rationale="positive expectancy, defined risk",
                legs_json=LEGS,
                suggestion_json={
                    "defined_risk": True,
                    "credit_or_debit": 120.0,
                    "max_loss": 380.0,
                    "max_profit": 120.0,
                    "prob_profit": 0.68,
                    "expected_value": 14.5,
                    "suggested_quantity": 1,
                },
            )
        ).inserted_primary_key
    return int(score_pk[0])


def test_pending_picks_returns_grounded_pre_trade_packet(server_context: ServerContext) -> None:
    score_id = _snapshot_with_score(server_context)
    pending_picks = get_tools(register)["pending_picks"]

    result = pending_picks(limit=5, min_score=70.0, max_age_minutes=60, ctx=FakeCtx(server_context))

    assert result["ok"] is True
    assert result["count"] == 1
    pick = result["picks"][0]
    assert pick["pick_id"] == score_id
    assert pick["symbol"] == "SPY"
    assert pick["strategy"] == "bull_put_spread"
    assert pick["suggestion"]["expected_value"] == 14.5
    assert pick["market"]["iv_rank"] == 0.62
    assert pick["market"]["relative_strength"] == 0.03
    assert "news/catalyst corroboration" in result["rubric"]["must_check"]


def test_request_exit_queues_valid_request_for_open_position(server_context: ServerContext) -> None:
    with server_context.engine.begin() as conn:
        pk = conn.execute(
            insert(orders).values(
                intent="open",
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=LEGS,
                quantity=1,
                status="filled",
                staged_ts=NOW,
                submitted_ts=NOW,
                terminal_ts=NOW,
                ib_order_id=11,
                order_ref="obot-1",
                reprice_count=0,
            )
        ).inserted_primary_key
    position_id = int(pk[0])
    request_exit = get_tools(register)["request_exit"]

    result = request_exit(
        position_id=position_id,
        catalyst_type="downgrade_upgrade",
        confidence=0.82,
        sources=["Reuters headline", "price/volume corroboration"],
        reason="Downgrade plus adverse tape; asking daemon gate to evaluate close-only exit.",
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is True
    assert result["status"] == "requested"
    with server_context.engine.connect() as conn:
        row = conn.execute(select(exit_requests)).one()
    assert row.position_id == position_id
    assert row.catalyst_type == "downgrade_upgrade"
    assert row.confidence == 0.82
    assert row.sources_json == ["Reuters headline", "price/volume corroboration"]
    assert row.status == "requested"


def test_request_exit_refuses_unknown_position(server_context: ServerContext) -> None:
    request_exit = get_tools(register)["request_exit"]

    result = request_exit(
        position_id=999,
        catalyst_type="downgrade_upgrade",
        confidence=0.9,
        sources=["source A", "source B"],
        reason="test",
        ctx=FakeCtx(server_context),
    )

    assert result["ok"] is False
    assert result["error"] == "position_not_open"


def test_halt_requires_exact_confirmation(server_context: ServerContext) -> None:
    halt = get_tools(register)["halt"]

    refused = halt(reason="test", confirm="wrong", ctx=FakeCtx(server_context))
    assert refused["ok"] is False
    assert refused["error"] == "confirmation_required"
    assert load_state(server_context.engine).killed is False

    accepted = halt(
        reason="IBK-138 smoke test halt", confirm="HALT_OPTIONSBOT", ctx=FakeCtx(server_context)
    )
    assert accepted["ok"] is True
    assert accepted["killed"] is True
    assert load_state(server_context.engine).killed is True
