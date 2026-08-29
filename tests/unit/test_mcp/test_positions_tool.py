"""Tests for the positions MCP tool (IBK-112)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import Engine, insert

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.positions import _persisted_open_book, register
from optionsbot.storage.schema import orders, position_settlements
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


async def test_positions_tool_returns_view(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    fake_view = {
        "as_of": "2026-06-08T00:00:00+00:00", "net_unrealized_pnl": 30.0,
        "group_count": 1, "position_count": 1,
        "groups": [{"underlying": "SPY", "net_unrealized_pnl": 30.0, "legs": []}],
    }
    tool = get_tools(register)["positions"]
    with patch(
        "optionsbot.mcp_server.tools.positions.assemble_open_book",
        new=AsyncMock(return_value=fake_view),
    ):
        result = await tool(ctx=FakeCtx(server_context))
    assert result["ok"] is True
    assert result["net_unrealized_pnl"] == 30.0
    assert result["groups"][0]["underlying"] == "SPY"


async def test_positions_tool_ibkr_failure(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    tool = get_tools(register)["positions"]
    with patch(
        "optionsbot.mcp_server.tools.positions.assemble_open_book",
        new=AsyncMock(side_effect=ConnectionError("down")),
    ):
        result = await tool(ctx=FakeCtx(server_context))
    assert result["ok"] is False and result["error"] == "ibkr_unavailable"


async def test_positions_tool_passes_history_client_and_returns_beta(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    fake_view = {
        "as_of": "2026-06-08T00:00:00+00:00", "net_unrealized_pnl": 0.0,
        "group_count": 0, "position_count": 0, "groups": [],
        "beta_weighted": {"dollar_per_1pct_spy": 480.0, "spy_equiv_shares": 80.0,
                          "underlyings_total": 2, "underlyings_covered": 2,
                          "complete": True, "benchmark": "SPY"},
    }
    mock = AsyncMock(return_value=fake_view)
    tool = get_tools(register)["positions"]
    with patch("optionsbot.mcp_server.tools.positions.assemble_open_book", new=mock):
        result = await tool(ctx=FakeCtx(server_context))
    assert result["ok"] is True
    assert result["beta_weighted"]["spy_equiv_shares"] == 80.0
    # history_client threaded through (portfolio.enabled defaults True)
    assert mock.await_args.kwargs["history_client"] is not None
    assert mock.await_args.kwargs["benchmark_symbol"] == "SPY"


def test_persisted_open_book_excludes_expiration_settlements(
    mcp_engine: Engine,
) -> None:
    now = datetime.now(UTC)
    with mcp_engine.begin() as conn:
        entry_id = int(
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol="SPY",
                    strategy="long_call",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=now,
                    terminal_ts=now,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(position_settlements).values(
                entry_order_id=entry_id,
                kind="expired_worthless",
                expiry="20260828",
                terminal_spot=500.0,
                pnl=-25.0,
                commissions=0.70,
                settled_at=now,
            )
        )

    result = _persisted_open_book(SimpleNamespace(engine=mcp_engine))  # type: ignore[arg-type]

    assert result["position_count"] == 0
    assert result["positions"] == []
