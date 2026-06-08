"""Tests for the positions MCP tool (IBK-112)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.positions import register
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
