"""track_record tool (IBK-117): the bot's realized outcomes report."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from optionsbot.mcp_server.context import ServerContext
from optionsbot.validation.outcomes import outcomes_report, report_to_dict


def register(server: FastMCP) -> None:
    @server.tool()
    async def track_record(
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """The bot's realized track record from pick_outcomes: realized win-rate vs the
        predicted PoP (calibration), P&L, overall + by strategy + by risk tier. Reporting
        only -- outcomes are not fed back into scoring."""
        lifespan = ctx.request_context.lifespan_context
        return {"ok": True, **report_to_dict(outcomes_report(lifespan.engine))}
