"""Entry point for the `optionsbot-mcp` console script."""

from __future__ import annotations

from optionsbot.config import get_settings
from optionsbot.mcp_server.server import build_server
from optionsbot.observability import configure_logging


def main() -> int:
    """Run the optionsbot MCP server over stdio."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, env="dev")
    server = build_server()
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
