"""Entry point for the `optionsbot-mcp` console script."""

from __future__ import annotations

import logging

from optionsbot.config import get_settings
from optionsbot.mcp_server.server import build_server


def main() -> int:
    """Run the optionsbot MCP server over stdio."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = build_server()
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
