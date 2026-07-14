"""Entry point for the `optionsbot-mcp` console script."""

from __future__ import annotations

import os

from optionsbot.mcp_server.server import build_server
from optionsbot.observability import configure_logging


def main() -> int:
    """Run the optionsbot MCP server over stdio."""
    restricted = os.environ.get("OPTIONSBOT_MCP_RESTRICTED", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if restricted:
        log_level = os.environ.get("OPTIONSBOT_MCP_LOG_LEVEL", "INFO")
    else:
        from optionsbot.config import get_settings

        log_level = get_settings().log_level
    configure_logging(log_level=log_level, env="dev")
    server = build_server(restricted=restricted)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
