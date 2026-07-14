"""Console entry points for the isolated market-context MCP servers."""

from __future__ import annotations

import logging

from optionsbot.market_context.server import build_finnhub_server, build_fred_server


def configure_secret_safe_logging() -> None:
    """Prevent request URLs or auth details from reaching MCP stderr logs."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def fred_main() -> int:
    """Run the FRED context server over stdio."""
    configure_secret_safe_logging()
    build_fred_server().run()
    return 0


def finnhub_main() -> int:
    """Run the Finnhub context server over stdio."""
    configure_secret_safe_logging()
    build_finnhub_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit("Use optionsbot-fred-mcp or optionsbot-finnhub-mcp")
