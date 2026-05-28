"""Entry point for the `optionsbot-daemon` console script."""

from __future__ import annotations

import asyncio

from optionsbot.config import get_settings
from optionsbot.daemon.runner import Daemon
from optionsbot.observability import configure_logging


def main() -> int:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, env="dev")
    daemon = Daemon(settings=settings)
    return asyncio.run(daemon.start())


if __name__ == "__main__":
    raise SystemExit(main())
