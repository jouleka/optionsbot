"""Entry point for the `optionsbot-daemon` console script."""

from __future__ import annotations

import asyncio
import logging

from optionsbot.config import get_settings
from optionsbot.daemon.runner import Daemon


def main() -> int:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    daemon = Daemon(settings=settings)
    return asyncio.run(daemon.start())


if __name__ == "__main__":
    raise SystemExit(main())
