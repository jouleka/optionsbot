"""Entry point for the `optionsbot-daemon` console script."""

from __future__ import annotations

import argparse
import asyncio

from optionsbot.config import get_settings
from optionsbot.daemon.runner import Daemon
from optionsbot.observability import configure_logging


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="optionsbot-daemon",
        description=(
            "Run the optionsbot scanning daemon: scheduled option-chain scans with "
            "Telegram alerts. Config is read from ~/.config/optionsbot/config.toml and "
            "OPTIONSBOT_* env vars. Send SIGHUP to reload config without a restart."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _parse_args(argv)
    settings = get_settings()
    configure_logging(log_level=settings.log_level, env="dev")
    daemon = Daemon(settings=settings)
    return asyncio.run(daemon.start())


if __name__ == "__main__":
    raise SystemExit(main())
