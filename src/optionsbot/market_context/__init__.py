"""Bounded, read-only external market context for Hermes."""

from optionsbot.market_context.clients import FinnhubClient, FredClient
from optionsbot.market_context.server import build_finnhub_server, build_fred_server

__all__ = [
    "FinnhubClient",
    "FredClient",
    "build_finnhub_server",
    "build_fred_server",
]
