"""Lifespan-scoped container for daemon state."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from optionsbot.config import Settings
from optionsbot.daemon.telegram_client import TelegramClient
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver


@dataclass
class DaemonContext:
    """Shared state for the daemon process.

    Unlike the MCP's ServerContext, the IBKRClient is eagerly constructed
    and the ContractResolver is shared across all watchlist scans within
    a single tick (and across ticks) to maximize contract-cache reuse.
    """

    settings: Settings
    engine: Engine
    ibkr: IBKRClient
    resolver: ContractResolver
    telegram: TelegramClient
