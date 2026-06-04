"""Lifespan-scoped container for daemon state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

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
    # IBK-102: serialize all IBKR market-data work (scheduled tick + on-demand
    # /scan, /screen) so the single market-data line is never used concurrently.
    ibkr_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # In-memory alerting pause (/pause, /resume). Resets to on at restart.
    alerting_paused: bool = False
    # For /status uptime.
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
