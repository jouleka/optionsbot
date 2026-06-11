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
from optionsbot.ibkr.orders import OrderClient


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
    # IBK-126 execution plumbing. The exec connection (clientId 3) is
    # lazy-connected on first order operation; order events are only delivered
    # to the placing clientId, hence the dedicated connection. None when the
    # daemon was built without execution wiring (tests).
    exec_ibkr: IBKRClient | None = None
    order_client: OrderClient | None = None
    # Order-watcher state: terminal orders with terminal_ts beyond this
    # watermark have been notified (initialized lazily to daemon start so
    # restarts don't replay history); registry-miss cancel warnings fire once.
    orders_notified_through: datetime | None = None
    orders_cancel_warned: set[int] = field(default_factory=set)
