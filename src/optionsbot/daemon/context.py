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
    # IBK-127: strong refs to in-flight price-walk tasks (asyncio only holds
    # weak refs; a GC'd walk would strand its order at the decision mid until
    # the TTL watcher cancels it).
    walk_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    # IBK-128: watermark for the periodic broker reconciliation pass.
    last_reconcile_ts: datetime | None = None
    # IBK-129: entries whose half-closed state was already escalated (a close
    # partially filled then died — human handoff, never auto-restaged).
    exit_handoff_warned: set[int] = field(default_factory=set)
    # PHASE 0 B1: in-memory mirror of the persisted day-start net-liq baseline
    # for the equity drawdown breaker (None until the first net-liq read this
    # session). The authoritative value lives on the execution_state row.
    day_start_net_liq: float | None = None
    # IBK-PHASE0-C1: entries for which a stale-quote suppression of the
    # quote-priced exit was already alerted (re-alerted only after a fresh
    # quote clears it).
    exit_stale_warned: set[int] = field(default_factory=set)
