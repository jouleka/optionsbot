"""APScheduler factory for the daemon."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from optionsbot.daemon.context import DaemonContext


def build_scheduler(
    context: DaemonContext,
    scan_job: Callable[[], Awaitable[Any]],
) -> AsyncIOScheduler:
    """Build an AsyncIOScheduler with a single 'scan' job at the configured interval.

    ``scan_job`` is a no-arg async callable that runs one scan tick. The job
    is registered with ``max_instances=1`` so a slow scan never overlaps the
    next tick, and ``coalesce=True`` so a suspended daemon doesn't fire a
    backlog of missed scans on resume.

    The first scan fires ~20s after startup (not one full interval later), so a
    daemon restart doesn't blackout scanning for a whole interval -- the market
    may already be open and the prior tick's picks are stale.
    """
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scan_job,
        trigger=IntervalTrigger(minutes=context.settings.scan.interval_minutes),
        id="scan",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
        next_run_time=datetime.now(UTC) + timedelta(seconds=20),
    )
    return scheduler
