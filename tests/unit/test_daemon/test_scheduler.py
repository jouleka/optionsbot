"""Tests for the scheduler factory."""

from __future__ import annotations

from unittest.mock import MagicMock

from optionsbot.daemon.scheduler import build_scheduler


def test_build_scheduler_returns_asyncio_scheduler(daemon_context) -> None:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    job_callable = MagicMock()
    sched = build_scheduler(daemon_context, job_callable)
    assert isinstance(sched, AsyncIOScheduler)


def test_build_scheduler_registers_scan_job_with_correct_interval(daemon_context) -> None:
    daemon_context.settings.scan.interval_minutes = 7
    job_callable = MagicMock()
    sched = build_scheduler(daemon_context, job_callable)
    jobs = sched.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "scan"
    trigger = jobs[0].trigger
    # IntervalTrigger has `.interval` (timedelta).
    assert trigger.interval.total_seconds() == 7 * 60


def test_build_scheduler_job_is_max_one_instance(daemon_context) -> None:
    """A long-running scan must not overlap with the next tick."""
    sched = build_scheduler(daemon_context, MagicMock())
    job = sched.get_jobs()[0]
    assert job.max_instances == 1


def test_build_scheduler_coalesces_missed_runs(daemon_context) -> None:
    """If the daemon was suspended, only run once on resume — not the backlog."""
    sched = build_scheduler(daemon_context, MagicMock())
    job = sched.get_jobs()[0]
    assert job.coalesce is True


def test_build_scheduler_first_scan_fires_shortly_after_start(daemon_context) -> None:
    """The first scan is scheduled ~20s after build, not one full interval later,
    so a daemon restart doesn't blackout scanning for an entire interval."""
    from datetime import UTC, datetime, timedelta

    daemon_context.settings.scan.interval_minutes = 15
    sched = build_scheduler(daemon_context, MagicMock())
    job = sched.get_jobs()[0]
    assert job.next_run_time is not None
    delta = job.next_run_time - datetime.now(UTC)
    # Built at now+20s; far below the 15-min interval. Generous upper slack for
    # slow test machines, but must be well under one interval.
    assert timedelta(seconds=0) <= delta <= timedelta(seconds=90)
