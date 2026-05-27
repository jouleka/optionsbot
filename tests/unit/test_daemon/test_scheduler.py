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
