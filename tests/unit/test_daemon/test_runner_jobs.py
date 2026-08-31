"""Tests for Daemon periodic-job registration (IBK-117)."""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from optionsbot.config import Settings
from optionsbot.daemon.runner import Daemon, managed_capture_cron_seconds


def _daemon(outcomes_eval_hours: int, *, zero_dte_only: bool = False) -> Daemon:
    s = Settings()
    s.validation.outcomes_eval_hours = outcomes_eval_hours
    s.execution.zero_dte_only = zero_dte_only
    d = Daemon(s)
    d._scheduler = AsyncIOScheduler()  # constructed, not started -> jobs are retrievable
    return d


def test_register_periodic_jobs_adds_outcomes_when_enabled() -> None:
    d = _daemon(24)
    d._register_periodic_jobs()
    assert d._scheduler.get_job("outcomes") is not None
    assert d._scheduler.get_job("heartbeat") is not None  # default heartbeat_minutes 60 > 0


def test_zero_dte_outcomes_settle_within_fifteen_minutes_after_close() -> None:
    d = _daemon(24, zero_dte_only=True)
    d._register_periodic_jobs()
    job = d._scheduler.get_job("outcomes")
    assert job is not None
    assert job.trigger.interval.total_seconds() == 15 * 60


def test_non_zero_dte_outcomes_keep_configured_hourly_cadence() -> None:
    d = _daemon(24)
    d._register_periodic_jobs()
    job = d._scheduler.get_job("outcomes")
    assert job is not None
    assert job.trigger.interval.total_seconds() == 24 * 60 * 60


def test_register_periodic_jobs_skips_outcomes_when_zero() -> None:
    d = _daemon(0)
    d._register_periodic_jobs()
    assert d._scheduler.get_job("outcomes") is None


def test_register_periodic_jobs_adds_entry_review_consumer() -> None:
    d = _daemon(0)
    d._register_periodic_jobs()

    job = d._scheduler.get_job("entry_reviews")
    assert job is not None
    assert job.trigger.interval.total_seconds() == 60


def test_register_periodic_jobs_offsets_managed_capture_from_exits() -> None:
    d = _daemon(0)
    d._register_periodic_jobs()

    job = d._scheduler.get_job("managed_capture")
    assert job is not None
    assert "second='5,20,35,50'" in str(job.trigger)
    assert managed_capture_cron_seconds(15, 5) == "5,20,35,50"


def test_register_periodic_jobs_can_disable_managed_capture() -> None:
    d = _daemon(0)
    d._settings.validation.managed_capture_enabled = False
    d._register_periodic_jobs()

    assert d._scheduler.get_job("managed_capture") is None
