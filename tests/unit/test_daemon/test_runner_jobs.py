"""Tests for Daemon periodic-job registration (IBK-117)."""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from optionsbot.config import Settings
from optionsbot.daemon.runner import Daemon


def _daemon(outcomes_eval_hours: int) -> Daemon:
    s = Settings()
    s.validation.outcomes_eval_hours = outcomes_eval_hours
    d = Daemon(s)
    d._scheduler = AsyncIOScheduler()  # constructed, not started -> jobs are retrievable
    return d


def test_register_periodic_jobs_adds_outcomes_when_enabled() -> None:
    d = _daemon(24)
    d._register_periodic_jobs()
    assert d._scheduler.get_job("outcomes") is not None
    assert d._scheduler.get_job("heartbeat") is not None  # default heartbeat_minutes 60 > 0


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
