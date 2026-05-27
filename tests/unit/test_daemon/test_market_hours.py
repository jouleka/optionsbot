"""Tests for the market-hours gate."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from optionsbot.daemon.market_hours import is_market_open

ET = ZoneInfo("America/New_York")


def test_market_open_on_regular_weekday_at_10am_et() -> None:
    # Wednesday 2026-05-27, 10:00 ET — regular trading hours.
    assert is_market_open(datetime(2026, 5, 27, 10, 0, tzinfo=ET))


def test_market_closed_before_930_et() -> None:
    assert not is_market_open(datetime(2026, 5, 27, 9, 0, tzinfo=ET))


def test_market_closed_after_1600_et() -> None:
    assert not is_market_open(datetime(2026, 5, 27, 16, 30, tzinfo=ET))


def test_market_closed_on_saturday() -> None:
    assert not is_market_open(datetime(2026, 5, 30, 12, 0, tzinfo=ET))


def test_market_closed_on_sunday() -> None:
    assert not is_market_open(datetime(2026, 5, 31, 12, 0, tzinfo=ET))


def test_market_closed_on_us_holiday_independence_day() -> None:
    # 2026-07-03 is the observed Independence Day (July 4 is Saturday).
    assert not is_market_open(datetime(2026, 7, 3, 12, 0, tzinfo=ET))


def test_market_closed_on_us_holiday_christmas() -> None:
    assert not is_market_open(datetime(2026, 12, 25, 12, 0, tzinfo=ET))


def test_market_open_handles_utc_input() -> None:
    """Caller may pass UTC datetimes; the function must convert correctly."""
    from datetime import UTC
    # 14:00 UTC = 10:00 ET on a Wednesday — open.
    assert is_market_open(datetime(2026, 5, 27, 14, 0, tzinfo=UTC))


def test_market_closed_when_tz_naive_raises() -> None:
    """Naive datetimes are rejected to avoid silent UTC/ET confusion."""
    with pytest.raises(ValueError):
        is_market_open(datetime(2026, 5, 27, 10, 0))


def test_market_closed_on_early_close_after_1pm_et() -> None:
    """Half-day (e.g., day after Thanksgiving 2026-11-27); market closes 13:00 ET."""
    # 14:00 ET on a half-day should report closed even though normal hours run to 16:00.
    assert not is_market_open(datetime(2026, 11, 27, 14, 0, tzinfo=ET))
