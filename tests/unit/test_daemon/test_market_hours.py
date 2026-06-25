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


def test_nyse_session_date_uses_et_not_utc() -> None:
    from datetime import UTC, date

    from optionsbot.daemon.market_hours import nyse_session_date

    # 2026-06-15 22:30 ET is 2026-06-16 02:30 UTC. The session date is the 15th
    # (ET), NOT the 16th — a UTC-keyed boundary would mis-bucket this.
    et_late = datetime(2026, 6, 15, 22, 30, tzinfo=ET)
    assert nyse_session_date(et_late) == date(2026, 6, 15)
    # Same instant expressed in UTC must yield the SAME session date.
    assert nyse_session_date(et_late.astimezone(UTC)) == date(2026, 6, 15)


def test_nyse_session_date_spans_dst_change() -> None:
    from datetime import date

    from optionsbot.daemon.market_hours import nyse_session_date

    # US DST 2026: spring forward Sun 2026-03-08 (EST -05:00 -> EDT -04:00).
    # 22:00 ET on the day BEFORE (still EST) and on a day AFTER (EDT) must both
    # resolve to their own ET calendar date despite the offset change.
    before = datetime(2026, 3, 6, 22, 0, tzinfo=ET)   # EST -05:00
    after = datetime(2026, 3, 10, 22, 0, tzinfo=ET)   # EDT -04:00
    assert nyse_session_date(before) == date(2026, 3, 6)
    assert nyse_session_date(after) == date(2026, 3, 10)


def test_nyse_session_start_utc_converts_et_midnight() -> None:
    from datetime import UTC

    from optionsbot.daemon.market_hours import nyse_session_start_utc

    # ET midnight on 2026-06-15 (EDT -04:00) == 04:00 UTC the same date.
    start = nyse_session_start_utc(datetime(2026, 6, 15, 22, 30, tzinfo=ET))
    assert start == datetime(2026, 6, 15, 4, 0, tzinfo=UTC)
    # ET midnight on 2026-01-15 (EST -05:00) == 05:00 UTC.
    start_winter = nyse_session_start_utc(datetime(2026, 1, 15, 22, 30, tzinfo=ET))
    assert start_winter == datetime(2026, 1, 15, 5, 0, tzinfo=UTC)


def test_nyse_session_date_requires_aware() -> None:
    import pytest

    from optionsbot.daemon.market_hours import nyse_session_date

    with pytest.raises(ValueError):
        nyse_session_date(datetime(2026, 6, 15, 22, 30))
