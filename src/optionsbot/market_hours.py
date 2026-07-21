"""NYSE session clock shared by scanning, execution, and daemon scheduling."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import cast
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")
_NYSE = mcal.get_calendar("NYSE")


def nyse_session_close_utc(now: datetime) -> datetime | None:
    """Return today's official NYSE close in UTC, or ``None`` off-session."""
    if now.tzinfo is None:
        raise ValueError("nyse_session_close_utc requires a tz-aware datetime")
    et_date = now.astimezone(ET).date()
    schedule = _NYSE.schedule(start_date=et_date, end_date=et_date)
    if schedule.empty:
        return None
    return cast(
        datetime,
        schedule.iloc[0]["market_close"].to_pydatetime().astimezone(UTC),
    )


def minutes_to_nyse_close(now: datetime) -> float | None:
    """Minutes until today's official close; negative after the close."""
    close = nyse_session_close_utc(now)
    if close is None:
        return None
    return (close - now.astimezone(UTC)).total_seconds() / 60.0


def is_market_open(now: datetime) -> bool:
    """Return True iff ``now`` falls within an official NYSE session."""
    if now.tzinfo is None:
        raise ValueError("is_market_open requires a tz-aware datetime")
    et_date = now.astimezone(ET).date()
    schedule = _NYSE.schedule(start_date=et_date, end_date=et_date)
    if schedule.empty:
        return False
    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()
    return bool(market_open <= now <= market_close)


def nyse_session_date(now: datetime) -> date:
    """Return the America/New_York calendar date containing ``now``."""
    if now.tzinfo is None:
        raise ValueError("nyse_session_date requires a tz-aware datetime")
    return now.astimezone(ET).date()


def nyse_session_start_utc(now: datetime) -> datetime:
    """Return UTC instant of ET midnight for ``now``'s NYSE session date."""
    session = nyse_session_date(now)
    et_midnight = datetime(
        session.year, session.month, session.day, 0, 0, tzinfo=ET
    )
    return et_midnight.astimezone(UTC)
