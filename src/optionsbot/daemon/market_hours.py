"""Market-hours gate for the daemon's scan scheduler.

Uses ``pandas_market_calendars``'s NYSE calendar as the source of truth for
trading days, half-days, and holidays. Equity options follow equity hours.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")

# Load once at import; the calendar object is reusable across calls.
_NYSE = mcal.get_calendar("NYSE")


def is_market_open(now: datetime) -> bool:
    """Return True iff ``now`` falls within an NYSE trading session.

    ``now`` MUST be timezone-aware. Naive datetimes are rejected to avoid
    silent UTC/ET confusion. The function normalizes to ET only to
    determine the calendar date for the schedule lookup; the open/close
    comparison itself uses the caller-supplied ``now`` (Python resolves
    tz offsets transparently). This correctly handles half-days such as
    the day after Thanksgiving, which closes 13:00 ET.
    """
    if now.tzinfo is None:
        raise ValueError("is_market_open requires a tz-aware datetime")
    et_now = now.astimezone(ET)
    et_date = et_now.date()
    schedule = _NYSE.schedule(start_date=et_date, end_date=et_date)
    if schedule.empty:
        return False
    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()
    # schedule() returns a DataFrame typed as Any; explicitly cast to bool
    # so mypy's no-any-return check is satisfied.
    return bool(market_open <= now <= market_close)


def nyse_session_date(now: datetime) -> date:
    """The America/New_York calendar date ``now`` belongs to.

    This is the right anchor for "today's" trading bucket: the daily-loss
    window must roll at ET midnight, NOT UTC midnight, or late-session losses
    after ~20:00 ET (when the UTC date has already advanced) land in the wrong
    day — and the ET/UTC offset itself shifts an hour across DST transitions.
    """
    if now.tzinfo is None:
        raise ValueError("nyse_session_date requires a tz-aware datetime")
    return now.astimezone(ET).date()


def nyse_session_start_utc(now: datetime) -> datetime:
    """UTC instant of ET midnight that opens ``now``'s NYSE session date.

    Returned tz-aware in UTC so it compares directly against UTC-stored
    ``closed_ts`` values. DST-correct: ET midnight is -04:00 in summer (EDT)
    and -05:00 in winter (EST), and ``astimezone`` resolves the offset.
    """
    session = nyse_session_date(now)
    et_midnight = datetime(
        session.year, session.month, session.day, 0, 0, tzinfo=ET
    )
    return et_midnight.astimezone(UTC)
