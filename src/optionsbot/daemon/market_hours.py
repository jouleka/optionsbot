"""Market-hours gate for the daemon's scan scheduler.

Uses ``pandas_market_calendars``'s NYSE calendar as the source of truth for
trading days, half-days, and holidays. Equity options follow equity hours.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")

# Load once at import; the calendar object is reusable across calls.
_NYSE = mcal.get_calendar("NYSE")


def is_market_open(now: datetime) -> bool:
    """Return True iff ``now`` falls within an NYSE trading session.

    ``now`` MUST be timezone-aware. Naive datetimes are rejected to avoid
    silent UTC/ET confusion. The function normalizes to ET before
    comparing against the trading day's open/close, which correctly
    handles half-days (e.g., day after Thanksgiving closes 13:00 ET).
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
