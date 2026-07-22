"""Earnings-date detection via yfinance with optional manual overrides.

yfinance is the v1 default because it's free and zero-config. It's also
occasionally stale or missing for individual tickers; the manual override
path lets callers supply a known next-earnings date that wins over yfinance.

The shape of ``yfinance.Ticker(symbol).calendar`` has shifted across
versions. Recent versions (1.4.0+) return a ``dict`` keyed by field name
with ``"Earnings Date"`` mapping to a ``list[date]``. Older versions
returned a pandas DataFrame with an ``"Earnings Date"`` row. This module
handles both shapes for forward/backward compatibility.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import yfinance as yf  # type: ignore[import-untyped]

from optionsbot.analysis.types import EarningsInfo

_NO_EARNINGS_ETFS = frozenset(
    {
        "DIA",
        "GLD",
        "IWM",
        "QQQ",
        "SLV",
        "SPY",
        "TLT",
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    }
)


def _coerce_to_date(value: Any) -> date | None:
    """Best-effort conversion of a yfinance value to ``datetime.date``.

    Handles ``None``, ``date``, ``datetime``, and pandas ``Timestamp``
    (which exposes a ``.date()`` method).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return value.date()  # type: ignore[no-any-return]
    except AttributeError:
        return None


def _extract_next_date(calendar: Any) -> date | None:
    """Pull the next earnings date out of a yfinance ``calendar`` payload.

    Returns ``None`` if no date can be located. Tolerates both the
    newer dict shape and the legacy DataFrame shape.
    """
    if isinstance(calendar, dict):
        raw = calendar.get("Earnings Date")
        if isinstance(raw, list):
            if not raw:
                return None
            return _coerce_to_date(raw[0])
        return _coerce_to_date(raw)
    # Fall back to the legacy DataFrame layout: row "Earnings Date" with
    # dated columns. We index defensively.
    try:
        raw = calendar.loc["Earnings Date"][0]
    except (KeyError, IndexError, AttributeError, TypeError):
        return None
    return _coerce_to_date(raw)


def next_earnings(
    symbol: str,
    manual_overrides: dict[str, date] | None = None,
) -> EarningsInfo:
    """Return the next earnings date for ``symbol``.

    Resolution order:
      1. ``manual_overrides[symbol]`` if present -- ``source="manual"``.
         yfinance is NOT consulted in this branch.
      2. ``yfinance.Ticker(symbol).calendar`` -- ``source="yfinance"``.
      3. Anything else (network error, missing calendar, unparseable
         payload) -- ``source="unknown"`` with ``next_date=None``.
    """
    if manual_overrides is not None and symbol in manual_overrides:
        return EarningsInfo(
            next_date=manual_overrides[symbol], source="manual"
        )
    # Broad/index/sector ETFs do not report corporate earnings. Asking Yahoo
    # for their earnings calendar emits a misleading ERROR-level 404 on every
    # scan even though the scan correctly continues. Skip that inapplicable
    # request so operational error logs retain signal.
    if symbol.upper() in _NO_EARNINGS_ETFS:
        return EarningsInfo(next_date=None, source="unknown")
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
    except Exception:  # noqa: BLE001 -- yfinance raises a variety of network/parse errors
        return EarningsInfo(next_date=None, source="unknown")
    if calendar is None:
        return EarningsInfo(next_date=None, source="unknown")
    next_date = _extract_next_date(calendar)
    if next_date is None:
        return EarningsInfo(next_date=None, source="unknown")
    return EarningsInfo(next_date=next_date, source="yfinance")


def earnings_within(
    symbol: str,
    days: int,
    manual_overrides: dict[str, date] | None = None,
    today: date | None = None,
) -> bool:
    """True if ``symbol`` has earnings within ``days`` of ``today``.

    ``today`` defaults to ``date.today()`` but can be injected for tests.
    Past earnings dates are treated as "no upcoming earnings in window"
    and return False.
    """
    info = next_earnings(symbol, manual_overrides=manual_overrides)
    if info.next_date is None:
        return False
    reference = today if today is not None else date.today()
    delta = (info.next_date - reference).days
    return 0 <= delta <= days
