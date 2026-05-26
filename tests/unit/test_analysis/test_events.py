"""Tests for earnings-window detection."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from optionsbot.analysis.events import earnings_within, next_earnings
from optionsbot.analysis.types import EarningsInfo


def test_manual_override_wins_over_yfinance() -> None:
    manual = {"AAPL": date.today() + timedelta(days=5)}
    with patch("optionsbot.analysis.events.yf") as mock_yf:
        # If we called yfinance we'd fail the test (the mock is unset).
        info = next_earnings("AAPL", manual_overrides=manual)
        mock_yf.Ticker.assert_not_called()
    assert isinstance(info, EarningsInfo)
    assert info.next_date == manual["AAPL"]
    assert info.source == "manual"


def test_yfinance_calendar_returns_next_date() -> None:
    future = date.today() + timedelta(days=12)
    mock_ticker = MagicMock()
    mock_ticker.calendar = {"Earnings Date": [future]}
    with patch("optionsbot.analysis.events.yf") as mock_yf:
        mock_yf.Ticker.return_value = mock_ticker
        info = next_earnings("AAPL")
    assert info.next_date == future
    assert info.source == "yfinance"


def test_yfinance_missing_returns_unknown() -> None:
    mock_ticker = MagicMock()
    mock_ticker.calendar = None
    with patch("optionsbot.analysis.events.yf") as mock_yf:
        mock_yf.Ticker.return_value = mock_ticker
        info = next_earnings("ZZZNOTREAL")
    assert info.next_date is None
    assert info.source == "unknown"


def test_yfinance_exception_returns_unknown() -> None:
    with patch("optionsbot.analysis.events.yf") as mock_yf:
        mock_yf.Ticker.side_effect = Exception("network down")
        info = next_earnings("AAPL")
    assert info.next_date is None
    assert info.source == "unknown"


def test_earnings_within_true_when_inside_window() -> None:
    manual = {"AAPL": date.today() + timedelta(days=10)}
    assert earnings_within("AAPL", days=14, manual_overrides=manual) is True


def test_earnings_within_false_when_outside_window() -> None:
    manual = {"AAPL": date.today() + timedelta(days=30)}
    assert earnings_within("AAPL", days=14, manual_overrides=manual) is False


def test_earnings_within_false_when_unknown() -> None:
    with patch("optionsbot.analysis.events.yf") as mock_yf:
        mock_yf.Ticker.return_value = MagicMock(calendar=None)
        assert earnings_within("XYZ", days=14) is False


def test_earnings_within_past_dates_treated_as_no_upcoming() -> None:
    manual = {"AAPL": date.today() - timedelta(days=5)}  # in the past
    assert earnings_within("AAPL", days=14, manual_overrides=manual) is False
