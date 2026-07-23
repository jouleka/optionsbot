"""Day-aware exact-0DTE universe selection."""

from datetime import date

from optionsbot.screener.universe import zero_dte_universe_for_session

UNIVERSE = [
    "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AMD", "MU", "GLD", "IBIT",
    "SMH", "XLF", "UNG", "USO", "COIN",
]


def test_tuesday_and_thursday_only_keep_daily_expirations() -> None:
    assert zero_dte_universe_for_session(UNIVERSE, date(2026, 7, 21)) == (
        "SPY", "QQQ", "IWM",
    )
    assert zero_dte_universe_for_session(UNIVERSE, date(2026, 7, 23)) == (
        "SPY", "QQQ", "IWM",
    )


def test_monday_adds_eligible_single_names_and_etps() -> None:
    assert zero_dte_universe_for_session(UNIVERSE, date(2026, 7, 20)) == (
        "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AMD", "MU", "GLD", "IBIT",
        "SMH", "XLF",
    )


def test_wednesday_uses_its_eligible_asset_set() -> None:
    assert zero_dte_universe_for_session(UNIVERSE, date(2026, 7, 22)) == (
        "SPY", "QQQ", "IWM", "NVDA", "TSLA", "GLD", "IBIT", "UNG", "USO",
    )


def test_friday_or_shifted_end_of_week_keeps_configured_scope() -> None:
    expected = tuple(UNIVERSE)
    assert zero_dte_universe_for_session(UNIVERSE, date(2026, 7, 24)) == expected
    assert zero_dte_universe_for_session(
        UNIVERSE,
        date(2026, 7, 23),
        end_of_week_expiry=True,
    ) == expected
