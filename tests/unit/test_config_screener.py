"""ScreenerSettings + DEFAULT_UNIVERSE (IBK-95)."""

from __future__ import annotations

from optionsbot.config import Settings
from optionsbot.screener.universe import DEFAULT_UNIVERSE


def test_screener_settings_defaults() -> None:
    s = Settings()
    assert s.screener.universe is None
    assert s.screener.min_dollar_volume > 0
    assert s.screener.top_n >= 1


def test_default_universe_is_nonempty_unique_uppercase() -> None:
    assert len(DEFAULT_UNIVERSE) >= 50
    assert len(set(DEFAULT_UNIVERSE)) == len(DEFAULT_UNIVERSE)  # no dupes
    assert all(t == t.upper() and t.isalpha() for t in DEFAULT_UNIVERSE)
