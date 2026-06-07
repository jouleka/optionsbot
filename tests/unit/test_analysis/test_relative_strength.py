"""Tests for relative strength vs benchmark (IBK-109)."""

from __future__ import annotations

import pandas as pd

from optionsbot.analysis.relative_strength import relative_strength


def _bars(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


def test_relative_strength_outperformance() -> None:
    sym = _bars([100, 101, 102, 103, 104, 110])    # 100 -> 110 = +10% over window 5
    bench = _bars([100, 100, 101, 101, 102, 102])  # 100 -> 102 = +2%
    rs = relative_strength(sym, bench, window=5)
    assert rs is not None
    assert round(rs, 4) == round(0.10 - 0.02, 4)


def test_relative_strength_underperformance_negative() -> None:
    sym = _bars([100, 100, 100, 100, 100, 98])     # -2%
    bench = _bars([100, 100, 100, 100, 100, 105])  # +5%
    rs = relative_strength(sym, bench, window=5)
    assert rs is not None and rs < 0


def test_relative_strength_none_when_too_few_bars() -> None:
    sym = _bars([100, 101, 102])                   # 3 bars, window 5 needs 6
    bench = _bars([100, 101, 102, 103, 104, 105])
    assert relative_strength(sym, bench, window=5) is None


def test_relative_strength_none_on_nonpositive_start() -> None:
    bench = _bars([100, 101, 102, 103, 104, 105])
    assert relative_strength(_bars([0, 1, 2, 3, 4, 5]), bench, window=5) is None
    assert relative_strength(_bars([-5, 1, 2, 3, 4, 5]), bench, window=5) is None
