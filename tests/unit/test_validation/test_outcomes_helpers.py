"""Tests for the outcomes-loop helpers: make_close_fetcher + report_to_dict (IBK-117)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd

from optionsbot.validation.outcomes import make_close_fetcher, report_to_dict
from optionsbot.validation.types import OutcomeGroup, OutcomesReport


async def test_make_close_fetcher_finds_close_on_or_before_expiry() -> None:
    history = AsyncMock()
    idx = pd.to_datetime(["2026-06-15", "2026-06-16", "2026-06-17"])
    history.get_history.return_value = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=idx)
    fetch = make_close_fetcher(history)
    assert await fetch("SPY", "20260617") == 12.0
    # expiry on a non-trading day (Sat 2026-06-20) -> backscan to 2026-06-17
    assert await fetch("SPY", "20260620") == 12.0


async def test_make_close_fetcher_none_when_no_data() -> None:
    history = AsyncMock()
    history.get_history.return_value = pd.DataFrame({"close": []}, index=pd.to_datetime([]))
    assert await make_close_fetcher(history)("SPY", "20260617") is None


def test_report_to_dict_shape() -> None:
    g = OutcomeGroup("overall", 3, 0.667, 0.6, 150.0, 50.0)
    report = OutcomesReport(
        overall=g, by_strategy={"bull_put_spread": g}, by_risk_tier={"balanced": g}
    )
    d = report_to_dict(report)
    assert d["overall"]["count"] == 3 and d["overall"]["win_rate"] == 0.667
    assert d["by_strategy"]["bull_put_spread"]["avg_pnl"] == 50.0
    assert d["by_risk_tier"]["balanced"]["mean_pred_pop"] == 0.6
