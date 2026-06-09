"""Tests for the plain-text track-record formatter (IBK-117)."""

from __future__ import annotations

from optionsbot.alerts.formatter import format_track_record
from optionsbot.validation.types import OutcomeGroup, OutcomesReport


def _report(count: int = 23) -> OutcomesReport:
    overall = OutcomeGroup("overall", count, 0.61, 0.58, 1240.0, 54.0)
    bs = OutcomeGroup("bull_put_spread", 9, 0.67, 0.60, 639.0, 71.0)
    return OutcomesReport(
        overall=overall, by_strategy={"bull_put_spread": bs}, by_risk_tier={"conservative": bs}
    )


def test_format_track_record_renders() -> None:
    out = format_track_record(_report())
    assert "track record" in out and "23 picks" in out and "win 0.61" in out
    assert "predicted 0.58" in out and "by strategy:" in out and "bull_put_spread" in out


def test_format_track_record_empty() -> None:
    empty = OutcomesReport(OutcomeGroup("overall", 0, 0.0, 0.0, 0.0, 0.0), {}, {})
    assert "no evaluated outcomes yet" in format_track_record(empty)
