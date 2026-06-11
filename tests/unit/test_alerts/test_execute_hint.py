"""Alert execute-hint rendering (IBK-126). Lookup test lives in test_daemon."""

from __future__ import annotations

from datetime import UTC, datetime

from optionsbot.alerts.formatter import format_alert_markdown
from optionsbot.analysis.types import MarketView
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.strategies.base import Leg, StrategySuggestion

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)


def _scored() -> ScoredStrategy:
    suggestion = StrategySuggestion(
        strategy_name="bull_put_spread",
        legs=(Leg(symbol="SPY", side="sell", expiry="20260717", strike=580.0, right="P"),),
        credit_or_debit=120.0, max_loss=380.0, max_profit=120.0, prob_profit=0.7,
        suggested_quantity=1, defined_risk=True, rationale="t",
    )
    return ScoredStrategy(
        "bull_put_spread", 78.0, FactorBreakdown(.5, .5, .5, .5, .5, .5),
        suggestion, "t",
    )


def test_formatter_appends_execute_hint() -> None:
    view = MarketView("neutral", "weak", "high", 0.7, False, False)
    with_hint = format_alert_markdown("SPY", view, _scored(), NOW, execute_hint="/execute 42")
    without = format_alert_markdown("SPY", view, _scored(), NOW)
    assert "/execute 42" in with_hint
    assert "/execute" not in without
