"""StrategySuggestion.risk_normalized_expectancy (IBK-104)."""

from __future__ import annotations

from optionsbot.strategies.base import StrategySuggestion


def _sug(expected_value: float | None, max_loss: float | None) -> StrategySuggestion:
    return StrategySuggestion(
        strategy_name="x", legs=(), credit_or_debit=1.0,
        max_loss=max_loss, max_profit=300.0, prob_profit=0.6,
        suggested_quantity=1, defined_risk=True, rationale="",
        reward_risk=None, expected_value=expected_value, risk_tier="balanced",
    )


def test_risk_normalized_expectancy_is_ev_over_max_loss() -> None:
    assert _sug(50.0, 200.0).risk_normalized_expectancy == 0.25
    assert _sug(-40.0, 800.0).risk_normalized_expectancy == -0.05


def test_risk_normalized_expectancy_none_when_inputs_missing() -> None:
    assert _sug(None, 200.0).risk_normalized_expectancy is None      # no EV
    assert _sug(50.0, None).risk_normalized_expectancy is None        # undefined risk
    assert _sug(50.0, 0.0).risk_normalized_expectancy is None         # zero max_loss
    assert _sug(50.0, -10.0).risk_normalized_expectancy is None       # guard
