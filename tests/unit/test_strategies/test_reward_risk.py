"""build_suggestion fills reward_risk = max_profit / max_loss (IBK-93)."""

from __future__ import annotations

from optionsbot.strategies.base import StrategySuggestion


def test_reward_risk_field_defaults_none() -> None:
    s = StrategySuggestion(
        strategy_name="x", legs=(), credit_or_debit=0.0, max_loss=None,
        max_profit=None, prob_profit=None, suggested_quantity=0,
        defined_risk=True, rationale="",
    )
    assert s.reward_risk is None


def test_reward_risk_field_accepts_value() -> None:
    s = StrategySuggestion(
        strategy_name="x", legs=(), credit_or_debit=0.0, max_loss=200.0,
        max_profit=300.0, prob_profit=0.5, suggested_quantity=1,
        defined_risk=True, rationale="", reward_risk=1.5,
    )
    assert s.reward_risk == 1.5
