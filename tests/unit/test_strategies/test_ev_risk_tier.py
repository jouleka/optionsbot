"""Expected-value + risk-tier helpers (IBK-93 Phase B)."""

from __future__ import annotations

from optionsbot.strategies.base import _expected_value, _risk_tier


def test_expected_value_defined_risk() -> None:
    # 60% win of +300, 40% loss of -200 => 0.6*300 - 0.4*200 = 100.0
    assert _expected_value(0.6, 300.0, 200.0, defined_risk=True) == 100.0


def test_expected_value_none_when_unbounded_or_missing() -> None:
    assert _expected_value(0.6, None, 200.0, defined_risk=True) is None  # unbounded profit
    assert _expected_value(None, 300.0, 200.0, defined_risk=True) is None  # no prob
    assert _expected_value(0.6, 300.0, None, defined_risk=True) is None  # no max loss
    assert _expected_value(0.6, 300.0, 200.0, defined_risk=False) is None  # undefined risk


def test_risk_tier_conservative() -> None:
    assert _risk_tier(defined_risk=True, prob_profit=0.70) == "conservative"
    assert _risk_tier(defined_risk=True, prob_profit=0.65) == "conservative"  # boundary inclusive


def test_risk_tier_aggressive() -> None:
    assert _risk_tier(defined_risk=False, prob_profit=0.90) == "aggressive"  # undefined risk
    assert _risk_tier(defined_risk=True, prob_profit=0.30) == "aggressive"  # low prob


def test_risk_tier_balanced() -> None:
    assert _risk_tier(defined_risk=True, prob_profit=0.50) == "balanced"
    assert _risk_tier(defined_risk=True, prob_profit=None) == "balanced"  # unknown prob, known risk
