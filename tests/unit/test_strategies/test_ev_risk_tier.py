"""Expected-value + risk-tier helpers (IBK-93 Phase B)."""

from __future__ import annotations

from optionsbot.strategies.base import _risk_tier


def test_risk_tier_conservative() -> None:
    assert _risk_tier(defined_risk=True, prob_profit=0.70) == "conservative"
    assert _risk_tier(defined_risk=True, prob_profit=0.65) == "conservative"  # boundary inclusive


def test_risk_tier_aggressive() -> None:
    assert _risk_tier(defined_risk=False, prob_profit=0.90) == "aggressive"  # undefined risk
    assert _risk_tier(defined_risk=True, prob_profit=0.30) == "aggressive"  # low prob


def test_risk_tier_balanced() -> None:
    assert _risk_tier(defined_risk=True, prob_profit=0.50) == "balanced"
    assert _risk_tier(defined_risk=True, prob_profit=None) == "balanced"  # unknown prob, known risk
