"""Tests for the daily_brief MCP tool (IBK-107)."""

from __future__ import annotations

from optionsbot.mcp_server.tools.daily_brief import _edge_tier, _reconstruct_suggestion


def test_reconstruct_suggestion_enables_canonical_edge() -> None:
    sj = {
        "expected_value": -49.0, "max_loss": 737.0, "credit_or_debit": 2.6,
        "max_profit": 263.0, "prob_profit": 0.67, "suggested_quantity": 1,
        "defined_risk": True, "reward_risk": 0.36, "risk_tier": "balanced",
    }
    sug = _reconstruct_suggestion(sj, "bull_put_spread", "ok")
    assert sug.expected_value == -49.0
    assert sug.max_loss == 737.0
    # The canonical property recomputes from the reconstructed fields:
    assert sug.risk_normalized_expectancy == -49.0 / 737.0
    assert _edge_tier(sug) == "negative"


def test_edge_tier_mapping() -> None:
    positive = _reconstruct_suggestion({"expected_value": 5.0, "max_loss": 100.0}, "x", "")
    none_edge = _reconstruct_suggestion({"expected_value": 5.0, "max_loss": None}, "x", "")
    break_even = _reconstruct_suggestion({"expected_value": 0.0, "max_loss": 100.0}, "x", "")
    assert _edge_tier(positive) == "positive"
    assert _edge_tier(none_edge) == "undefined"
    assert _edge_tier(break_even) == "negative"
