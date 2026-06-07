"""Sign-aware edge_sort_key + has_positive_edge (IBK-106)."""

from __future__ import annotations

from optionsbot.scoring.composite import edge_sort_key, has_positive_edge
from optionsbot.strategies.base import StrategySuggestion


def _sug(expected_value: float | None, max_loss: float | None) -> StrategySuggestion:
    return StrategySuggestion(
        strategy_name="x", legs=(), credit_or_debit=1.0,
        max_loss=max_loss, max_profit=300.0, prob_profit=0.6,
        suggested_quantity=1, defined_risk=True, rationale="",
        reward_risk=None, expected_value=expected_value, risk_tier="balanced",
    )


def _rank(*sugs: StrategySuggestion) -> list[StrategySuggestion]:
    return sorted(sugs, key=edge_sort_key, reverse=True)


def test_positive_edge_ranks_above_negative() -> None:
    pos = _sug(5.0, 1000.0)        # EV +5 -> tier 2
    neg = _sug(-49.0, 737.0)       # EV -49 -> tier 1
    assert _rank(neg, pos) == [pos, neg]


def test_negative_group_orders_by_raw_ev_not_ratio() -> None:
    # The NVDA regression: the spread (EV -49, ml 737) must outrank the
    # capital-hungry CSP (EV -83, ml 19397) even though EV/max_loss rates the
    # CSP "less bad per dollar" (-0.0043 vs -0.0665).
    csp = _sug(-83.26, 19397.50)
    spread = _sug(-49.04, 737.50)
    assert _rank(csp, spread) == [spread, csp]


def test_positive_group_orders_by_ev_over_max_loss() -> None:
    # Within positive edge, IBK-104 capital-efficiency ordering is preserved.
    high = _sug(20.0, 100.0)       # edge 0.20
    low = _sug(20.0, 400.0)        # edge 0.05
    assert _rank(low, high) == [high, low]


def test_none_edge_sorts_last_even_when_positive_ev() -> None:
    naked = _sug(50.0, None)       # +EV but undefined risk -> edge None -> tier 0
    losing = _sug(-10.0, 500.0)    # tier 1
    winning = _sug(1.0, 500.0)     # tier 2
    assert _rank(naked, losing, winning) == [winning, losing, naked]


def test_has_positive_edge() -> None:
    assert has_positive_edge(_sug(5.0, 100.0)) is True
    assert has_positive_edge(_sug(0.0, 100.0)) is False     # break-even is not positive
    assert has_positive_edge(_sug(-5.0, 100.0)) is False
    assert has_positive_edge(_sug(5.0, None)) is False       # undefined risk
    assert has_positive_edge(_sug(None, 100.0)) is False     # non-modelable EV
