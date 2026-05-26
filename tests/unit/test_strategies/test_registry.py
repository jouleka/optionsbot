"""Verify the strategy registry exposes all 16 strategies."""

from optionsbot.strategies import all_strategies, get_strategy

_EXPECTED_NAMES = {
    "iron_condor",
    "iron_butterfly",
    "bull_put_spread",
    "bear_call_spread",
    "bull_call_spread",
    "bear_put_spread",
    "long_straddle",
    "long_strangle",
    "short_straddle",
    "short_strangle",
    "calendar_spread",
    "diagonal_spread",
    "covered_call",
    "cash_secured_put",
    "long_call",
    "long_put",
}


def test_registry_has_all_16_strategies() -> None:
    names = {s.name for s in all_strategies()}
    assert names == _EXPECTED_NAMES


def test_get_strategy_returns_correct_instance() -> None:
    s = get_strategy("iron_condor")
    assert s.name == "iron_condor"


def test_all_factor_weights_sum_to_one() -> None:
    for strategy in all_strategies():
        s = sum(strategy.factor_weights.values())
        assert abs(s - 1.0) < 1e-6, f"{strategy.name} weights sum to {s}"


def test_undefined_risk_strategies_are_flagged() -> None:
    undefined = {s.name for s in all_strategies() if not s.defined_risk}
    assert undefined == {"short_straddle", "short_strangle"}
