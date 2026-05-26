"""Tests for the Strategy ABC and supporting dataclasses."""

from __future__ import annotations

import pytest

from optionsbot.strategies.base import Leg, Strategy


def test_leg_defaults_for_option() -> None:
    leg = Leg(symbol="SPY", side="buy", expiry="20260619", strike=400.0, right="C")
    assert leg.sec_type == "OPT"
    assert leg.quantity == 1


def test_leg_can_represent_stock_leg() -> None:
    leg = Leg(symbol="SPY", side="buy", sec_type="STK", quantity=100)
    assert leg.expiry is None
    assert leg.strike is None
    assert leg.right is None


def test_strategy_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_suggest_size_returns_zero_for_undefined_risk() -> None:
    from optionsbot.strategies.iron_condor import IronCondor

    ic = IronCondor()
    assert ic.suggest_size(account_value=10_000, max_loss_per_contract=None) == 0


def test_suggest_size_returns_zero_for_zero_loss() -> None:
    from optionsbot.strategies.iron_condor import IronCondor

    ic = IronCondor()
    assert ic.suggest_size(account_value=10_000, max_loss_per_contract=0) == 0


def test_suggest_size_floors_at_one() -> None:
    from optionsbot.strategies.iron_condor import IronCondor

    ic = IronCondor()
    # budget = 10000 * 0.02 = 200; max_loss=500 -> 200//500 = 0; floor to 1
    assert ic.suggest_size(account_value=10_000, max_loss_per_contract=500) == 1


def test_suggest_size_scales_with_budget() -> None:
    from optionsbot.strategies.iron_condor import IronCondor

    ic = IronCondor()
    # budget = 100_000 * 0.02 = 2000; max_loss=400 -> 2000//400 = 5
    assert ic.suggest_size(account_value=100_000, max_loss_per_contract=400) == 5
