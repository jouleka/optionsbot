"""Managed expectancy for the exact ORB/FVG stop/target plan."""

import pytest

from optionsbot.opening_range_economics import (
    estimated_round_trip_cost,
    managed_expected_value,
)


def _plan(*, target_r: float = 1.5) -> dict[str, object]:
    return {
        "status": "entry_confirmed",
        "source": "trusted_daemon",
        "stop_pct": 0.15,
        "target_r": target_r,
        "target_pct": 0.15 * target_r,
    }


def test_managed_ev_uses_the_actual_stop_and_target() -> None:
    value = managed_expected_value(
        credit_or_debit=-270.0,
        prob_profit=0.564424975189262,
        plan=_plan(),
    )

    assert value is not None and value > 0


def test_managed_ev_requires_probability_above_plan_break_even() -> None:
    assert (
        managed_expected_value(
            credit_or_debit=-270.0,
            prob_profit=0.39,
            plan=_plan(),
        )
        is not None
    )
    assert managed_expected_value(
        credit_or_debit=-270.0,
        prob_profit=0.39,
        plan=_plan(),
    ) < 0


def test_managed_ev_does_not_apply_to_credit_or_untrusted_plans() -> None:
    assert (
        managed_expected_value(
            credit_or_debit=100.0,
            prob_profit=0.70,
            plan=_plan(),
        )
        is None
    )


def test_cost_adjusted_ev_rejects_august_10_googl_marginal_edge() -> None:
    """The live two-leg GOOGL candidate was +$0.47 gross but not after costs."""
    cost = estimated_round_trip_cost(
        option_contracts_per_unit=2,
        combo_spread_per_share=0.09,
        commission_per_contract=0.70,
        slippage_spread_fraction=1.0,
    )
    assert cost == pytest.approx(11.80)

    gross = managed_expected_value(
        credit_or_debit=-69.5,
        prob_profit=0.34827346286975674,
        plan=_plan(target_r=2.0),
    )
    after_costs = managed_expected_value(
        credit_or_debit=-69.5,
        prob_profit=0.34827346286975674,
        plan=_plan(target_r=2.0),
        estimated_round_trip_cost=cost,
    )

    assert gross == pytest.approx(0.4672525512516428)
    assert after_costs == pytest.approx(-11.332747448748358)
    untrusted = {**_plan(), "source": "external"}
    assert (
        managed_expected_value(
            credit_or_debit=-100.0,
            prob_profit=0.70,
            plan=untrusted,
        )
        is None
    )
