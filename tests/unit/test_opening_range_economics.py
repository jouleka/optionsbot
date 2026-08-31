"""Managed expectancy for the exact ORB/FVG stop/target plan."""

import pytest

from optionsbot.opening_range_economics import (
    estimated_round_trip_cost,
    managed_break_even_probability,
    managed_expected_value,
    managed_path_expected_values,
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
        target_hit_probability=0.564424975189262,
        plan=_plan(),
    )

    assert value is not None and value > 0


def test_managed_ev_requires_probability_above_plan_break_even() -> None:
    assert (
        managed_expected_value(
            credit_or_debit=-270.0,
            target_hit_probability=0.39,
            plan=_plan(),
        )
        is not None
    )
    assert managed_expected_value(
        credit_or_debit=-270.0,
        target_hit_probability=0.39,
        plan=_plan(),
    ) < 0


def test_managed_ev_does_not_apply_to_credit_or_untrusted_plans() -> None:
    assert (
        managed_expected_value(
            credit_or_debit=100.0,
            target_hit_probability=0.70,
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
        target_hit_probability=0.34827346286975674,
        plan=_plan(target_r=2.0),
    )
    after_costs = managed_expected_value(
        credit_or_debit=-69.5,
        target_hit_probability=0.34827346286975674,
        plan=_plan(target_r=2.0),
        estimated_round_trip_cost=cost,
    )

    assert gross == pytest.approx(0.4672525512516428)
    assert after_costs == pytest.approx(-11.332747448748358)
    untrusted = {**_plan(), "source": "external"}
    assert (
        managed_expected_value(
            credit_or_debit=-100.0,
            target_hit_probability=0.70,
            plan=untrusted,
        )
        is None
    )


def test_break_even_probability_includes_costs() -> None:
    assert managed_break_even_probability(
        credit_or_debit=-100.0,
        plan=_plan(),
    ) == pytest.approx(0.4)
    assert managed_break_even_probability(
        credit_or_debit=-100.0,
        plan=_plan(),
        estimated_round_trip_cost=10.0,
    ) == pytest.approx(25.0 / 37.5)


def test_finite_spread_with_unreachable_net_target_fails_closed() -> None:
    assert managed_expected_value(
        credit_or_debit=-82.0,
        target_hit_probability=0.70,
        plan=_plan(),
        estimated_round_trip_cost=6.80,
        maximum_profit=18.0,
    ) is None
    assert managed_break_even_probability(
        credit_or_debit=-82.0,
        plan=_plan(),
        estimated_round_trip_cost=6.80,
        maximum_profit=18.0,
    ) is None


def test_three_event_expected_value_includes_timeout_costs_and_scaled_lcb() -> None:
    values = managed_path_expected_values(
        credit_or_debit=-100.0,
        target_probability=0.50,
        stop_probability=0.30,
        timeout_probability=0.20,
        timeout_expected_return=-0.02,
        ev_residual_return_q05=-0.03,
        plan={
            "status": "entry_confirmed",
            "source": "trusted_daemon",
            "stop_pct": 0.15,
            "target_pct": 0.30,
        },
        estimated_round_trip_cost=1.40,
    )
    assert values is not None
    point, lower = values
    assert point == pytest.approx(8.7)
    assert lower == pytest.approx(5.7)
