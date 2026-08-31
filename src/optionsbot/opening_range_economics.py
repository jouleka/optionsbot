"""Economics for the explicitly managed opening-range/FVG trade plan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from optionsbot.strategies import StrategySuggestion


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def managed_expected_value(
    *,
    credit_or_debit: object,
    target_hit_probability: object,
    plan: object,
    estimated_round_trip_cost: object = 0.0,
    maximum_profit: object = None,
) -> float | None:
    """Return expectancy at the configured premium stop/target boundaries.

    ``target_hit_probability`` must come from a versioned, out-of-sample model
    of the *managed option path*: target reached before stop/timeout.  A generic
    probability of terminal profit is deliberately not accepted here; expiry
    profitability and first passage through intraday premium boundaries are
    different events.

    This is the conservative binary target/stop form.  Timeout observations
    must be incorporated by the calibrated model before it supplies the target
    probability.  Finite-payoff structures also fail closed when the desired
    net target cannot fit below their maximum profit.
    """
    if not isinstance(plan, Mapping):
        return None
    if plan.get("status") != "entry_confirmed" or plan.get("source") != "trusted_daemon":
        return None
    cashflow = _number(credit_or_debit)
    win_probability = _number(target_hit_probability)
    stop_pct = _number(plan.get("stop_pct"))
    target_r = _number(plan.get("target_r"))
    target_pct = _number(plan.get("target_pct"))
    round_trip_cost = _number(estimated_round_trip_cost)
    finite_maximum_profit = _number(maximum_profit)
    if None in (
        cashflow,
        win_probability,
        stop_pct,
        target_r,
        target_pct,
        round_trip_cost,
    ):
        return None
    assert cashflow is not None
    assert win_probability is not None
    assert stop_pct is not None
    assert target_r is not None
    assert target_pct is not None
    assert round_trip_cost is not None
    # The approved ORB structures are long calls/puts and debit verticals.
    if (
        cashflow >= 0
        or not 0 < win_probability < 1
        or not 0 < stop_pct < 1
        or round_trip_cost < 0
    ):
        return None
    if target_r < 1 or not math.isclose(target_pct, stop_pct * target_r, rel_tol=1e-9):
        return None
    debit = abs(cashflow)
    target_dollars = debit * target_pct
    stop_dollars = debit * stop_pct
    if finite_maximum_profit is not None and (
        finite_maximum_profit <= 0
        or target_dollars + round_trip_cost > finite_maximum_profit
    ):
        return None
    gross_expectancy = (
        win_probability * target_dollars - (1 - win_probability) * stop_dollars
    )
    return gross_expectancy - round_trip_cost


def managed_path_expected_values(
    *,
    credit_or_debit: object,
    target_probability: object,
    stop_probability: object,
    timeout_probability: object,
    timeout_expected_return: object,
    ev_residual_return_q05: object,
    plan: object,
    estimated_round_trip_cost: object = 0.0,
    maximum_profit: object = None,
) -> tuple[float, float] | None:
    """Return point and conservative EV for target/stop/timeout paths.

    All returns use one fresh entry basis. The residual is a dimensionless
    out-of-fold error quantile and therefore scales with the candidate's
    current debit rather than leaking the dollar size of historical samples.
    """
    if not isinstance(plan, Mapping):
        return None
    if plan.get("status") != "entry_confirmed" or plan.get("source") != "trusted_daemon":
        return None
    cashflow = _number(credit_or_debit)
    target = _number(target_probability)
    stop = _number(stop_probability)
    timeout = _number(timeout_probability)
    timeout_return = _number(timeout_expected_return)
    residual_return = _number(ev_residual_return_q05)
    stop_pct = _number(plan.get("stop_pct"))
    target_pct = _number(plan.get("target_pct"))
    round_trip_cost = _number(estimated_round_trip_cost)
    finite_maximum_profit = _number(maximum_profit)
    if None in (
        cashflow,
        target,
        stop,
        timeout,
        timeout_return,
        residual_return,
        stop_pct,
        target_pct,
        round_trip_cost,
    ):
        return None
    assert cashflow is not None
    assert target is not None and stop is not None and timeout is not None
    assert timeout_return is not None and residual_return is not None
    assert stop_pct is not None and target_pct is not None
    assert round_trip_cost is not None
    probabilities = (target, stop, timeout)
    if (
        cashflow >= 0.0
        or any(probability < 0.0 or probability > 1.0 for probability in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        or not 0.0 < stop_pct < 1.0
        or target_pct <= 0.0
        or round_trip_cost < 0.0
    ):
        return None
    basis = abs(cashflow)
    target_gain = basis * target_pct
    stop_loss = basis * stop_pct
    if finite_maximum_profit is not None and (
        finite_maximum_profit <= 0.0
        or target_gain + round_trip_cost > finite_maximum_profit
    ):
        return None
    point = (
        target * target_gain
        - stop * stop_loss
        + timeout * timeout_return * basis
        - round_trip_cost
    )
    return point, point + residual_return * basis


def managed_break_even_probability(
    *,
    credit_or_debit: object,
    plan: object,
    estimated_round_trip_cost: object = 0.0,
    maximum_profit: object = None,
) -> float | None:
    """Return the target-first probability required to break even after costs."""
    if not isinstance(plan, Mapping):
        return None
    if plan.get("status") != "entry_confirmed" or plan.get("source") != "trusted_daemon":
        return None
    cashflow = _number(credit_or_debit)
    stop_pct = _number(plan.get("stop_pct"))
    target_r = _number(plan.get("target_r"))
    target_pct = _number(plan.get("target_pct"))
    round_trip_cost = _number(estimated_round_trip_cost)
    finite_maximum_profit = _number(maximum_profit)
    if None in (cashflow, stop_pct, target_r, target_pct, round_trip_cost):
        return None
    assert cashflow is not None
    assert stop_pct is not None
    assert target_r is not None
    assert target_pct is not None
    assert round_trip_cost is not None
    if cashflow >= 0 or not 0 < stop_pct < 1 or round_trip_cost < 0:
        return None
    if target_r < 1 or not math.isclose(target_pct, stop_pct * target_r, rel_tol=1e-9):
        return None
    debit = abs(cashflow)
    target_dollars = debit * target_pct
    stop_dollars = debit * stop_pct
    if finite_maximum_profit is not None and (
        finite_maximum_profit <= 0
        or target_dollars + round_trip_cost > finite_maximum_profit
    ):
        return None
    return (stop_dollars + round_trip_cost) / (target_dollars + stop_dollars)


def estimated_round_trip_cost(
    *,
    option_contracts_per_unit: object,
    combo_spread_per_share: object,
    commission_per_contract: object,
    slippage_spread_fraction: object,
) -> float | None:
    """Conservative one-unit entry+exit transaction-cost reserve in dollars."""
    contracts = _number(option_contracts_per_unit)
    combo_spread = _number(combo_spread_per_share)
    commission = _number(commission_per_contract)
    slippage_fraction = _number(slippage_spread_fraction)
    if None in (contracts, combo_spread, commission, slippage_fraction):
        return None
    assert contracts is not None
    assert combo_spread is not None
    assert commission is not None
    assert slippage_fraction is not None
    if (
        contracts <= 0
        or not contracts.is_integer()
        or combo_spread < 0
        or commission < 0
        or slippage_fraction < 0
    ):
        return None
    round_trip_commissions = contracts * 2.0 * commission
    round_trip_slippage = combo_spread * 100.0 * slippage_fraction
    return round_trip_commissions + round_trip_slippage


def with_managed_expected_value(
    suggestion: StrategySuggestion,
    plan: object,
    *,
    target_hit_probability: object = None,
    estimated_round_trip_cost: object = 0.0,
) -> StrategySuggestion:
    """Apply managed ORB expectancy to an immutable strategy suggestion."""
    expected_value = managed_expected_value(
        credit_or_debit=suggestion.credit_or_debit,
        target_hit_probability=target_hit_probability,
        plan=plan,
        estimated_round_trip_cost=estimated_round_trip_cost,
        maximum_profit=suggestion.max_profit,
    )
    if not isinstance(plan, Mapping) or plan.get("status") != "entry_confirmed":
        return suggestion
    return replace(suggestion, expected_value=expected_value)
