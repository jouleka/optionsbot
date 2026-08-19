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
    prob_profit: object,
    plan: object,
    estimated_round_trip_cost: object = 0.0,
) -> float | None:
    """Return expectancy at the configured premium stop/target boundaries.

    The generic strategy model estimates terminal-expiry value. Exact ORB/FVG
    entries are instead closed at a premium-percent stop or R target, so their
    admission metric must use those same outcomes. ``prob_profit`` remains the
    bot's bounded probability estimate and is treated as the target-hit proxy;
    both terminal EV and this managed EV are retained downstream for learning.
    """
    if not isinstance(plan, Mapping):
        return None
    if plan.get("status") != "entry_confirmed" or plan.get("source") != "trusted_daemon":
        return None
    cashflow = _number(credit_or_debit)
    win_probability = _number(prob_profit)
    stop_pct = _number(plan.get("stop_pct"))
    target_r = _number(plan.get("target_r"))
    target_pct = _number(plan.get("target_pct"))
    round_trip_cost = _number(estimated_round_trip_cost)
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
    gross_expectancy = (
        win_probability * target_dollars - (1 - win_probability) * stop_dollars
    )
    return gross_expectancy - round_trip_cost


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
    estimated_round_trip_cost: object = 0.0,
) -> StrategySuggestion:
    """Apply managed ORB expectancy to an immutable strategy suggestion."""
    expected_value = managed_expected_value(
        credit_or_debit=suggestion.credit_or_debit,
        prob_profit=suggestion.prob_profit,
        plan=plan,
        estimated_round_trip_cost=estimated_round_trip_cost,
    )
    if not isinstance(plan, Mapping) or plan.get("status") != "entry_confirmed":
        return suggestion
    return replace(suggestion, expected_value=expected_value)
