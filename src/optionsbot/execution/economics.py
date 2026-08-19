"""Reconcile scanner economics to the exact fresh entry package price."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from optionsbot.execution.risk_structure import (
    structural_max_loss_dollars,
    structural_max_profit_dollars,
)
from optionsbot.opening_range_economics import managed_expected_value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class ReconciledEntryEconomics:
    """Authoritative price-sensitive metrics for one contract set."""

    credit_or_debit: float
    max_loss: float
    max_profit: float | None
    reward_risk: float | None
    expected_value: float | None
    terminal_expected_value: float | None
    gross_managed_expected_value: float | None
    managed_expected_value: float | None
    estimated_round_trip_cost: float | None
    scan_credit_or_debit: float | None
    scan_expected_value: float | None
    scan_terminal_expected_value: float | None
    fresh_net_per_share: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def reconcile_entry_economics(
    legs: list[dict[str, Any]],
    suggestion: dict[str, Any],
    *,
    fresh_net_per_share: float,
    estimated_round_trip_cost: object = 0.0,
) -> ReconciledEntryEconomics | None:
    """Reprice all entry-price-sensitive metrics from one fresh combo mid.

    Expected terminal intrinsic value is independent of entry price. Therefore
    fresh terminal EV equals scan terminal EV plus the exact change in entry
    cashflow. Exact ORB/FVG entries additionally use their configured premium
    stop and R target for the authoritative managed expectancy.
    """
    if not math.isfinite(fresh_net_per_share):
        return None
    max_loss = structural_max_loss_dollars(
        legs,
        entry_net_per_share=fresh_net_per_share,
    )
    if max_loss is None or not math.isfinite(max_loss) or max_loss <= 0:
        return None
    max_profit = structural_max_profit_dollars(
        legs,
        entry_net_per_share=fresh_net_per_share,
    )
    fresh_cashflow = fresh_net_per_share * 100.0
    scan_cashflow = _finite_number(suggestion.get("credit_or_debit"))
    scan_ev = _finite_number(suggestion.get("expected_value"))
    scan_terminal_ev = _finite_number(suggestion.get("terminal_expected_value"))
    if scan_terminal_ev is None and suggestion.get("opening_range_fvg") is None:
        scan_terminal_ev = scan_ev
    terminal_expected_value = (
        scan_terminal_ev + fresh_cashflow - scan_cashflow
        if scan_terminal_ev is not None and scan_cashflow is not None
        else None
    )
    opening_range_plan = suggestion.get("opening_range_fvg")
    opening_range_candidate = (
        isinstance(opening_range_plan, dict)
        and opening_range_plan.get("status") == "entry_confirmed"
    )
    gross_managed_ev = managed_expected_value(
        credit_or_debit=fresh_cashflow,
        prob_profit=suggestion.get("prob_profit"),
        plan=opening_range_plan,
    )
    managed_ev = managed_expected_value(
        credit_or_debit=fresh_cashflow,
        prob_profit=suggestion.get("prob_profit"),
        plan=opening_range_plan,
        estimated_round_trip_cost=estimated_round_trip_cost,
    )
    round_trip_cost = _finite_number(estimated_round_trip_cost)
    expected_value = managed_ev if opening_range_candidate else terminal_expected_value
    reward_risk = (
        max_profit / max_loss
        if max_profit is not None and math.isfinite(max_profit) and max_profit > 0
        else None
    )
    return ReconciledEntryEconomics(
        credit_or_debit=fresh_cashflow,
        max_loss=max_loss,
        max_profit=max_profit,
        reward_risk=reward_risk,
        expected_value=expected_value,
        terminal_expected_value=terminal_expected_value,
        gross_managed_expected_value=gross_managed_ev,
        managed_expected_value=managed_ev,
        estimated_round_trip_cost=round_trip_cost,
        scan_credit_or_debit=scan_cashflow,
        scan_expected_value=scan_ev,
        scan_terminal_expected_value=scan_terminal_ev,
        fresh_net_per_share=fresh_net_per_share,
    )
