"""Thesis-aware 0DTE option structure grid for shadow comparison.

The legacy strategy constructors pick one fixed delta.  This module instead
freezes an underlying thesis and builds a bounded, liquid grid of long options
and debit verticals.  It does not decide whether to trade: the managed-outcome
model must supply calibrated path probabilities before any candidate can be
ranked or admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from optionsbot.analysis.opening_range_fvg import OpeningRangeFVGSignal
from optionsbot.ibkr.types import OptionChainLeg, OptionRight
from optionsbot.strategies.base import Leg

Direction = Literal["bull", "bear"]


@dataclass(frozen=True, slots=True)
class UnderlyingThesis:
    direction: Direction
    entry_spot: float
    invalidation_spot: float
    target_spot: float
    timeout_minutes: float


@dataclass(frozen=True, slots=True)
class ShadowStructureCandidate:
    candidate_id: str
    strategy: str
    legs: tuple[Leg, ...]
    entry_debit_dollars: float
    maximum_loss_dollars: float
    maximum_profit_dollars: float | None
    round_trip_friction_dollars: float
    desired_premium_target_dollars: float
    premium_target_feasible: bool
    target_scenario_pnl_dollars: float
    invalidation_scenario_pnl_dollars: float
    timeout_scenario_pnl_dollars: float
    features: dict[str, float | int | str | bool | None]
    schema_version: str = "thesis_structure_grid_v1"
    admission_enabled: bool = False


def underlying_thesis(
    signal: OpeningRangeFVGSignal,
    *,
    timeout_minutes: float,
) -> UnderlyingThesis:
    """Derive measurable invalidation/target levels from the frozen setup."""
    if not math.isfinite(timeout_minutes) or timeout_minutes <= 0.0:
        raise ValueError("timeout_minutes must be positive")
    entry = float(signal.entry_underlying_price)
    invalidation = float(signal.fvg_low if signal.direction == "bull" else signal.fvg_high)
    risk_distance = (
        entry - invalidation if signal.direction == "bull" else invalidation - entry
    )
    if not entry > 0.0 or not risk_distance > 0.0:
        raise ValueError("signal does not define a positive underlying invalidation distance")
    target = (
        entry + risk_distance * signal.target_r
        if signal.direction == "bull"
        else entry - risk_distance * signal.target_r
    )
    if target <= 0.0:
        raise ValueError("signal target must remain positive")
    return UnderlyingThesis(
        direction=signal.direction,
        entry_spot=entry,
        invalidation_spot=invalidation,
        target_spot=target,
        timeout_minutes=timeout_minutes,
    )


def _usable(leg: OptionChainLeg, right: OptionRight, expiry: str) -> bool:
    values = (leg.bid, leg.ask, leg.delta, leg.gamma, leg.theta, leg.vega)
    return (
        leg.right == right
        and leg.expiry == expiry
        and all(value is not None and math.isfinite(float(value)) for value in values)
        and leg.bid is not None
        and leg.ask is not None
        and 0.0 <= leg.bid <= leg.ask
    )


def _nearest_unique(
    legs: Sequence[OptionChainLeg], targets: Sequence[float]
) -> list[OptionChainLeg]:
    selected: list[OptionChainLeg] = []
    seen: set[tuple[str, float, str]] = set()
    for target in targets:
        candidate = min(
            legs,
            key=lambda leg: abs(float(leg.delta or 0.0) - target),
            default=None,
        )
        if candidate is None:
            continue
        key = (candidate.expiry, candidate.strike, candidate.right)
        if key not in seen:
            selected.append(candidate)
            seen.add(key)
    return selected


def _project_mid(
    leg: OptionChainLeg,
    *,
    current_spot: float,
    future_spot: float,
    elapsed_minutes: float,
) -> float:
    assert leg.bid is not None
    assert leg.ask is not None
    assert leg.delta is not None
    assert leg.gamma is not None
    assert leg.theta is not None
    mid = (leg.bid + leg.ask) / 2.0
    move = future_spot - current_spot
    # IBKR theta is conventionally an approximate one-calendar-day change.
    projected = (
        mid
        + leg.delta * move
        + 0.5 * leg.gamma * move * move
        + leg.theta * (elapsed_minutes / 1_440.0)
    )
    intrinsic = (
        max(0.0, future_spot - leg.strike)
        if leg.right == "C"
        else max(0.0, leg.strike - future_spot)
    )
    upper = future_spot if leg.right == "C" else leg.strike
    return min(max(projected, intrinsic, 0.0), max(upper, intrinsic))


def _entry_debit(quotes: Sequence[tuple[OptionChainLeg, str]]) -> float:
    per_share = 0.0
    for quote, side in quotes:
        assert quote.bid is not None and quote.ask is not None
        per_share += quote.ask if side == "buy" else -quote.bid
    return per_share * 100.0


def _liquidation_value(
    quotes: Sequence[tuple[OptionChainLeg, str]],
    *,
    current_spot: float,
    future_spot: float,
    elapsed_minutes: float,
) -> float:
    per_share = 0.0
    for quote, side in quotes:
        assert quote.bid is not None and quote.ask is not None
        projected_mid = _project_mid(
            quote,
            current_spot=current_spot,
            future_spot=future_spot,
            elapsed_minutes=elapsed_minutes,
        )
        half_spread = (quote.ask - quote.bid) / 2.0
        if side == "buy":
            per_share += max(0.0, projected_mid - half_spread)
        else:
            per_share -= projected_mid + half_spread
    return per_share * 100.0


def _candidate(
    quotes: Sequence[tuple[OptionChainLeg, str]],
    *,
    strategy: str,
    thesis: UnderlyingThesis,
    target_pct: float,
    commission_per_contract: float,
) -> ShadowStructureCandidate | None:
    debit = _entry_debit(quotes)
    if not math.isfinite(debit) or debit <= 0.0:
        return None
    leg_count = len(quotes)
    commission = leg_count * 2.0 * commission_per_contract
    current_liquidation = _liquidation_value(
        quotes,
        current_spot=thesis.entry_spot,
        future_spot=thesis.entry_spot,
        elapsed_minutes=0.0,
    )
    friction = max(0.0, debit - current_liquidation) + commission
    legs = tuple(
        Leg(
            symbol=quote.symbol,
            side=side,  # type: ignore[arg-type]
            expiry=quote.expiry,
            strike=quote.strike,
            right=quote.right,
        )
        for quote, side in quotes
    )
    maximum_profit: float | None = None
    if len(quotes) == 2:
        bought = next(quote for quote, side in quotes if side == "buy")
        sold = next(quote for quote, side in quotes if side == "sell")
        maximum_profit = abs(sold.strike - bought.strike) * 100.0 - debit
        if maximum_profit <= 0.0:
            return None
    target_value = _liquidation_value(
        quotes,
        current_spot=thesis.entry_spot,
        future_spot=thesis.target_spot,
        elapsed_minutes=thesis.timeout_minutes / 2.0,
    )
    invalidation_value = _liquidation_value(
        quotes,
        current_spot=thesis.entry_spot,
        future_spot=thesis.invalidation_spot,
        elapsed_minutes=thesis.timeout_minutes / 2.0,
    )
    timeout_value = _liquidation_value(
        quotes,
        current_spot=thesis.entry_spot,
        future_spot=thesis.entry_spot,
        elapsed_minutes=thesis.timeout_minutes,
    )
    desired_target = debit * target_pct
    feasible = maximum_profit is None or desired_target + commission <= maximum_profit
    serialized = [asdict(leg) for leg in legs]
    candidate_id = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    net_delta = sum(
        float(quote.delta or 0.0) * (1.0 if side == "buy" else -1.0)
        for quote, side in quotes
    )
    net_gamma = sum(
        float(quote.gamma or 0.0) * (1.0 if side == "buy" else -1.0)
        for quote, side in quotes
    )
    net_theta = sum(
        float(quote.theta or 0.0) * (1.0 if side == "buy" else -1.0)
        for quote, side in quotes
    )
    net_vega = sum(
        float(quote.vega or 0.0) * (1.0 if side == "buy" else -1.0)
        for quote, side in quotes
    )
    return ShadowStructureCandidate(
        candidate_id=candidate_id,
        strategy=strategy,
        legs=legs,
        entry_debit_dollars=debit,
        maximum_loss_dollars=debit,
        maximum_profit_dollars=maximum_profit,
        round_trip_friction_dollars=friction,
        desired_premium_target_dollars=desired_target,
        premium_target_feasible=feasible,
        target_scenario_pnl_dollars=target_value - debit - commission,
        invalidation_scenario_pnl_dollars=invalidation_value - debit - commission,
        timeout_scenario_pnl_dollars=timeout_value - debit - commission,
        features={
            "structure_kind": "long_option" if len(quotes) == 1 else "debit_vertical",
            "leg_count": leg_count,
            "entry_debit_dollars": debit,
            "round_trip_friction_dollars": friction,
            "friction_fraction": friction / debit,
            "maximum_profit_dollars": maximum_profit,
            "net_delta": net_delta,
            "net_gamma": net_gamma,
            "net_theta": net_theta,
            "net_vega": net_vega,
            "thesis_entry_spot": thesis.entry_spot,
            "thesis_invalidation_spot": thesis.invalidation_spot,
            "thesis_target_spot": thesis.target_spot,
            "underlying_risk_fraction": abs(
                thesis.entry_spot - thesis.invalidation_spot
            )
            / thesis.entry_spot,
            "underlying_reward_risk": abs(
                thesis.target_spot - thesis.entry_spot
            )
            / abs(thesis.entry_spot - thesis.invalidation_spot),
            "timeout_minutes": thesis.timeout_minutes,
            "premium_target_feasible": feasible,
        },
    )


def build_shadow_grid_for_thesis(
    chain: Sequence[OptionChainLeg],
    thesis: UnderlyingThesis,
    *,
    expiry: str,
    target_pct: float,
    commission_per_contract: float,
    max_candidates: int | None = None,
) -> tuple[ShadowStructureCandidate, ...]:
    """Build a bounded grid for any frozen directional underlying thesis."""
    if commission_per_contract < 0.0 or not math.isfinite(commission_per_contract):
        raise ValueError("commission_per_contract must be finite and non-negative")
    if not math.isfinite(target_pct) or target_pct <= 0.0:
        raise ValueError("target_pct must be finite and positive")
    if len(expiry) != 8 or not expiry.isdigit():
        raise ValueError("expiry must be YYYYMMDD")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive when provided")
    values = (
        thesis.entry_spot,
        thesis.invalidation_spot,
        thesis.target_spot,
        thesis.timeout_minutes,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("thesis prices and timeout must be finite and positive")
    valid_geometry = (
        thesis.invalidation_spot < thesis.entry_spot < thesis.target_spot
        if thesis.direction == "bull"
        else thesis.target_spot < thesis.entry_spot < thesis.invalidation_spot
    )
    if not valid_geometry:
        raise ValueError("thesis levels do not match its direction")
    right: OptionRight = "C" if thesis.direction == "bull" else "P"
    usable = [leg for leg in chain if _usable(leg, right, expiry)]
    if not usable:
        return ()
    sign = 1.0 if right == "C" else -1.0
    long_quotes = _nearest_unique(usable, [sign * value for value in (0.35, 0.50, 0.65)])
    short_quotes = _nearest_unique(usable, [sign * value for value in (0.15, 0.25, 0.35)])
    candidates: list[ShadowStructureCandidate] = []
    for quote in long_quotes:
        delta = abs(float(quote.delta or 0.0))
        candidate = _candidate(
            [(quote, "buy")],
            strategy=f"long_{'call' if right == 'C' else 'put'}_d{round(delta * 100):02d}",
            thesis=thesis,
            target_pct=target_pct,
            commission_per_contract=commission_per_contract,
        )
        if candidate is not None:
            candidates.append(candidate)
    for bought in long_quotes:
        for sold in short_quotes:
            valid_width = (
                bought.strike < sold.strike
                if right == "C"
                else bought.strike > sold.strike
            )
            if not valid_width or abs(float(bought.delta or 0.0)) <= abs(
                float(sold.delta or 0.0)
            ):
                continue
            long_delta = round(abs(float(bought.delta or 0.0)) * 100)
            short_delta = round(abs(float(sold.delta or 0.0)) * 100)
            candidate = _candidate(
                [(bought, "buy"), (sold, "sell")],
                strategy=(
                    f"{'bull_call' if right == 'C' else 'bear_put'}_spread_"
                    f"d{long_delta:02d}_{short_delta:02d}"
                ),
                thesis=thesis,
                target_pct=target_pct,
                commission_per_contract=commission_per_contract,
            )
            if candidate is not None:
                candidates.append(candidate)
    unique = {candidate.candidate_id: candidate for candidate in candidates}
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            not item.premium_target_feasible,
            item.round_trip_friction_dollars / item.entry_debit_dollars,
            item.strategy,
        ),
    )
    return tuple(ordered[:max_candidates] if max_candidates is not None else ordered)


def build_shadow_structure_grid(
    chain: Sequence[OptionChainLeg],
    signal: OpeningRangeFVGSignal,
    *,
    timeout_minutes: float,
    commission_per_contract: float,
    max_candidates: int | None = None,
) -> tuple[ShadowStructureCandidate, ...]:
    """Backward-compatible OR/FVG wrapper around the generic thesis grid."""
    thesis = underlying_thesis(signal, timeout_minutes=timeout_minutes)
    return build_shadow_grid_for_thesis(
        chain,
        thesis,
        expiry=signal.session.replace("-", ""),
        target_pct=signal.target_pct,
        commission_per_contract=commission_per_contract,
        max_candidates=max_candidates,
    )
