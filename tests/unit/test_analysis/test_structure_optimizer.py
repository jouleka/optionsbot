from __future__ import annotations

from datetime import UTC, datetime

import pytest

from optionsbot.analysis.opening_range_fvg import OpeningRangeFVGSignal
from optionsbot.analysis.structure_optimizer import (
    UnderlyingThesis,
    build_shadow_grid_for_thesis,
    build_shadow_structure_grid,
    underlying_thesis,
)
from optionsbot.ibkr.types import OptionChainLeg


def _signal(direction: str = "bull") -> OpeningRangeFVGSignal:
    return OpeningRangeFVGSignal(
        signal_id="2026-08-28:SPY:bull:fvg",
        session="2026-08-28",
        timeframe_minutes=1,
        direction=direction,  # type: ignore[arg-type]
        opening_range_high=100.5,
        opening_range_low=99.5,
        breakout_ts=datetime(2026, 8, 28, 13, 41, tzinfo=UTC),
        fvg_formed_ts=datetime(2026, 8, 28, 13, 43, tzinfo=UTC),
        fvg_low=100.4 if direction == "bull" else 99.4,
        fvg_high=100.6 if direction == "bull" else 99.6,
        respected_ts=datetime(2026, 8, 28, 13, 45, tzinfo=UTC),
        entry_underlying_price=101.0 if direction == "bull" else 99.0,
        stop_pct=0.15,
        target_r=1.5,
        target_pct=0.225,
    )


def _chain(right: str) -> list[OptionChainLeg]:
    result: list[OptionChainLeg] = []
    deltas = (0.15, 0.25, 0.35, 0.50, 0.65)
    strikes = (104.0, 103.0, 102.0, 100.0, 99.0)
    if right == "P":
        strikes = tuple(reversed(tuple(200.0 - strike for strike in strikes)))
    for delta, strike in zip(deltas, strikes, strict=True):
        signed_delta = delta if right == "C" else -delta
        result.append(
            OptionChainLeg(
                symbol="SPY",
                expiry="20260828",
                strike=strike,
                right=right,  # type: ignore[arg-type]
                bid=max(0.05, 2.0 * delta - 0.05),
                ask=2.0 * delta + 0.05,
                iv=0.25,
                delta=signed_delta,
                gamma=0.03,
                theta=-0.12,
                vega=0.04,
                open_interest=1_000,
                volume=500,
            )
        )
    return result


def test_underlying_thesis_uses_frozen_invalidation_and_r_multiple() -> None:
    thesis = underlying_thesis(_signal(), timeout_minutes=90)
    assert thesis.invalidation_spot == pytest.approx(100.4)
    assert thesis.target_spot == pytest.approx(101.9)


def test_grid_builds_unique_long_calls_and_bull_call_spreads() -> None:
    candidates = build_shadow_structure_grid(
        _chain("C"),
        _signal(),
        timeout_minutes=90,
        commission_per_contract=0.70,
    )
    assert candidates
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert any(len(candidate.legs) == 1 for candidate in candidates)
    assert any(len(candidate.legs) == 2 for candidate in candidates)
    assert all(candidate.admission_enabled is False for candidate in candidates)
    assert all(candidate.entry_debit_dollars > 0 for candidate in candidates)
    assert all(candidate.features["thesis_entry_spot"] == 101.0 for candidate in candidates)
    assert all(
        candidate.features["thesis_invalidation_spot"] == 100.4
        for candidate in candidates
    )
    assert all(
        candidate.features["thesis_target_spot"] == pytest.approx(101.9)
        for candidate in candidates
    )


def test_grid_mirrors_for_bear_puts() -> None:
    candidates = build_shadow_structure_grid(
        _chain("P"),
        _signal("bear"),
        timeout_minutes=60,
        commission_per_contract=0.70,
    )
    assert any(candidate.strategy.startswith("long_put") for candidate in candidates)
    for candidate in candidates:
        if len(candidate.legs) == 2:
            bought = next(leg for leg in candidate.legs if leg.side == "buy")
            sold = next(leg for leg in candidate.legs if leg.side == "sell")
            assert bought.strike is not None and sold.strike is not None
            assert bought.strike > sold.strike


def test_grid_rejects_wrong_expiry_or_incomplete_greeks() -> None:
    wrong = [
        OptionChainLeg(
            symbol="SPY",
            expiry="20260901",
            strike=100.0,
            right="C",
            bid=1.0,
            ask=1.1,
            iv=0.2,
            delta=0.5,
            gamma=None,
            theta=-0.1,
            vega=0.1,
            open_interest=100,
            volume=10,
        )
    ]
    assert not build_shadow_structure_grid(
        wrong,
        _signal(),
        timeout_minutes=90,
        commission_per_contract=0.70,
    )


def test_generic_thesis_entrypoint_is_bounded_and_preserves_or_wrapper() -> None:
    thesis = UnderlyingThesis(
        direction="bull",
        entry_spot=101.0,
        invalidation_spot=100.4,
        target_spot=101.9,
        timeout_minutes=90.0,
    )
    generic = build_shadow_grid_for_thesis(
        _chain("C"),
        thesis,
        expiry="20260828",
        target_pct=0.225,
        commission_per_contract=0.70,
        max_candidates=2,
    )
    wrapped = build_shadow_structure_grid(
        _chain("C"),
        _signal(),
        timeout_minutes=90.0,
        commission_per_contract=0.70,
        max_candidates=2,
    )
    assert 0 < len(generic) <= 2
    assert [item.candidate_id for item in generic] == [
        item.candidate_id for item in wrapped
    ]
    assert all(item.admission_enabled is False for item in generic)


def test_generic_thesis_entrypoint_rejects_directionally_invalid_levels() -> None:
    with pytest.raises(ValueError, match="direction"):
        build_shadow_grid_for_thesis(
            _chain("C"),
            UnderlyingThesis(
                direction="bull",
                entry_spot=101.0,
                invalidation_spot=102.0,
                target_spot=103.0,
                timeout_minutes=30.0,
            ),
            expiry="20260828",
            target_pct=0.225,
            commission_per_contract=0.70,
        )
