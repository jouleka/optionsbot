"""Unit tests for the rationale text generator (IBK-49)."""

from __future__ import annotations

from optionsbot.scoring.composite import score_strategy
from optionsbot.scoring.rationale import _FACTOR_DESCRIPTIONS, build_rationale
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.strategies import IronCondor, ShortStraddle, StrategySnapshot


def _breakdown(
    iv_rank: float = 0.8,
    iv_hv: float = 0.7,
    liquidity: float = 0.9,
    dte_match: float = 0.6,
    earnings_penalty: float = 1.0,
    range_bound: float = 1.0,
) -> FactorBreakdown:
    return FactorBreakdown(
        iv_rank=iv_rank,
        iv_hv=iv_hv,
        liquidity=liquidity,
        dte_match=dte_match,
        earnings_penalty=earnings_penalty,
        range_bound=range_bound,
    )


def test_rationale_contains_display_name() -> None:
    rationale = build_rationale(82.5, _breakdown(), IronCondor())
    assert "Iron Condor" in rationale


def test_rationale_contains_score_formatted_to_one_decimal() -> None:
    rationale = build_rationale(82.5, _breakdown(), IronCondor())
    assert "82.5 / 100" in rationale


def test_rationale_cites_at_least_two_factor_descriptions() -> None:
    rationale = build_rationale(82.5, _breakdown(), IronCondor())
    cited = [d for d in _FACTOR_DESCRIPTIONS.values() if d in rationale]
    assert len(cited) >= 2


def test_rationale_flags_undefined_risk_for_short_straddle() -> None:
    rationale = build_rationale(75.0, _breakdown(), ShortStraddle())
    assert "UNDEFINED RISK" in rationale


def test_rationale_omits_undefined_risk_for_defined_risk_strategies() -> None:
    rationale = build_rationale(82.5, _breakdown(), IronCondor())
    assert "UNDEFINED RISK" not in rationale


def test_score_strategy_populates_rationale_field(
    base_snapshot: StrategySnapshot,
) -> None:
    scored = score_strategy(base_snapshot, IronCondor(), account_value=100_000)
    assert scored is not None
    assert scored.rationale != ""
    assert "Iron Condor" in scored.rationale
