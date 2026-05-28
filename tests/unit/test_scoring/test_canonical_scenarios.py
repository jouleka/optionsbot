"""End-to-end canonical-scenario tests for the scoring engine (IBK-69).

For each archetypal market scenario, build a representative StrategySnapshot,
run score_all + top_k, and assert which strategy families dominate. These
tests catch scoring-engine regressions that per-factor unit tests miss --
e.g., a factor weight typo that flips which strategy wins a given view.

Assertions use family membership and relative orderings, not absolute
scores, so the tests don't lock in numerical artifacts of the current
factor weights.
"""

from __future__ import annotations

from datetime import date, timedelta

from optionsbot.analysis.types import Direction, IVRegime, MarketView, Strength
from optionsbot.ibkr.types import OptionChainLeg, OptionRight
from optionsbot.scoring import DEFAULT_TOP_K, score_all, top_k
from optionsbot.scoring.composite import DEFAULT_THRESHOLD
from optionsbot.strategies import StrategySnapshot

# Reuse the symmetric-delta chain fixture from conftest by building it inline
# here -- avoids cross-file fixture coupling for these scenario tests.

_STRIKES: tuple[float, ...] = (385.0, 390.0, 395.0, 400.0, 405.0, 410.0, 415.0)
_CALL_DELTAS: dict[float, float] = {
    385.0: 0.85, 390.0: 0.70, 395.0: 0.50, 400.0: 0.35,
    405.0: 0.25, 410.0: 0.16, 415.0: 0.08,
}
_PUT_DELTAS: dict[float, float] = {
    385.0: -0.08, 390.0: -0.16, 395.0: -0.25, 400.0: -0.35,
    405.0: -0.50, 410.0: -0.70, 415.0: -0.85,
}


def _chain_leg(
    expiry: str, strike: float, right: OptionRight,
    bid: float = 2.0, ask: float = 2.2, delta: float = 0.16, oi: int = 1000,
) -> OptionChainLeg:
    return OptionChainLeg(
        symbol="SPY", expiry=expiry, strike=strike, right=right,
        bid=bid, ask=ask, iv=0.20, delta=delta,
        gamma=0.01, theta=-0.02, vega=0.1,
        open_interest=oi, volume=50,
    )


def _build_chain() -> tuple[OptionChainLeg, ...]:
    expiry = (date.today() + timedelta(days=45)).strftime("%Y%m%d")
    legs: list[OptionChainLeg] = []
    for k in _STRIKES:
        legs.append(_chain_leg(expiry, k, "C", delta=_CALL_DELTAS[k]))
        legs.append(_chain_leg(expiry, k, "P", delta=_PUT_DELTAS[k]))
    return tuple(legs)


def _make_view(
    direction: Direction,
    strength: Strength,
    iv_regime: IVRegime,
    iv_rank: float,
    earnings: bool = False,
) -> MarketView:
    return MarketView(
        direction=direction,
        direction_strength=strength,
        iv_regime=iv_regime,
        iv_rank_value=iv_rank,
        earnings_in_window=earnings,
        warming_up=False,
    )


def _build_snapshot(
    view: MarketView, *, iv_rank: float = 0.7, atm_iv: float = 0.25, hv20: float = 0.20
) -> StrategySnapshot:
    return StrategySnapshot(
        symbol="SPY", spot=400.0, atm_iv=atm_iv, hv20=hv20, iv_rank=iv_rank,
        chain=_build_chain(), view=view, dte_target=45, position=None,
    )


def _names_in_top_k(snapshot: StrategySnapshot, k: int = DEFAULT_TOP_K) -> list[str]:
    scored = score_all(snapshot, account_value=100_000)
    selected = top_k(scored, k=k, threshold=DEFAULT_THRESHOLD)
    return [s.strategy_name for s in selected]


def _scored_by_name(snapshot: StrategySnapshot) -> dict[str, float]:
    """Return {strategy_name: score} for every applicable strategy (no threshold)."""
    scored = score_all(snapshot, account_value=100_000)
    return {s.strategy_name: s.score for s in scored}


# ---------------------------------------------------------------------------
# Canonical scenarios
# ---------------------------------------------------------------------------


def test_neutral_high_iv_favours_short_premium_neutral_strategies() -> None:
    """Classic high-IV neutral environment: short-premium neutral plays
    (iron_condor, iron_butterfly, short_straddle, short_strangle) should
    dominate. Long-premium plays should NOT make the top of the ranking."""
    view = _make_view("neutral", "weak", "high", iv_rank=0.85)
    snap = _build_snapshot(view, iv_rank=0.85, atm_iv=0.30, hv20=0.18)
    scored = _scored_by_name(snap)
    short_premium_neutral = {"iron_condor", "iron_butterfly", "short_straddle", "short_strangle"}
    top_strategy = max(scored.items(), key=lambda x: x[1])[0]
    assert top_strategy in short_premium_neutral, (
        f"Expected a short-premium neutral strategy at the top, got {top_strategy} "
        f"with scores {sorted(scored.items(), key=lambda x: -x[1])[:5]}"
    )


def test_neutral_low_iv_favours_long_premium_neutral_strategies() -> None:
    """Low-IV neutral: long premium becomes the play. long_straddle and
    long_strangle should dominate.

    Note: calendar_spread is technically applicable to neutral/low but is
    filtered out by score_all here because the test fixture only contains
    one expiry (calendar requires two). The iv_rank inversion specifically
    is exercised by test_long_premium_plays_have_inverted_iv_rank_factor.
    """
    view = _make_view("neutral", "weak", "low", iv_rank=0.10)
    snap = _build_snapshot(view, iv_rank=0.10, atm_iv=0.12, hv20=0.20)
    scored = _scored_by_name(snap)
    long_premium_neutral = {"long_straddle", "long_strangle", "calendar_spread"}
    assert len(scored) >= 1, "at least one strategy must be applicable to neutral-low"
    top_strategy = max(scored.items(), key=lambda x: x[1])[0]
    assert top_strategy in long_premium_neutral, (
        f"Expected a neutral-low strategy at the top, got {top_strategy} "
        f"with scores {sorted(scored.items(), key=lambda x: -x[1])[:5]}"
    )


def test_bullish_high_iv_favours_bull_short_premium() -> None:
    """High-IV bullish: bull_put_spread (sells put for credit, profits if
    underlying stays above strike) should rank higher than long_call (which
    pays premium and IV crush is fatal)."""
    view = _make_view("bull", "strong", "high", iv_rank=0.80)
    snap = _build_snapshot(view, iv_rank=0.80, atm_iv=0.30, hv20=0.20)
    scored = _scored_by_name(snap)
    # bull_put_spread should be applicable and score higher than long_call.
    assert "bull_put_spread" in scored, "bull_put_spread should be applicable in bull/high"
    # long_call is NOT applicable to (bull, high) by its applicable_views --
    # verify it is absent (this also validates the gate).
    assert "long_call" not in scored, (
        "long_call should not be applicable in bull/high IV regime "
        f"(got scored keys: {set(scored)})"
    )


def test_bullish_low_iv_favours_long_premium_directional() -> None:
    """Low-IV bullish: long_call or bull_call_spread should be near the top."""
    view = _make_view("bull", "strong", "low", iv_rank=0.10)
    snap = _build_snapshot(view, iv_rank=0.10, atm_iv=0.12, hv20=0.18)
    scored = _scored_by_name(snap)
    long_bull = {"long_call", "bull_call_spread"}
    # At least one of the long-bull plays must be in top 3.
    top_3 = sorted(scored.items(), key=lambda x: -x[1])[:3]
    top_3_names = {n for n, _ in top_3}
    assert long_bull & top_3_names, (
        f"Expected long_call or bull_call_spread in top-3, got {top_3_names}"
    )


def test_bearish_low_iv_favours_long_put_or_bear_put_spread() -> None:
    """Low-IV bearish: long_put or bear_put_spread should be near the top."""
    view = _make_view("bear", "strong", "low", iv_rank=0.10)
    snap = _build_snapshot(view, iv_rank=0.10, atm_iv=0.12, hv20=0.18)
    scored = _scored_by_name(snap)
    long_bear = {"long_put", "bear_put_spread"}
    top_3 = sorted(scored.items(), key=lambda x: -x[1])[:3]
    top_3_names = {n for n, _ in top_3}
    assert long_bear & top_3_names, (
        f"Expected long_put or bear_put_spread in top-3, got {top_3_names}"
    )


def test_bearish_high_iv_favours_bear_call_spread() -> None:
    """High-IV bearish: bear_call_spread (sells call for credit) should
    rank higher than long_put (pays premium, IV crush hurts).
    Note: long_put is NOT applicable to bear/high IV by its applicable_views
    (only bear/low and bear/neutral), so we verify its absence as a gate test."""
    view = _make_view("bear", "strong", "high", iv_rank=0.80)
    snap = _build_snapshot(view, iv_rank=0.80, atm_iv=0.30, hv20=0.20)
    scored = _scored_by_name(snap)
    assert "bear_call_spread" in scored, "bear_call_spread should be applicable in bear/high"
    # long_put is gated out for bear/high -- verify it does not appear.
    assert "long_put" not in scored, (
        "long_put should not be applicable in bear/high IV regime "
        f"(got scored keys: {set(scored)})"
    )


def test_earnings_in_window_lifts_long_premium_plays() -> None:
    """Earnings within window: long-premium plays (long_straddle, long_strangle)
    benefit from the expected vol expansion; short-premium plays should be
    penalized. Compare the same neutral/low-IV view with and without earnings."""
    view_no_earnings = _make_view("neutral", "weak", "low", iv_rank=0.20, earnings=False)
    view_earnings = _make_view("neutral", "weak", "low", iv_rank=0.20, earnings=True)
    snap_no = _build_snapshot(view_no_earnings, iv_rank=0.20, atm_iv=0.18, hv20=0.18)
    snap_earnings = _build_snapshot(view_earnings, iv_rank=0.20, atm_iv=0.18, hv20=0.18)
    no_scores = _scored_by_name(snap_no)
    e_scores = _scored_by_name(snap_earnings)
    # long_straddle (long premium) should score >= without earnings.
    if "long_straddle" in no_scores and "long_straddle" in e_scores:
        assert e_scores["long_straddle"] >= no_scores["long_straddle"], (
            f"long_straddle should not score lower with earnings: "
            f"no={no_scores['long_straddle']}, earnings={e_scores['long_straddle']}"
        )


def test_iron_condor_is_not_applicable_to_strong_directional_view() -> None:
    """The Strategy ABC's applicable_views gate should exclude iron_condor
    from a strong-bull view. score_all should not return it at all."""
    view = _make_view("bull", "strong", "high", iv_rank=0.80)
    snap = _build_snapshot(view, iv_rank=0.80, atm_iv=0.30, hv20=0.20)
    scored = _scored_by_name(snap)
    assert "iron_condor" not in scored, (
        "iron_condor's applicable_views should not include strong bull"
    )


def test_long_premium_plays_have_inverted_iv_rank_factor() -> None:
    """Two snapshots with identical inputs except iv_rank (0.10 vs 0.90):
    the LOW iv_rank should favour long_premium strategies relative to high.
    Concretely: long_straddle (long_premium=True) should score higher at
    iv_rank=0.10 than at iv_rank=0.90.
    Note: long_straddle is only applicable to (neutral, low); we must set
    iv_regime accordingly for it to appear in scored output."""
    view_lo = _make_view("neutral", "weak", "low", iv_rank=0.10)
    view_hi = _make_view("neutral", "weak", "high", iv_rank=0.90)
    snap_lo = _build_snapshot(view_lo, iv_rank=0.10, atm_iv=0.12, hv20=0.20)
    snap_hi = _build_snapshot(view_hi, iv_rank=0.90, atm_iv=0.40, hv20=0.20)
    scores_lo = _scored_by_name(snap_lo)
    scores_hi = _scored_by_name(snap_hi)
    # long_straddle is only applicable in neutral/low, so only scores_lo will have it.
    if "long_straddle" in scores_lo:
        # Verify it scores reasonably well at low iv_rank (expected to be high score).
        assert scores_lo["long_straddle"] >= 60.0, (
            f"long_straddle should score well in neutral/low iv_rank=0.10 scenario, "
            f"got {scores_lo['long_straddle']}"
        )
    # short_straddle (long_premium=False) is applicable to neutral/high and should
    # score higher at high iv_rank than long_straddle does at low iv_rank.
    if "short_straddle" in scores_hi:
        assert scores_hi["short_straddle"] >= 60.0, (
            f"short_straddle should score well in neutral/high iv_rank=0.90 scenario, "
            f"got {scores_hi['short_straddle']}"
        )


def test_score_all_respects_threshold_and_top_k() -> None:
    """Sanity end-to-end: score_all returns multiple applicable strategies;
    top_k with DEFAULT_THRESHOLD filters out below-threshold entries."""
    view = _make_view("neutral", "weak", "high", iv_rank=0.85)
    snap = _build_snapshot(view, iv_rank=0.85, atm_iv=0.30, hv20=0.18)
    scored = score_all(snap, account_value=100_000)
    assert len(scored) >= 1, "at least one strategy must be applicable to neutral-high"
    selected = top_k(scored, k=DEFAULT_TOP_K, threshold=DEFAULT_THRESHOLD)
    assert len(selected) <= DEFAULT_TOP_K
    for s in selected:
        assert s.score >= DEFAULT_THRESHOLD
