"""Tests for alert formatter (IBK-66)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from optionsbot.alerts import format_alert_markdown
from optionsbot.analysis.types import MarketView
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown


def _view(direction="bull", iv_regime="high") -> MarketView:
    return MarketView(
        direction=direction, direction_strength="strong", iv_regime=iv_regime,
        iv_rank_value=0.72, earnings_in_window=False, warming_up=False,
    )


def _scored(
    name: str = "iron_condor", score: float = 85.0,
    defined_risk: bool = True, credit: float = 1.25, max_loss: float | None = 3.75,
) -> ScoredStrategy:
    sug = MagicMock()
    sug.legs = (
        MagicMock(
            sec_type="OPT", symbol="SPY", side="sell", strike=410.0, right="C", expiry="20260711"
        ),
        MagicMock(
            sec_type="OPT", symbol="SPY", side="buy", strike=415.0, right="C", expiry="20260711"
        ),
    )
    sug.credit_or_debit = credit
    sug.max_loss = max_loss
    sug.max_profit = credit
    sug.prob_profit = 0.68
    sug.suggested_quantity = 5
    sug.defined_risk = defined_risk
    return ScoredStrategy(
        strategy_name=name, score=score,
        factors=FactorBreakdown(0.7, 0.6, 0.8, 0.9, 1.0, 0.5),
        suggestion=sug, rationale="High IV rank + tight liquidity.",
    )


def test_format_includes_symbol_and_view() -> None:
    md = format_alert_markdown(
        symbol="SPY", view=_view(), scored=_scored(),
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
    )
    assert "SPY" in md
    assert "bull" in md.lower() or "Bull" in md
    assert "high" in md.lower() or "High" in md


def test_format_includes_score_and_strategy_name() -> None:
    md = format_alert_markdown(
        symbol="SPY", view=_view(), scored=_scored(name="bull_put_spread", score=87.3),
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
    )
    assert "bull_put_spread" in md or "Bull Put Spread" in md
    assert "87" in md


def test_format_includes_legs_with_strike_and_expiry() -> None:
    md = format_alert_markdown(
        symbol="SPY", view=_view(), scored=_scored(),
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
    )
    assert "410" in md
    assert "415" in md
    assert "20260711" in md or "2026-07-11" in md


def test_format_includes_credit_max_loss_prob_profit() -> None:
    md = format_alert_markdown(
        symbol="SPY", view=_view(),
        scored=_scored(credit=1.25, max_loss=3.75),
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
    )
    assert "1.25" in md
    assert "3.75" in md
    assert "68" in md  # 0.68 prob_profit -> "68%"


def test_format_warns_when_undefined_risk() -> None:
    md = format_alert_markdown(
        symbol="SPY", view=_view(),
        scored=_scored(defined_risk=False, max_loss=None),
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
    )
    assert "UNDEFINED RISK" in md


def test_format_does_not_warn_for_defined_risk() -> None:
    md = format_alert_markdown(
        symbol="SPY", view=_view(),
        scored=_scored(defined_risk=True),
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
    )
    assert "UNDEFINED RISK" not in md


def test_format_includes_rationale() -> None:
    md = format_alert_markdown(
        symbol="SPY", view=_view(),
        scored=_scored(),
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
    )
    assert "High IV rank" in md
