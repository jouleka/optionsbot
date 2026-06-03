"""top_k rank_by='expectancy' sorts by expected_value (IBK-93 Phase B)."""

from __future__ import annotations

from types import SimpleNamespace

from optionsbot.scoring.composite import top_k
from optionsbot.scoring.types import FactorBreakdown, ScoredStrategy


def _scored(name: str, score: float, ev: float | None) -> ScoredStrategy:
    sug = SimpleNamespace(expected_value=ev)
    return ScoredStrategy(
        strategy_name=name, score=score,
        factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        suggestion=sug, rationale="",  # type: ignore[arg-type]
    )


def test_rank_by_score_is_default() -> None:
    a = _scored("a", 90.0, ev=10.0)
    b = _scored("b", 80.0, ev=999.0)
    assert [s.strategy_name for s in top_k((a, b), k=2, threshold=0.0)] == ["a", "b"]


def test_rank_by_expectancy_sorts_by_ev() -> None:
    a = _scored("a", 90.0, ev=10.0)
    b = _scored("b", 80.0, ev=999.0)
    ranked = top_k((a, b), k=2, threshold=0.0, rank_by="expectancy")
    assert [s.strategy_name for s in ranked] == ["b", "a"]  # higher EV first


def test_rank_by_expectancy_none_ev_sorts_last() -> None:
    a = _scored("a", 90.0, ev=None)
    b = _scored("b", 70.0, ev=5.0)
    ranked = top_k((a, b), k=2, threshold=0.0, rank_by="expectancy")
    assert [s.strategy_name for s in ranked] == ["b", "a"]  # None EV ranks last
