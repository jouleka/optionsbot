"""top_k rank_by='expectancy' sorts by risk-normalized expectancy (IBK-104)."""

from __future__ import annotations

from types import SimpleNamespace

from optionsbot.scoring.composite import top_k
from optionsbot.scoring.types import FactorBreakdown, ScoredStrategy


def _scored(name: str, score: float, rne: float | None) -> ScoredStrategy:
    sug = SimpleNamespace(risk_normalized_expectancy=rne)
    return ScoredStrategy(
        strategy_name=name, score=score,
        factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        suggestion=sug, rationale="",  # type: ignore[arg-type]
    )


def test_rank_by_score_is_default() -> None:
    a = _scored("a", 90.0, rne=0.01)
    b = _scored("b", 80.0, rne=0.99)
    assert [s.strategy_name for s in top_k((a, b), k=2, threshold=0.0)] == ["a", "b"]


def test_rank_by_expectancy_sorts_by_normalized_edge() -> None:
    # b has lower score but higher edge-per-risk -> ranks first under expectancy.
    a = _scored("a", 90.0, rne=0.01)
    b = _scored("b", 80.0, rne=0.50)
    ranked = top_k((a, b), k=2, threshold=0.0, rank_by="expectancy")
    assert [s.strategy_name for s in ranked] == ["b", "a"]


def test_rank_by_expectancy_none_edge_sorts_last() -> None:
    a = _scored("a", 90.0, rne=None)
    b = _scored("b", 70.0, rne=0.02)
    ranked = top_k((a, b), k=2, threshold=0.0, rank_by="expectancy")
    assert [s.strategy_name for s in ranked] == ["b", "a"]
