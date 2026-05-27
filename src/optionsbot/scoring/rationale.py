"""Rationale text generator for scored strategies.

Generates a 1-3 sentence string per :class:`ScoredStrategy`, citing the top
two highest-weighted factor contributions (factor value * weight) and flags
strategies with ``defined_risk=False`` so downstream consumers can warn the
user. The MCP ``analyze`` tool (IBK-6) and the daemon's alert dispatcher
(IBK-7) include this string in their output so the rationale never has to
be re-derived from raw factor numbers.
"""

from __future__ import annotations

from optionsbot.scoring.types import FactorBreakdown
from optionsbot.strategies import Strategy

# Human-readable labels for the six factor names. Kept in sync with the
# field order of :class:`FactorBreakdown` and the
# :mod:`optionsbot.scoring.factors` calculators.
_FACTOR_DESCRIPTIONS: dict[str, str] = {
    "iv_rank": "IV rank",
    "iv_hv": "IV/HV ratio",
    "liquidity": "options liquidity",
    "dte_match": "DTE match",
    "earnings_penalty": "earnings-window cleanliness",
    "range_bound": "range-bound underlying",
}


def build_rationale(
    score: float,
    factors: FactorBreakdown,
    strategy: Strategy,
) -> str:
    """Return a short human-readable rationale citing the top contributors.

    Pattern::

        "<Strategy display name> scored <NN.N> / 100. top contributors:
         <factor_a> (<v>) and <factor_b> (<v>)[. UNDEFINED RISK -- ...]."

    The top two factors are picked by ``value * weight`` (the actual
    contribution to the composite score), not by raw factor value alone --
    a high factor with zero weight contributes nothing and shouldn't be
    cited. Period-separated, single-line; no newlines.
    """
    factors_d = factors.as_dict()
    weighted = [
        (name, value, value * strategy.factor_weights.get(name, 0.0))
        for name, value in factors_d.items()
    ]
    # Sort by contribution (value * weight) descending.
    weighted.sort(key=lambda t: t[2], reverse=True)
    top2 = weighted[:2]
    parts = [f"{strategy.display_name} scored {score:.1f} / 100"]
    if top2:
        factor_phrases = [
            f"{_FACTOR_DESCRIPTIONS.get(name, name)} ({value:.2f})"
            for name, value, _ in top2
        ]
        parts.append("top contributors: " + " and ".join(factor_phrases))
    if not strategy.defined_risk:
        parts.append("UNDEFINED RISK -- consider the defined-risk alternative")
    return ". ".join(parts) + "."
