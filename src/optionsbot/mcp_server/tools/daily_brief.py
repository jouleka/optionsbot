"""daily_brief tool (IBK-107).

Read-only cross-symbol synthesis: assembles a grounded "what makes most sense
today" decision packet from the latest PERSISTED snapshots (no IBKR, no LLM, no
writes) for a Claude MCP client to reason over. Edge is reconstructed from each
persisted ``suggestion_json`` so the canonical ``edge_sort_key`` /
``has_positive_edge`` apply unchanged -- the brief can never disagree with /scan.
"""

from __future__ import annotations

from typing import Any

from optionsbot.scoring.composite import edge_sort_key
from optionsbot.strategies.base import StrategySuggestion

_TIER_NAMES = {2: "positive", 1: "negative", 0: "undefined"}

RUBRIC = (
    "You are reasoning over a grounded cross-symbol options brief. Rules:\n"
    "1. If any_positive_edge is false, say plainly that nothing has positive edge "
    "today and do NOT manufacture a pick; you may mention the least-bad setup for "
    "reference, clearly labeled not a recommendation.\n"
    "2. Otherwise lead with the single best risk-adjusted setup -- the first ranked "
    "entry whose top setup has edge_tier 'positive' -- and state its symbol, strategy, "
    "expected_value, prob_profit, max_loss, and why.\n"
    "3. Offer one higher-reward alternative ONLY if it also has edge_tier 'positive'.\n"
    "4. Flag any stale snapshot_ts. Earnings proximity is NOT in this packet -- remind "
    "the user to check the earnings calendar before trading.\n"
    "5. Reason ONLY over the numbers in this packet; never invent expected_value, "
    "prob_profit, or edge."
)


def _reconstruct_suggestion(
    suggestion_json: dict[str, Any] | None, strategy_name: str, rationale: str
) -> StrategySuggestion:
    """Rebuild a StrategySuggestion from a persisted strategy_scores row.

    ``legs=()`` -- the edge math ignores legs; the raw ``legs_json`` is carried
    separately in the packet for display. Reusing the real class means
    ``risk_normalized_expectancy`` / ``edge_sort_key`` / ``has_positive_edge``
    apply canonically (no edge-formula duplication).
    """
    sj = suggestion_json or {}
    return StrategySuggestion(
        strategy_name=strategy_name,
        legs=(),
        credit_or_debit=sj.get("credit_or_debit", 0.0),
        max_loss=sj.get("max_loss"),
        max_profit=sj.get("max_profit"),
        prob_profit=sj.get("prob_profit"),
        suggested_quantity=sj.get("suggested_quantity", 0),
        defined_risk=sj.get("defined_risk", True),
        rationale=rationale,
        reward_risk=sj.get("reward_risk"),
        expected_value=sj.get("expected_value"),
        risk_tier=sj.get("risk_tier", "balanced"),
    )


def _edge_tier(suggestion: StrategySuggestion) -> str:
    """Human-readable tier behind edge_sort_key: positive / negative / undefined."""
    return _TIER_NAMES[edge_sort_key(suggestion)[0]]
