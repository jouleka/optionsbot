"""Daemon-side gates for Hermes-requested exits (IBK-138).

Hermes may ask the bot to consider a close because of news/catalyst context,
but news is only an input. This module keeps the trading-soundness policy as
pure code so MCP, daemon, and tests agree on the refusal reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN_CONFIDENCE = 0.70
ADVERSE_LOSS_FRACTION = 0.25
MAX_CLOSES_PER_POSITION_PER_DAY = 1
MAX_CLOSES_PER_PORTFOLIO_PER_DAY = 2

ALLOWED_CATALYST_TYPES: frozenset[str] = frozenset(
    {
        "headline_news",
        "downgrade_upgrade",
        "earnings_guidance",
        "sec_filing",
        "macro_rate",
        "volatility_shock",
        "price_action",
        "risk_management",
        "broker_reconcile",
    }
)

HEADLINE_CATALYST_TYPES: frozenset[str] = frozenset(
    {
        "headline_news",
        "downgrade_upgrade",
        "earnings_guidance",
        "sec_filing",
        "macro_rate",
        "volatility_shock",
    }
)


@dataclass(frozen=True, slots=True)
class ExitRequestGateInput:
    position_id: int
    catalyst_type: str
    confidence: float
    sources: list[Any]
    reason: str
    today_position_requests: int = 0
    today_portfolio_requests: int = 0


@dataclass(frozen=True, slots=True)
class QuoteGateState:
    entry_net: float
    current_net: float | None
    dte: int | None
    deterministic_exit_reason: str | None


@dataclass(frozen=True, slots=True)
class ExitRequestGateDecision:
    allowed: bool
    reason: str


def evaluate_exit_request_gate(
    request: ExitRequestGateInput,
    quote_state: QuoteGateState,
) -> ExitRequestGateDecision:
    """Return whether a Hermes exit request may become a close order.

    The rules intentionally err on the side of refusing:
    * closed catalyst enum and confidence floor;
    * one close per position per UTC day, two per portfolio per UTC day;
    * deterministic exit reasons always pass (the bot already wanted out);
    * otherwise at least two sources/corroborations and a material adverse P&L move;
    * headline/news catalysts may not close a current winner.
    """
    catalyst = request.catalyst_type.strip().lower()
    if catalyst not in ALLOWED_CATALYST_TYPES:
        return ExitRequestGateDecision(False, f"unknown catalyst_type: {request.catalyst_type}")
    if request.confidence < MIN_CONFIDENCE:
        return ExitRequestGateDecision(
            False,
            f"confidence {request.confidence:.2f} below floor {MIN_CONFIDENCE:.2f}",
        )
    if request.today_position_requests >= MAX_CLOSES_PER_POSITION_PER_DAY:
        return ExitRequestGateDecision(False, "daily cap: ≤1 close/position/day")
    if request.today_portfolio_requests >= MAX_CLOSES_PER_PORTFOLIO_PER_DAY:
        return ExitRequestGateDecision(False, "daily cap: ≤2 closes/portfolio/day")

    deterministic = quote_state.deterministic_exit_reason
    if deterministic:
        return ExitRequestGateDecision(
            True, f"deterministic exit already triggered: {deterministic}"
        )

    if len(request.sources) < 2:
        return ExitRequestGateDecision(False, "need at least 2 corroborating sources")

    if quote_state.current_net is None:
        return ExitRequestGateDecision(
            False, "deterministic HOLD and no live quote P&L corroboration"
        )

    basis = abs(quote_state.entry_net)
    if basis <= 0:
        return ExitRequestGateDecision(False, "entry basis unavailable for request_exit gate")
    pnl = quote_state.entry_net - quote_state.current_net
    if catalyst in HEADLINE_CATALYST_TYPES and pnl > 0:
        return ExitRequestGateDecision(False, "never close a winner on a headline/news catalyst")

    adverse_threshold = ADVERSE_LOSS_FRACTION * basis
    if pnl > -adverse_threshold:
        return ExitRequestGateDecision(
            False,
            "deterministic HOLD and adverse move is below request_exit threshold "
            f"({pnl:+.2f} vs -{adverse_threshold:.2f})",
        )
    return ExitRequestGateDecision(
        True,
        f"corroborated request_exit with adverse P&L move ({pnl:+.2f} <= -{adverse_threshold:.2f})",
    )
