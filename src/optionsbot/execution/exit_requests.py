"""Daemon-side gates for Hermes-requested exits (IBK-138).

Hermes may ask the bot to consider a close because of news/catalyst context,
but news is only an input. This module keeps the trading-soundness policy as
pure code so MCP, daemon, and tests agree on the refusal reasons.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True, slots=True)
class HermesLossCapDecision:
    allowed: bool
    evaluable: bool
    cumulative_realized_pnl: float
    cap_dollars: float | None
    reason: str


def evaluate_hermes_loss_cap(
    *,
    cumulative_realized_pnl: float,
    day_start_net_liq: float | None,
    max_daily_loss_pct: float,
) -> HermesLossCapDecision:
    """Fail closed when the daily Hermes-driven realized-loss cap is unavailable or breached."""
    numeric_inputs = (
        cumulative_realized_pnl,
        max_daily_loss_pct,
        *(() if day_start_net_liq is None else (day_start_net_liq,)),
    )
    if any(not math.isfinite(value) for value in numeric_inputs):
        return HermesLossCapDecision(
            allowed=False,
            evaluable=False,
            cumulative_realized_pnl=cumulative_realized_pnl,
            cap_dollars=None,
            reason="non-finite input for Hermes loss cap",
        )
    if day_start_net_liq is None or day_start_net_liq <= 0 or max_daily_loss_pct <= 0:
        return HermesLossCapDecision(
            allowed=False,
            evaluable=False,
            cumulative_realized_pnl=cumulative_realized_pnl,
            cap_dollars=None,
            reason="current-session net-liq baseline unavailable for Hermes loss cap",
        )
    cap_dollars = day_start_net_liq * max_daily_loss_pct
    if cumulative_realized_pnl <= -cap_dollars:
        return HermesLossCapDecision(
            allowed=False,
            evaluable=True,
            cumulative_realized_pnl=cumulative_realized_pnl,
            cap_dollars=cap_dollars,
            reason=(
                "daily cumulative Hermes realized-loss cap breached "
                f"(${cumulative_realized_pnl:,.2f} <= -${cap_dollars:,.2f})"
            ),
        )
    return HermesLossCapDecision(
        allowed=True,
        evaluable=True,
        cumulative_realized_pnl=cumulative_realized_pnl,
        cap_dollars=cap_dollars,
        reason=(
            "daily cumulative Hermes realized P&L remains above cap "
            f"(${cumulative_realized_pnl:,.2f} > -${cap_dollars:,.2f})"
        ),
    )


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
    if not math.isfinite(request.confidence) or not 0.0 <= request.confidence <= 1.0:
        return ExitRequestGateDecision(
            False, "confidence evidence must be finite and within [0, 1]"
        )
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

    if (
        not isinstance(request.sources, list)
        or len(request.sources) < 2
        or any(not isinstance(source, str) or not source.strip() for source in request.sources)
        or len({source.strip() for source in request.sources}) != len(request.sources)
    ):
        return ExitRequestGateDecision(
            False, "need at least 2 distinct non-empty corroborating sources"
        )

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
