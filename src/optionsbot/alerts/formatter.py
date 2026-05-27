"""Markdown alert formatter (IBK-66).

Pure function that converts a (symbol, MarketView, ScoredStrategy, snapshot_ts)
into a MarkdownV2-safe string suitable for Telegram sendMessage. Defined-risk
strategies render plainly; undefined-risk (Short Straddle, Short Strangle)
get a prominent "UNDEFINED RISK" warning so the reader is reminded each
time.

MarkdownV2 escaping strategy: we escape interpolated *values* (symbol, direction,
regime, strategy name, rationale) but leave code-span content (inside backticks)
and the markdown syntax characters themselves unescaped. This keeps numbers like
1.25 readable while still being valid MarkdownV2.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from optionsbot.analysis.types import MarketView
from optionsbot.scoring import ScoredStrategy
from optionsbot.strategies import Leg

# MarkdownV2 special chars that need escaping in plain text per Telegram docs.
# Inside `code` spans and inside *bold*/_italic_ the rules differ, but we
# apply this only to plain-text user-supplied values.
_MD_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def _md_escape(text: str) -> str:
    """Escape a plain-text value for use in MarkdownV2 outside of code spans."""
    return "".join("\\" + c if c in _MD_ESCAPE_CHARS else c for c in text)


def _format_legs(legs: Iterable[Leg]) -> str:
    parts: list[str] = []
    for leg in legs:
        if leg.sec_type == "STK":
            parts.append(f"  • {leg.side} {leg.quantity} shares {leg.symbol}")
            continue
        strike = f"{leg.strike:g}" if leg.strike is not None else "?"
        right = leg.right or "?"
        expiry = leg.expiry or "?"
        parts.append(
            f"  • {leg.side} {expiry} {strike}{right}"
        )
    return "\n".join(parts)


def format_alert_markdown(
    symbol: str,
    view: MarketView,
    scored: ScoredStrategy,
    snapshot_ts: datetime,
) -> str:
    """Return a MarkdownV2-escaped Telegram message for one scored strategy."""
    sug = scored.suggestion
    lines: list[str] = []

    # Undefined-risk warning (plain text — ⚠ is safe, UNDEFINED RISK contains no special chars)
    if not sug.defined_risk:
        lines.append("⚠ *UNDEFINED RISK*")

    # Header line: bold symbol, plain strategy name (underscores safe in plain text between bold),
    # score in a code span (no escaping needed inside backticks).
    lines.append(f"*{symbol}* — {scored.strategy_name} score `{scored.score:.1f}`")

    # View line: direction/regime are plain text strings (no special chars expected),
    # iv_rank_value formatted as decimal inside parentheses — parens need escaping.
    if view.iv_rank_value is not None:
        lines.append(
            f"view: {view.direction}/{view.iv_regime} \\(rank {view.iv_rank_value:.2f}\\)"
        )
    else:
        lines.append(f"view: {view.direction}/{view.iv_regime}")

    lines.append("")
    lines.append("legs:")
    lines.append(_format_legs(sug.legs))
    lines.append("")

    # Financial values in code spans — no escaping needed inside backticks.
    kind = "credit" if sug.credit_or_debit > 0 else "debit"
    lines.append(f"net {kind} `${abs(sug.credit_or_debit):.2f}`")
    if sug.max_loss is not None:
        lines.append(f"max loss `${sug.max_loss:.2f}`")
    if sug.prob_profit is not None:
        lines.append(f"prob profit `{sug.prob_profit * 100:.0f}%`")
    if sug.suggested_quantity > 0:
        lines.append(f"size `{sug.suggested_quantity}` contracts")

    lines.append("")
    # Rationale: escape user-supplied text that may contain special chars.
    lines.append(_md_escape(scored.rationale))
    lines.append("")
    # Snapshot timestamp in italic — escape the timestamp string (contains - and :).
    lines.append(f"_snapshot {snapshot_ts.isoformat()}_")

    return "\n".join(lines)
