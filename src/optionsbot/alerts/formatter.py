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
from typing import Any

from optionsbot.analysis.management import ManagementAlert, ProfitAlert
from optionsbot.analysis.types import MarketView
from optionsbot.scoring import ScoredStrategy
from optionsbot.strategies import Leg
from optionsbot.validation.types import OutcomeGroup, OutcomesReport

# MarkdownV2 special chars that need escaping in plain text per Telegram docs.
# Inside `code` spans and inside *bold*/_italic_ the rules differ, but we
# apply this only to plain-text user-supplied values.
_MD_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def _md_escape(text: str) -> str:
    """Escape a plain-text value for use in MarkdownV2 outside of code spans."""
    return "".join("\\" + c if c in _MD_ESCAPE_CHARS else c for c in text)


def _format_legs(legs: Iterable[Leg]) -> str:
    # Wrap each leg's data in a backtick code span so fractional strikes
    # (e.g., 410.5 -> the "." char) and any future leg-data special chars
    # don't break MarkdownV2 parsing. Inside code spans, only ` and \
    # need escaping.
    parts: list[str] = []
    for leg in legs:
        if leg.sec_type == "STK":
            parts.append(f"  • `{leg.side} {leg.quantity} shares {leg.symbol}`")
            continue
        strike = f"{leg.strike:g}" if leg.strike is not None else "?"
        right = leg.right or "?"
        expiry = leg.expiry or "?"
        parts.append(
            f"  • `{leg.side} {expiry} {strike}{right}`"
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

    # Header line: bold symbol, strategy name in backticks (snake_case names
    # like iron_condor or bull_put_spread contain `_` which would otherwise
    # parse as italic markers outside a code span). The symbol is escaped
    # because dotted tickers (BRK.B, BF.B) contain a `.` which is special
    # even inside a bold span -- Telegram requires escaping all special
    # chars inside formatting spans except the delimiter and `\`.
    lines.append(f"*{_md_escape(symbol)}* — `{scored.strategy_name}` score `{scored.score:.1f}`")

    # View line: direction/regime are Literal values with no special chars.
    # iv_rank_value contains a `.` so it goes inside a code span; parens
    # around it are escaped because they live outside the backticks.
    if view.iv_rank_value is not None:
        lines.append(
            f"view: {view.direction}/{view.iv_regime} \\(rank `{view.iv_rank_value:.2f}`\\)"
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
    if sug.reward_risk is not None:
        lines.append(f"reward:risk `{sug.reward_risk:.2f}`")
    if sug.expected_value is not None:
        lines.append(f"exp value `${sug.expected_value:.2f}` \\(est\\)")
    lines.append(f"risk tier `{sug.risk_tier}`")
    if sug.suggested_quantity > 0:
        lines.append(f"size `{sug.suggested_quantity}` contracts")

    lines.append("")
    # Rationale: escape user-supplied text that may contain special chars.
    lines.append(_md_escape(scored.rationale))
    lines.append("")
    # Snapshot timestamp inside italic: even inside _..._ all special chars
    # except `_` and `\` still need escaping (Telegram MarkdownV2 rule).
    # The isoformat() output has `-`, `:`, `+`, `.` — all of which must be escaped.
    lines.append(f"_snapshot {_md_escape(snapshot_ts.isoformat())}_")

    return "\n".join(lines)


def no_edge_note(symbol: str) -> str:
    """Plain-text banner shown when no surfaced pick has positive expected value.

    Plain text (no MarkdownV2): callers send it as a ``parse_mode=None`` reply or
    a CLI echo. The ``⚠`` glyph is safe in both (used the same way for the
    UNDEFINED RISK warning above).
    """
    return (
        f"⚠ No positive-edge trade on {symbol} right now — option premium looks "
        f"fairly priced vs how much {symbol} actually moves. "
        f"Shown for reference, not a recommendation."
    )


def _money(x: float | None) -> str:
    if x is None:
        return "$?"
    if round(x) == 0:  # avoid "$-0" for tiny negatives / -0.0
        return "$0"
    return f"${x:+,.0f}"


def _greeks_footer(g: dict[str, Any]) -> str:
    cov = (
        ""
        if g.get("complete", True)
        else f"  ({g['option_legs_with_greeks']}/{g['option_legs_total']} legs)"
    )
    return (
        f"book greeks: Δ {round(g['net_delta']):+}  Θ {_money(g['net_theta'])}/day  "
        f"vega {_money(g['net_vega'])}  Γ {g['net_gamma']:+.1f}{cov}"
    )


def _beta_weighted_footer(bw: dict[str, Any]) -> str:
    bench = bw.get("benchmark", "SPY")
    if bw["underlyings_total"] == 0:
        return "β-wtd: n/a (no weightable positions)"  # delta-neutral / flat book
    if bw["underlyings_covered"] == 0:
        return "β-wtd: n/a (no beta available)"  # have exposure, no beta for any name
    cov = (
        ""
        if bw.get("complete", True)
        else f"  ({bw['underlyings_covered']}/{bw['underlyings_total']} underlyings)"
    )
    per1 = _money(bw["dollar_per_1pct_spy"])
    equiv = bw.get("spy_equiv_shares")
    if equiv is None:
        return f"β-wtd: {per1}/1% {bench}{cov}"
    return f"β-wtd: Δ≈ {equiv:+,.0f} {bench}-eq  {per1}/1% {bench}{cov}"


def _short_expiry(expiry: str | None) -> str:
    """20260717 -> 17Jul; pass through anything not an 8-char YYYYMMDD."""
    if not expiry or len(expiry) != 8:
        return expiry or "?"
    try:
        d = datetime.strptime(expiry, "%Y%m%d")
    except ValueError:
        return expiry
    return f"{d.day}{d.strftime('%b')}"


def _position_leg_line(lg: dict[str, Any]) -> str:
    if lg.get("sec_type") != "OPT":
        return f"{lg['quantity']:+g} shares | P&L {_money(lg.get('unrealized_pnl'))}"
    strike = f"{lg['strike']:g}" if lg.get("strike") is not None else "?"
    mid = lg.get("market_price")
    mids = f"{mid:.2f}" if mid is not None else "?"
    dte = lg.get("dte")
    delta = lg.get("delta")
    ds = f"  Δ{delta:+.2f}" if delta is not None else ""
    return (
        f"{lg['quantity']:+g} {_short_expiry(lg.get('expiry'))} {strike}{lg.get('right') or '?'} "
        f"| mid {mids}  P&L {_money(lg.get('unrealized_pnl'))}  "
        f"DTE {dte if dte is not None else '?'}{ds}"
    )


def format_positions_text(view: dict[str, Any]) -> str:
    """Plain-text open book for Telegram (parse_mode=None): grouped by underlying,
    per-leg P&L / DTE / delta, with a header net total. Empty book -> short notice."""
    groups = view.get("groups", [])
    if not groups:
        return "no open positions"
    n = view.get("group_count", len(groups))
    lines = [
        f"open book — net P&L {_money(view.get('net_unrealized_pnl'))} "
        f"({n} underlying{'s' if n != 1 else ''})"
    ]
    for g in groups:
        lines.append(f"{g['underlying']}  net {_money(g['net_unrealized_pnl'])}")
        lines += ["  " + _position_leg_line(lg) for lg in g["legs"]]
    pg = view.get("portfolio_greeks")
    if pg is not None:
        lines.append(_greeks_footer(pg))
    bw = view.get("beta_weighted")
    if bw is not None:
        lines.append(_beta_weighted_footer(bw))
    return "\n".join(lines)


def format_management_alert(alert: ManagementAlert) -> str:
    """Plain-text management alert for Telegram (parse_mode=None). One message per leg,
    rendering the full set of firing triggers (IBK-119)."""
    leg = f"{alert.quantity:+g} {_short_expiry(alert.expiry)} {alert.strike:g}{alert.right}"
    dte = f"{alert.dte} DTE" if alert.dte is not None else "expiry ?"
    if "dte_urgent" in alert.triggers:
        word = "URGENT"
    elif "dte_manage" in alert.triggers:
        word = "manage"
    else:
        word = "assignment risk"
    has_dte = "dte_urgent" in alert.triggers or "dte_manage" in alert.triggers
    side = "put" if alert.right == "P" else "call"
    rel = "<" if alert.right == "P" else ">"
    spot_s = f"{alert.spot:.2f}" if alert.spot is not None else "?"
    clauses: list[str] = []
    if alert.quantity < 0:  # short
        if "assignment" in alert.triggers:
            clauses.append(f"short {side} ITM (spot {spot_s} {rel} {alert.strike:g})")
            if has_dte:
                clauses.append("approaching expiry")
        else:
            clauses.append("short option approaching expiry")
    else:  # long -- always has a DTE trigger; ITM only changes the wording
        if alert.itm is True:
            clauses.append(f"long {side} ITM — auto-exercises at expiry; close if unwanted")
        elif alert.itm is False:
            clauses.append("long option approaching expiry — premium decaying; close/roll")
        else:
            clauses.append("long option approaching expiry — close/roll")
    return f"⚠ {word} {alert.symbol} {leg} — {dte}, {', '.join(clauses)}"


def format_profit_alert(alert: ProfitAlert) -> str:
    """Plain-text take-profit / stop-loss alert for Telegram (parse_mode=None)."""
    amt = f"${alert.base_amount:,.0f}"
    pnl = _money(alert.net_pnl)
    pct = round(alert.profit_pct * 100)
    if alert.basis == "credit":
        if alert.trigger == "take_profit":
            return f"✅ take profit {alert.symbol} — captured {pct}% of {amt} credit (P&L {pnl})"
        mult = abs(alert.net_pnl / alert.base_amount)
        return f"🛑 stop loss {alert.symbol} — {pnl}, {mult:.1f}x the {amt} credit"
    if alert.trigger == "take_profit":
        return f"✅ take profit {alert.symbol} — {pct:+}% on {amt} debit (P&L {pnl})"
    return f"🛑 stop loss {alert.symbol} — {pct:+}% of the {amt} debit (P&L {pnl})"


def _track_group_line(name: str, g: OutcomeGroup) -> str:
    return (
        f"  {name} n={g.count} win {g.win_rate:.2f} "
        f"pred {g.mean_pred_pop:.2f} avg {_money(g.avg_pnl)}"
    )


def format_track_record(report: OutcomesReport) -> str:
    """Plain-text realized track record for Telegram (parse_mode=None)."""
    o = report.overall
    if o.count == 0:
        return "no evaluated outcomes yet (picks resolve at expiry)"
    lines = [
        f"track record: {o.count} picks, win {o.win_rate:.2f} vs predicted "
        f"{o.mean_pred_pop:.2f}, P&L {_money(o.total_pnl)} (avg {_money(o.avg_pnl)})"
    ]
    if report.by_strategy:
        lines.append("by strategy:")
        lines += [_track_group_line(n, g) for n, g in report.by_strategy.items()]
    if report.by_risk_tier:
        lines.append("by risk tier:")
        lines += [_track_group_line(n, g) for n, g in report.by_risk_tier.items()]
    return "\n".join(lines)
