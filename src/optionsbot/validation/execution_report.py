"""Execution-aware track record (IBK-131).

The third truth-track alongside the hypothetical pick_outcomes ledger and
the live open book: REALIZED round-trips from the bot's own fills, with
commissions, plus entry slippage measured against the decision-quote
journal (realized per-unit premium vs the combo mid the bot acted on).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine, select

from optionsbot.execution.orders import (
    ClosedPair,
    net_premium,
    realized_close_pairs,
)
from optionsbot.storage.schema import order_quotes, orders

# Below this many closed trades, a win rate is noise: with 20 trades a true
# 55% win rate is observed anywhere between roughly 35% and 75%.
_MIN_SAMPLE = 100


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    closed: int
    wins: int
    win_rate: float | None
    total_pnl: float
    avg_pnl: float | None
    total_commissions: float
    by_strategy: dict[str, tuple[int, float]]  # strategy -> (count, pnl)
    mean_entry_slippage: float | None  # per-unit, positive = worse than decision mid
    abandoned: int
    skipped: int
    sample_warning: str | None
    pairs: tuple[ClosedPair, ...]


def _entry_slippage(engine: Engine, pair: ClosedPair) -> float | None:
    """decision combo mid − realized per-unit premium (entries only).

    For a credit entry, positive slippage = collected less than the mid the
    bot decided on; for a debit entry the signed math works out the same
    (paid more than mid → realized net more negative → positive slippage).
    """
    with engine.connect() as conn:
        row = conn.execute(
            select(order_quotes.c.combo_mid)
            .where(order_quotes.c.order_id == pair.entry_id)
            .where(order_quotes.c.kind == "decision")
            .limit(1)
        ).first()
    if row is None or row.combo_mid is None:
        return None
    premium = net_premium(engine, pair.entry_id)
    if premium is None or pair.quantity < 1:
        return None
    realized_unit = premium / (100 * pair.quantity)
    return float(row.combo_mid) - realized_unit


def execution_report(engine: Engine) -> ExecutionReport:
    pairs = realized_close_pairs(engine)
    closed = len(pairs)
    wins = sum(1 for p in pairs if p.pnl > 0)
    total_pnl = sum(p.pnl for p in pairs)

    from optionsbot.execution.orders import total_commissions as _commissions

    commissions = sum(
        _commissions(engine, p.entry_id)
        + (_commissions(engine, p.close_id) if p.close_id is not None else 0.0)
        for p in pairs
    )
    by_strategy: dict[str, tuple[int, float]] = {}
    for pair in pairs:
        count, pnl = by_strategy.get(pair.strategy, (0, 0.0))
        by_strategy[pair.strategy] = (count + 1, pnl + pair.pnl)

    slippages = [s for p in pairs if (s := _entry_slippage(engine, p)) is not None]
    mean_slippage = sum(slippages) / len(slippages) if slippages else None

    with engine.connect() as conn:
        abandoned = len(conn.execute(
            select(orders.c.id).where(orders.c.status == "abandoned")
        ).fetchall())
        skipped = len(conn.execute(
            select(orders.c.id).where(orders.c.status == "skipped")
        ).fetchall())

    warning = None
    if closed < _MIN_SAMPLE:
        warning = (
            f"⚠ {closed} closed trade(s) — far below the ~{_MIN_SAMPLE} needed "
            "for a statistically meaningful win rate. Treat as anecdote."
        )
    return ExecutionReport(
        closed=closed,
        wins=wins,
        win_rate=(wins / closed) if closed else None,
        total_pnl=total_pnl,
        avg_pnl=(total_pnl / closed) if closed else None,
        total_commissions=commissions,
        by_strategy=by_strategy,
        mean_entry_slippage=mean_slippage,
        abandoned=abandoned,
        skipped=skipped,
        sample_warning=warning,
        pairs=tuple(pairs),
    )


def format_execution_report(report: ExecutionReport) -> str:
    """Plain-text block appended to /record."""
    if report.closed == 0:
        return (
            "EXECUTED (real fills): no closed round-trips yet.\n"
            f"abandoned {report.abandoned} / gate-skipped {report.skipped} orders so far."
        )
    lines = [
        "EXECUTED (real fills, commissions included):",
        f"closed {report.closed} | wins {report.wins} "
        f"({(report.win_rate or 0) * 100:.0f}%) | "
        f"P&L ${report.total_pnl:,.0f} (avg ${report.avg_pnl or 0:,.0f})",
        f"commissions paid ${report.total_commissions:,.2f}",
    ]
    if report.mean_entry_slippage is not None:
        lines.append(
            f"mean entry slippage {report.mean_entry_slippage:+.2f}/unit vs decision mid"
        )
    for strategy, (count, pnl) in sorted(report.by_strategy.items()):
        lines.append(f"  {strategy}: {count} closed, ${pnl:,.0f}")
    lines.append(
        f"unfilled walks (abandoned): {report.abandoned} | gate-skipped: {report.skipped}"
    )
    if report.sample_warning:
        lines.append(report.sample_warning)
    return "\n".join(lines)
