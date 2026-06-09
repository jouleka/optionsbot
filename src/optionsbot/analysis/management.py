"""Position-management triggers (IBK-113).

Pure evaluation of open positions into management alerts: manage-by DTE (a short
leg approaching expiry) and assignment risk (a short leg in-the-money). No I/O --
``daemon/manage_runner`` does the fetching, dedup, and Telegram send. Rendering
lives in ``alerts/formatter.format_management_alert``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from optionsbot.analysis.positions import position_dte
from optionsbot.config import ManageSettings
from optionsbot.ibkr.types import PortfolioPosition


@dataclass(frozen=True, slots=True)
class ManagementAlert:
    symbol: str
    expiry: str
    strike: float
    right: str  # 'C' / 'P'
    quantity: float  # signed; short < 0, long > 0
    triggers: tuple[str, ...]  # sorted, non-empty subset of: dte_manage, dte_urgent, assignment
    dte: int | None
    spot: float | None
    itm: bool | None  # moneyness when spot known (drives wording); None if spot unknown
    dedup_key: str


def _dedup_key(
    symbol: str, expiry: str, strike: float, right: str, triggers: tuple[str, ...]
) -> str:
    return f"{symbol}:{expiry}:{strike:g}:{right}:{'+'.join(triggers)}"


def _is_itm(right: str, spot: float, strike: float) -> bool:
    return spot < strike if right == "P" else spot > strike


def evaluate_position_triggers(
    positions: list[PortfolioPosition],
    spots: dict[str, float],
    today: date,
    settings: ManageSettings,
) -> list[ManagementAlert]:
    """One management alert per option leg, carrying the SET of firing triggers (IBK-119).

    DTE bucket (most-severe only, lower-bounded at 0 so an expired-but-held leg doesn't
    re-fire forever) applies to SHORT and LONG legs. Assignment (SHORT ITM, DTE-independent --
    early assignment can happen anytime, esp. around ex-dividend) is short-only. Long legs are
    gated by ``long_leg_expiry_alerts``; ITM never adds a trigger for a long, it only sets
    ``itm`` for wording. Non-options and zero-quantity legs are ignored. Pure."""
    out: list[ManagementAlert] = []
    for p in positions:
        if p.sec_type != "OPT":
            continue
        expiry, strike, right = p.expiry, p.strike, p.right
        if expiry is None or strike is None or right is None:
            continue
        is_short = p.position < 0
        is_long = p.position > 0
        if not (is_short or is_long):
            continue
        if is_long and not settings.long_leg_expiry_alerts:
            continue
        dte = position_dte(expiry, today)
        spot = spots.get(p.symbol)
        itm = _is_itm(right, spot, strike) if spot is not None else None
        triggers: list[str] = []
        if dte is not None and 0 <= dte <= settings.urgent_dte:
            triggers.append("dte_urgent")
        elif dte is not None and 0 <= dte <= settings.manage_dte:
            triggers.append("dte_manage")
        if is_short and settings.assignment_alerts and itm:
            triggers.append("assignment")
        if not triggers:
            continue
        t = tuple(sorted(triggers))
        out.append(
            ManagementAlert(
                symbol=p.symbol, expiry=expiry, strike=strike, right=right,
                quantity=p.position, triggers=t, dte=dte, spot=spot, itm=itm,
                dedup_key=_dedup_key(p.symbol, expiry, strike, right, t),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class ProfitAlert:
    symbol: str
    trigger: str  # 'take_profit' | 'stop_loss'
    basis: str  # 'credit' | 'debit'
    base_amount: float  # net credit received / net debit paid (positive $)
    net_pnl: float
    profit_pct: float  # net_pnl / base_amount
    dedup_key: str


def evaluate_profit_triggers(
    positions: list[PortfolioPosition], settings: ManageSettings
) -> list[ProfitAlert]:
    """Per-underlying take-profit / stop-loss. Pure.

    ``net_signed = sum(avg_cost * |position| * (+1 short / -1 long))`` over the underlying's
    option legs (stock excluded): > 0 is a net credit (base = credit; take/stop scale off
    ``take_profit_pct`` / ``stop_loss_mult``), < 0 is a net debit (base = debit paid; scale off
    ``debit_take_profit_pct`` / ``debit_stop_pct``). ``net_pnl = sum(unrealized_pnl)``; realized
    P&L from a partial close is excluded, so profit_pct is remaining-position-relative. Groups
    below the relevant floor (``min_credit`` / ``min_debit``) and exactly-balanced (net 0)
    groups are skipped. Returns ``[]`` when ``profit_alerts`` is disabled."""
    if not settings.profit_alerts:
        return []
    by_symbol: dict[str, list[PortfolioPosition]] = {}
    for p in positions:
        if p.sec_type == "OPT" and p.right is not None and p.position != 0:
            by_symbol.setdefault(p.symbol, []).append(p)
    out: list[ProfitAlert] = []
    for symbol in sorted(by_symbol):
        legs = by_symbol[symbol]
        net_signed = sum(
            p.avg_cost * abs(p.position) * (1.0 if p.position < 0 else -1.0) for p in legs
        )
        if net_signed > 0.0:
            basis, base = "credit", net_signed
            take, stop = settings.take_profit_pct * base, settings.stop_loss_mult * base
            floor = settings.min_credit
        elif net_signed < 0.0:
            basis, base = "debit", -net_signed
            take, stop = settings.debit_take_profit_pct * base, settings.debit_stop_pct * base
            floor = settings.min_debit
        else:
            continue
        if base < floor:
            continue
        # Unrealized only: realized P&L from any partial close is excluded, so profit_pct is
        # "% captured on what's still open" (remaining-position-relative).
        net_pnl = sum(p.unrealized_pnl or 0.0 for p in legs)
        trigger: str | None = None
        if net_pnl >= take:
            trigger = "take_profit"
        elif net_pnl <= -stop:
            trigger = "stop_loss"
        if trigger is not None:
            out.append(
                ProfitAlert(
                    symbol=symbol, trigger=trigger, basis=basis, base_amount=base,
                    net_pnl=net_pnl, profit_pct=net_pnl / base,
                    dedup_key=f"{symbol}:profit:{trigger}",
                )
            )
    return out
