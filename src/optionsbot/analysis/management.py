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
    quantity: float  # signed; short -> negative
    trigger: str  # 'dte_manage' | 'dte_urgent' | 'assignment'
    dte: int | None
    spot: float | None
    dedup_key: str


def _dedup_key(symbol: str, expiry: str, strike: float, right: str, trigger: str) -> str:
    return f"{symbol}:{expiry}:{strike:g}:{right}:{trigger}"


def _is_itm(right: str, spot: float, strike: float) -> bool:
    return spot < strike if right == "P" else spot > strike


def evaluate_position_triggers(
    positions: list[PortfolioPosition],
    spots: dict[str, float],
    today: date,
    settings: ManageSettings,
) -> list[ManagementAlert]:
    """Management alerts for SHORT option legs: a DTE bucket (most severe only, and
    only for not-yet-expired legs) and, when the underlying spot is known, an
    assignment-risk alert if the short leg is ITM. Assignment is intentionally
    DTE-independent (early assignment can happen anytime, esp. around ex-dividend);
    near-expiry urgency is conveyed by the co-firing DTE alert. Long legs and
    non-options are ignored. Pure."""
    out: list[ManagementAlert] = []
    for p in positions:
        if p.sec_type != "OPT" or p.position >= 0:
            continue
        expiry, strike, right = p.expiry, p.strike, p.right
        if expiry is None or strike is None or right is None:
            continue
        dte = position_dte(expiry, today)
        trigger: str | None = None
        # Lower-bound at 0: an expired-but-still-held short (negative DTE) is not an
        # "approaching expiry" alert -- without this it would re-fire dte_urgent forever.
        if dte is not None and 0 <= dte <= settings.urgent_dte:
            trigger = "dte_urgent"
        elif dte is not None and 0 <= dte <= settings.manage_dte:
            trigger = "dte_manage"
        if trigger is not None:
            out.append(
                ManagementAlert(
                    symbol=p.symbol, expiry=expiry, strike=strike, right=right,
                    quantity=p.position, trigger=trigger, dte=dte, spot=None,
                    dedup_key=_dedup_key(p.symbol, expiry, strike, right, trigger),
                )
            )
        if settings.assignment_alerts:
            spot = spots.get(p.symbol)
            if spot is not None and _is_itm(right, spot, strike):
                out.append(
                    ManagementAlert(
                        symbol=p.symbol, expiry=expiry, strike=strike, right=right,
                        quantity=p.position, trigger="assignment", dte=dte, spot=spot,
                        dedup_key=_dedup_key(p.symbol, expiry, strike, right, "assignment"),
                    )
                )
    return out
