"""Open-book position view (IBK-112).

Pure helpers (``position_dte``, ``build_positions_view``) plus a thin async
orchestrator (``assemble_open_book``) that fetches the enriched portfolio and
best-effort per-leg Greeks. Mirrors ``analysis/news.py``'s pure-plus-orchestrator
shape: the assembly is pure and unit-testable; the orchestrator does the I/O.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from optionsbot.ibkr.types import OptionQuote, PortfolioPosition

if TYPE_CHECKING:
    from optionsbot.ibkr.market_data import MarketDataClient
    from optionsbot.ibkr.positions import PositionsClient

log = logging.getLogger(__name__)

GreeksKey = tuple[str, str, float, str]  # (symbol, expiry, strike, right)


def position_dte(expiry: str | None, today: date) -> int | None:
    """Calendar days from ``today`` to ``expiry`` (YYYYMMDD). None if missing/unparseable."""
    if not expiry:
        return None
    try:
        exp = datetime.strptime(expiry, "%Y%m%d").date()
    except ValueError:
        return None
    return (exp - today).days


def _leg_dict(
    p: PortfolioPosition, greeks: dict[GreeksKey, OptionQuote], today: date
) -> dict[str, Any]:
    is_opt = p.sec_type == "OPT"
    leg: dict[str, Any] = {
        "sec_type": p.sec_type,
        "symbol": p.symbol,
        "quantity": p.position,
        "avg_cost": p.avg_cost,
        "market_price": p.market_price,
        "market_value": p.market_value,
        "unrealized_pnl": p.unrealized_pnl,
        "expiry": p.expiry,
        "strike": p.strike,
        "right": p.right,
        "dte": position_dte(p.expiry, today) if is_opt else None,
        "delta": None,
        "gamma": None,
        "theta": None,
        "vega": None,
        "iv": None,
    }
    if is_opt and p.expiry and p.strike is not None and p.right is not None:
        q = greeks.get((p.symbol, p.expiry, p.strike, p.right))
        if q is not None:
            leg["delta"], leg["gamma"], leg["theta"], leg["vega"], leg["iv"] = (
                q.delta, q.gamma, q.theta, q.vega, q.iv,
            )
    return leg


def build_positions_view(
    positions: list[PortfolioPosition],
    greeks: dict[GreeksKey, OptionQuote],
    as_of: datetime,
) -> dict[str, Any]:
    """Group legs by underlying with per-underlying + grand-total net unrealized P&L,
    DTE and Greeks per option leg. Pure. ``as_of`` is a tz-aware timestamp; legs sort
    DTE-ascending (None last), then strike; groups sort by underlying."""
    today = as_of.date()
    groups_map: dict[str, list[dict[str, Any]]] = {}
    for p in positions:
        groups_map.setdefault(p.symbol, []).append(_leg_dict(p, greeks, today))
    groups: list[dict[str, Any]] = []
    grand_total = 0.0
    for underlying in sorted(groups_map):
        legs = groups_map[underlying]
        legs.sort(key=lambda lg: (lg["dte"] is None, lg["dte"] or 0, lg["strike"] or 0.0))
        net = sum(lg["unrealized_pnl"] or 0.0 for lg in legs)
        grand_total += net
        groups.append({"underlying": underlying, "net_unrealized_pnl": net, "legs": legs})
    return {
        "as_of": as_of.isoformat(),
        "net_unrealized_pnl": grand_total,
        "group_count": len(groups),
        "position_count": len(positions),
        "groups": groups,
    }


async def assemble_open_book(
    positions_client: PositionsClient,
    market_data_client: MarketDataClient,
    as_of: datetime,
) -> dict[str, Any]:
    """Fetch the enriched portfolio + best-effort per-option Greeks, build the view.

    A Greeks-fetch failure for one leg is swallowed (that leg shows P&L, no Greeks);
    the portfolio fetch itself is allowed to propagate (callers map it to a structured
    'IBKR unavailable' response)."""
    positions = await positions_client.get_portfolio()
    greeks: dict[GreeksKey, OptionQuote] = {}
    for p in positions:
        if p.sec_type != "OPT" or not p.expiry or p.strike is None or p.right is None:
            continue
        try:
            q = await market_data_client.get_option_snapshot(
                p.symbol, p.expiry, p.strike, p.right
            )
        except Exception:  # noqa: BLE001 -- best-effort; leg keeps P&L, no Greeks
            log.exception(
                "greeks fetch failed for %s %s %s%s", p.symbol, p.expiry, p.strike, p.right
            )
            continue
        greeks[(p.symbol, p.expiry, p.strike, p.right)] = q
    return build_positions_view(positions, greeks, as_of)
