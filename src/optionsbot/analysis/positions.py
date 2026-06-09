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

from optionsbot.analysis.beta_weighting import beta, beta_weighted_delta
from optionsbot.ibkr.types import OptionQuote, PortfolioPosition

if TYPE_CHECKING:
    import pandas as pd

    from optionsbot.ibkr.history import HistoryClient
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


def portfolio_greeks(
    positions: list[PortfolioPosition], greeks: dict[GreeksKey, OptionQuote]
) -> dict[str, Any]:
    """Book-wide net Greeks by position-weighting (``greek * position * multiplier``).

    Delta includes stock legs (1 delta/share); gamma/theta/vega are option-only. An option
    leg whose Greeks didn't fetch (absent, or ``delta is None``) is excluded from the sums
    and counted as missing. Coverage (``complete``) is delta-gated: a leg whose delta
    computed but whose theta/vega did not still counts as covered, so ``net_theta`` /
    ``net_vega`` may silently omit it (rare -- IBKR model Greeks normally arrive together,
    and the -2 'not computed' sentinel is mapped to None in market_data). ``net_delta`` /
    ``net_gamma`` are share-equivalents (meaningful per underlying; unit-mixed across
    different underlyings -- see spec); ``net_theta`` is $/day and ``net_vega`` $/vol-point,
    which sum cleanly book-wide. Pure."""
    net_delta = net_gamma = net_theta = net_vega = 0.0
    option_total = 0
    option_with = 0
    for p in positions:
        if p.sec_type == "OPT":
            if p.expiry is None or p.strike is None or p.right is None:
                continue
            option_total += 1
            q = greeks.get((p.symbol, p.expiry, p.strike, p.right))
            if q is None or q.delta is None:
                continue
            option_with += 1
            scale = p.position * p.multiplier
            net_delta += q.delta * scale
            if q.gamma is not None:
                net_gamma += q.gamma * scale
            if q.theta is not None:
                net_theta += q.theta * scale
            if q.vega is not None:
                net_vega += q.vega * scale
        elif p.sec_type == "STK":
            net_delta += p.position * p.multiplier  # 1 delta/share; equity multiplier is 1
    return {
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_theta": net_theta,
        "net_vega": net_vega,
        "option_legs_total": option_total,
        "option_legs_with_greeks": option_with,
        "complete": option_with == option_total,
    }


def per_underlying_share_delta(
    positions: list[PortfolioPosition], greeks: dict[GreeksKey, OptionQuote]
) -> dict[str, float]:
    """Net share-delta per underlying (``delta * position * multiplier`` for delta-bearing
    option legs + ``position`` for stock). Same delta logic as ``portfolio_greeks`` but kept
    per-symbol for beta-weighting (IBK-118). Every underlying in ``positions`` gets an entry
    (0.0 if no delta-bearing legs). Pure."""
    out: dict[str, float] = {}
    for p in positions:
        out.setdefault(p.symbol, 0.0)
        if p.sec_type == "OPT":
            if p.expiry is None or p.strike is None or p.right is None:
                continue
            q = greeks.get((p.symbol, p.expiry, p.strike, p.right))
            if q is None or q.delta is None:
                continue
            out[p.symbol] += q.delta * p.position * p.multiplier
        elif p.sec_type == "STK":
            out[p.symbol] += p.position * p.multiplier
    return out


def build_positions_view(
    positions: list[PortfolioPosition],
    greeks: dict[GreeksKey, OptionQuote],
    as_of: datetime,
) -> dict[str, Any]:
    """Group legs by underlying with per-underlying + grand-total net unrealized P&L,
    DTE and Greeks per option leg. Pure. ``as_of`` is a tz-aware timestamp; legs sort
    DTE-ascending (None last), then strike; groups sort by underlying.

    Grouping is by ``symbol`` only; if ``positions`` ever spans multiple accounts the
    same ticker would merge across them (fine for the single-account setup today)."""
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
        "portfolio_greeks": portfolio_greeks(positions, greeks),
    }


async def _safe_spot(market_data_client: MarketDataClient, symbol: str) -> float | None:
    try:
        q = await market_data_client.get_stock_snapshot(symbol)
    except Exception:  # noqa: BLE001 -- best-effort; underlying excluded from weighting
        log.exception("beta-weight: spot fetch failed for %s", symbol)
        return None
    return q.last if q.last is not None else q.mid


async def _safe_beta(
    history_client: HistoryClient, symbol: str, benchmark_bars: pd.DataFrame, window: int
) -> float | None:
    try:
        bars = await history_client.get_history(symbol, days=window + 1)
    except Exception:  # noqa: BLE001 -- best-effort
        log.exception("beta-weight: history fetch failed for %s", symbol)
        return None
    return beta(bars, benchmark_bars, window)


async def _beta_weight_book(
    positions: list[PortfolioPosition],
    greeks: dict[GreeksKey, OptionQuote],
    market_data_client: MarketDataClient,
    history_client: HistoryClient,
    benchmark_symbol: str,
    beta_window: int,
) -> dict[str, Any] | None:
    """Best-effort beta-weighted book delta. None if benchmark history can't be fetched
    (nothing can be weighted then). Per-underlying spot/history failures exclude that
    underlying and ding coverage -- never crash, never fake beta. ``symbol == benchmark``
    short-circuits to beta=1.0 (no extra fetch)."""
    try:
        benchmark_bars = await history_client.get_history(
            benchmark_symbol, days=beta_window + 1
        )
    except Exception:  # noqa: BLE001
        log.exception("beta-weight: benchmark history fetch failed for %s", benchmark_symbol)
        return None
    benchmark_spot = await _safe_spot(market_data_client, benchmark_symbol)
    rows: list[dict[str, Any]] = []
    for symbol, share_delta in per_underlying_share_delta(positions, greeks).items():
        if not share_delta:  # delta-neutral / unknown -> not weightable; skip the I/O
            continue
        if symbol == benchmark_symbol:
            b: float | None = 1.0
            spot = benchmark_spot
        else:
            b = await _safe_beta(history_client, symbol, benchmark_bars, beta_window)
            spot = await _safe_spot(market_data_client, symbol)
        rows.append({"symbol": symbol, "share_delta": share_delta, "spot": spot, "beta": b})
    result = beta_weighted_delta(rows, benchmark_spot)
    result["benchmark"] = benchmark_symbol
    return result


async def assemble_open_book(
    positions_client: PositionsClient,
    market_data_client: MarketDataClient,
    as_of: datetime,
    *,
    history_client: HistoryClient | None = None,
    benchmark_symbol: str = "SPY",
    beta_window: int = 252,
) -> dict[str, Any]:
    """Fetch the enriched portfolio + best-effort per-option Greeks, build the view.

    A Greeks-fetch failure for one leg is swallowed (that leg shows P&L, no Greeks);
    the portfolio fetch itself is allowed to propagate (callers map it to a structured
    'IBKR unavailable' response).

    When ``history_client`` is provided, a best-effort beta-weighted delta (IBK-118) is
    attached as ``view["beta_weighted"]`` (None if the benchmark history can't be fetched).
    Omitting it leaves the view unchanged -- the pre-IBK-118 behavior."""
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
    view = build_positions_view(positions, greeks, as_of)
    if history_client is not None:
        view["beta_weighted"] = await _beta_weight_book(
            positions, greeks, market_data_client, history_client, benchmark_symbol, beta_window
        )
    return view
