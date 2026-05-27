"""Six per-strategy factor calculators.

Each factor is a pure function of a :class:`FactorContext` that returns a
``float`` in ``[0.0, 1.0]``. Higher = better for the strategy being scored.
Missing inputs (``None``) return ``0.5`` (neutral) unless a factor has a
domain-specific better default.
"""

from __future__ import annotations

from datetime import date, datetime

from optionsbot.scoring.types import FactorContext


def iv_rank_score(ctx: FactorContext) -> float:
    """High IV rank is GOOD for short-premium plays and BAD for long-premium.

    Returns ``0.5`` neutral when ``snapshot.iv_rank`` is None (no history yet).
    """
    rank = ctx.snapshot.iv_rank
    if rank is None:
        return 0.5
    if ctx.strategy.long_premium:
        return 1.0 - rank
    return rank


def iv_hv_score(ctx: FactorContext) -> float:
    """IV/HV ratio: above 1.0 means IV overprices realized vol (good for sellers).

    Maps the ratio of ``0.5 -> 0.0``, ``1.0 -> 0.5``, ``1.5 -> 1.0`` and clips
    outside that range. Inverts the result for ``long_premium`` strategies.
    """
    iv = ctx.snapshot.atm_iv
    hv = ctx.snapshot.hv20
    if iv is None or hv is None or hv == 0.0:
        return 0.5
    ratio = iv / hv
    score = (ratio - 0.5) / 1.0
    score = max(0.0, min(1.0, score))
    return 1.0 - score if ctx.strategy.long_premium else score


def liquidity_score(ctx: FactorContext) -> float:
    """Average per-option-leg liquidity score: bid-ask tightness + open interest.

    Stock legs (``sec_type == "STK"``) are skipped. Returns ``0.0`` when the
    suggestion has no option legs at all (so a stock-only strategy defers to
    the remaining factors via its weight on this term).
    """
    chain_by_key = {
        (leg.expiry, leg.strike, leg.right): leg
        for leg in ctx.snapshot.chain
    }
    leg_scores: list[float] = []
    for leg in ctx.suggestion.legs:
        if leg.sec_type != "OPT":
            continue
        # An OPT leg must have these populated; guard for mypy + safety.
        if leg.expiry is None or leg.strike is None or leg.right is None:
            leg_scores.append(0.0)
            continue
        chain_leg = chain_by_key.get((leg.expiry, leg.strike, leg.right))
        if chain_leg is None or chain_leg.bid is None or chain_leg.ask is None:
            leg_scores.append(0.0)
            continue
        spread = chain_leg.ask - chain_leg.bid
        mid = (chain_leg.bid + chain_leg.ask) / 2.0
        # spread_pct: 0% -> spread_score 1.0, 10%+ -> 0.0 (clipped linear)
        if mid <= 0:
            spread_score = 0.0
        else:
            spread_pct = spread / mid
            spread_score = max(0.0, min(1.0, 1.0 - spread_pct / 0.10))
        # OI: 0 -> 0.0, 500+ -> 1.0
        oi = chain_leg.open_interest or 0
        oi_score = min(1.0, oi / 500.0)
        leg_scores.append((spread_score + oi_score) / 2.0)
    if not leg_scores:
        return 0.0
    return sum(leg_scores) / len(leg_scores)


def _dte(expiry: str, today: date | None = None) -> int:
    today = today or date.today()
    return (datetime.strptime(expiry, "%Y%m%d").date() - today).days


def dte_match_score(ctx: FactorContext) -> float:
    """Linear decay around ``snapshot.dte_target``: 0 delta -> 1.0, 30+ -> 0.0."""
    target = ctx.snapshot.dte_target
    expiries = {
        leg.expiry for leg in ctx.suggestion.legs if leg.expiry is not None
    }
    if not expiries:
        return 0.5  # no option legs (e.g., pure stock) -> neutral
    deltas = [abs(_dte(e) - target) for e in expiries]
    closest = min(deltas)
    return max(0.0, 1.0 - closest / 30.0)


def earnings_penalty(ctx: FactorContext) -> float:
    """1.0 when no earnings in window; otherwise 0.0 for shorts, 1.0 for longs."""
    in_window = ctx.snapshot.view.earnings_in_window
    if not in_window:
        return 1.0
    return 1.0 if ctx.strategy.long_premium else 0.0


def range_bound_score(ctx: FactorContext) -> float:
    """Heuristic from :attr:`MarketView.direction` + ``direction_strength``.

    Neutral + weak trend = textbook range-bound (1.0). Strong directional view
    is the worst case for premium-selling neutrals (0.0).
    """
    view = ctx.snapshot.view
    if view.direction == "neutral" and view.direction_strength == "weak":
        return 1.0
    if view.direction == "neutral" and view.direction_strength == "strong":
        return 0.5
    return 0.3 if view.direction_strength == "weak" else 0.0
