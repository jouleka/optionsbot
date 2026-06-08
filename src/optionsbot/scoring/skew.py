"""Skew-aware PoP/EV: terminal distribution from the per-strike IV smile (IBK-111).

Pure stdlib. Where :mod:`optionsbot.scoring.payoff` sizes the terminal-price
distribution with a single ATM IV (PoP) / flat realized vol (EV), this module builds
it from the chain's per-strike IV smile via the smile-implied risk-neutral CDF
``F(K) = Phi(-d2(K; iv(K)))``, discretised into price cells. A FLAT smile reproduces
``payoff.py``'s lognormal exactly, so this is a strict, graceful refinement:
``strategies.base`` uses it when a usable smile exists and falls back to ``payoff.py``
otherwise.

Single-expiry, all-option positions only (same gate as ``payoff.py``).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.scoring.payoff import (
    _STEPS,
    _Z,
    is_terminal_modelable,
    terminal_pnl_dollars,
)
from optionsbot.strategies.base import Leg

_SQRT2 = math.sqrt(2.0)


def _phi_cdf(x: float) -> float:
    """Standard-normal CDF via erf (pure stdlib)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _interp(points: tuple[tuple[float, float], ...], x: float) -> float | None:
    """Linear interpolation in strike with flat extrapolation past the ends.

    ``points`` is sorted by strike. Empty -> None.
    """
    if not points:
        return None
    if len(points) == 1:
        return points[0][1]
    strikes = [p[0] for p in points]
    if x <= strikes[0]:
        return points[0][1]
    if x >= strikes[-1]:
        return points[-1][1]
    i = bisect.bisect_left(strikes, x)
    k0, v0 = points[i - 1]
    k1, v1 = points[i]
    if k1 == k0:
        return v0
    w = (x - k0) / (k1 - k0)
    return v0 + w * (v1 - v0)


@dataclass(frozen=True)
class VolSmile:
    """Per-strike IV smile for a single expiry (OTM-wing convention).

    ``put_points`` / ``call_points`` are ``(strike, iv)`` tuples sorted by strike with
    ``iv > 0``. Below spot the put wing is read; at/above spot the call wing. The two
    join continuously near spot (ATM put-IV ~= ATM call-IV by put-call parity).
    """

    spot: float
    put_points: tuple[tuple[float, float], ...]
    call_points: tuple[tuple[float, float], ...]

    def iv_at(self, price: float) -> float | None:
        """IV at ``price``: put wing below spot, call wing at/above. Falls back to the
        other wing if the chosen one is empty; ``None`` if both are empty."""
        # Boundary deliberately on the call side: at price == spot, ATM put-IV ~=
        # ATM call-IV by put-call parity, so the two wings join continuously.
        if price < self.spot:
            primary, secondary = self.put_points, self.call_points
        else:
            primary, secondary = self.call_points, self.put_points
        v = _interp(primary, price)
        if v is None:
            v = _interp(secondary, price)
        return v

    def atm_iv(self) -> float | None:
        return self.iv_at(self.spot)

    def max_iv(self) -> float | None:
        """Largest IV across both wings. Sizes the integration grid so it brackets the
        fattest wing -- otherwise a high-IV wing's tail mass is truncated and the
        renormalization redistributes it directionally, biasing PoP/EV (IBK-111 S1)."""
        ivs = [iv for _, iv in self.put_points]
        ivs += [iv for _, iv in self.call_points]
        return max(ivs) if ivs else None


def build_smile(
    chain: Iterable[OptionChainLeg], expiry: str, spot: float
) -> VolSmile | None:
    """Build a :class:`VolSmile` from the chain legs at ``expiry``.

    Returns ``None`` when no usable IV exists (so callers fall back to the flat model)
    or when the ATM reference IV cannot resolve. A one-sided chain (only puts or only
    calls) is kept: ``iv_at`` flat-extrapolates the present wing across the missing
    side -- losing skew on that side, but a reasonable scanner-grade degradation.
    """
    puts: list[tuple[float, float]] = []
    calls: list[tuple[float, float]] = []
    for leg in chain:
        if leg.expiry != expiry or leg.iv is None or leg.iv <= 0.0:
            continue
        if leg.right == "P":
            puts.append((leg.strike, leg.iv))
        else:
            calls.append((leg.strike, leg.iv))
    if not puts and not calls:
        return None
    smile = VolSmile(
        spot=spot,
        put_points=tuple(sorted(puts)),
        call_points=tuple(sorted(calls)),
    )
    atm = smile.atm_iv()
    if atm is None or atm <= 0.0:
        return None
    return smile


def _smile_cells(
    spot: float,
    dte_days: float,
    sigma_at: Callable[[float], float | None],
    sigma_ref: float,
) -> list[tuple[float, float]] | None:
    """``[(mid_price, mass), ...]`` over a log-price grid.

    Cell mass is the smile-implied risk-neutral probability in that cell:
    ``F(K) = Phi((ln(K/spot) + 0.5 s^2) / s)`` with ``s = sigma(K) * sqrt(T)`` -- i.e.
    each grid edge is priced with its OWN local vol. Masses are unnormalized; the
    caller divides by their sum. The grid half-width is set by ``sigma_ref`` -- the
    smile's MAX vol, passed by the caller -- so the grid brackets the fattest wing;
    sizing it off ATM instead would truncate a high-IV wing's tail and the
    renormalization would redistribute that mass directionally, biasing PoP/EV.
    Returns ``None`` on degenerate inputs.
    """
    if spot <= 0.0 or dte_days <= 0.0:
        return None
    if sigma_ref <= 0.0:
        return None
    sqrt_t = math.sqrt(dte_days / 365.0)
    span = _Z * sigma_ref * sqrt_t
    log_spot = math.log(spot)

    def cdf(price: float) -> float | None:
        sigma = sigma_at(price)
        if sigma is None or sigma <= 0.0:
            return None
        s = sigma * sqrt_t
        if s <= 0.0:
            return None
        return _phi_cdf((math.log(price / spot) + 0.5 * s * s) / s)

    edges = [
        math.exp(log_spot - span + (2.0 * span) * (j / _STEPS))
        for j in range(_STEPS + 1)
    ]
    cdfs: list[float] = []
    for price in edges:
        c = cdf(price)
        if c is None:
            return None
        cdfs.append(c)
    cells: list[tuple[float, float]] = []
    for i in range(_STEPS):
        mass = cdfs[i + 1] - cdfs[i]
        if mass < 0.0:  # clamp minor non-monotonicity (soft no-arbitrage guard)
            mass = 0.0
        mid = math.sqrt(edges[i] * edges[i + 1])
        cells.append((mid, mass))
    return cells


def prob_of_profit_smile(
    legs: Iterable[Leg],
    credit_or_debit: float,
    spot: float,
    smile: VolSmile,
    dte_days: float,
) -> float | None:
    """P(P&L at expiry > 0) under the smile-implied terminal distribution.

    ``None`` for non-modelable positions (stock/multi-expiry) or degenerate inputs.
    """
    legs = tuple(legs)
    if not is_terminal_modelable(legs):
        return None
    sigma_ref = smile.max_iv()
    if sigma_ref is None:
        return None
    cells = _smile_cells(spot, dte_days, smile.iv_at, sigma_ref)
    if cells is None:
        return None
    total = math.fsum(m for _, m in cells)
    if total <= 0.0:
        return None
    w_profit = math.fsum(
        m for mid, m in cells if terminal_pnl_dollars(legs, credit_or_debit, mid) > 0.0
    )
    return w_profit / total


def expected_value_smile(
    legs: Iterable[Leg],
    credit_or_debit: float,
    spot: float,
    smile: VolSmile,
    realized_vol: float | None,
    dte_days: float,
) -> float | None:
    """E[P&L at expiry] in dollars under a terminal distribution whose LEVEL is the
    realized vol and whose SHAPE is the IV smile: ``sigma_EV(K) = realized * iv(K)/atm``.

    Preserves the volatility-risk-premium intent (the realized-vol anchor) while
    borrowing the smile's asymmetry, so a downside-skewed smile makes premium-selling
    EV more conservative -- never more optimistic. ``None`` for non-modelable positions,
    missing realized vol, or degenerate inputs.
    """
    legs = tuple(legs)
    if realized_vol is None or realized_vol <= 0.0:
        return None
    if not is_terminal_modelable(legs):
        return None
    atm = smile.atm_iv()
    if atm is None or atm <= 0.0:
        return None
    scale = realized_vol / atm
    max_iv = smile.max_iv()
    if max_iv is None:
        return None

    def sigma_at(price: float) -> float | None:
        v = smile.iv_at(price)
        return None if v is None else v * scale

    cells = _smile_cells(spot, dte_days, sigma_at, scale * max_iv)
    if cells is None:
        return None
    total = math.fsum(m for _, m in cells)
    if total <= 0.0:
        return None
    w_pnl = math.fsum(
        m * terminal_pnl_dollars(legs, credit_or_debit, mid) for mid, m in cells
    )
    return w_pnl / total
