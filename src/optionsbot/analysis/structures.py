"""Per-structure reconstruction of held legs (IBK-120).

Pure: name a single underlying's held legs as a recognizable option structure
(Bull Put Spread, Iron Condor, Covered Call, ...) or honest "custom (N legs)".
WHOLE-GROUP exact match -- a label is returned only when every leg matches one
signature exactly; any ambiguity degrades to custom. No partitioning. Display
annotation only -- NOT an input to scoring, P&L, Greeks, or alerts.
"""

from __future__ import annotations

from optionsbot.ibkr.types import PortfolioPosition

# An option leg reduced to what the matcher needs: (expiry, strike, right, sign)
# with sign +1 long / -1 short.
_OptLeg = tuple[str, float, str, int]


def _with_mult(label: str, m: int) -> str:
    return f"{label} ×{m}" if m > 1 else label


def _match_single(leg: _OptLeg) -> str:
    _, _, right, sign = leg
    side = "Long" if sign > 0 else "Short"
    kind = "Call" if right == "C" else "Put"
    return f"{side} {kind}"


def _match_with_stock(
    stock: list[PortfolioPosition], opts: list[PortfolioPosition], custom: str
) -> str:
    if len(stock) != 1:
        return custom
    shares = int(stock[0].position)
    if not opts:
        return "Long Stock" if shares > 0 else "Short Stock"
    return custom


def _match_two(a: _OptLeg, b: _OptLeg) -> str | None:
    ea, ka, ra, sa = a
    eb, kb, rb, sb = b
    same_expiry = ea == eb
    if same_expiry and ra == rb:  # vertical
        if sa == sb or ka == kb:  # need opposite signs + different strikes
            return None
        short = a if sa < 0 else b
        long_ = b if sa < 0 else a
        if ra == "P":
            return "Bull Put Spread" if short[1] > long_[1] else "Bear Put Spread"
        return "Bear Call Spread" if short[1] < long_[1] else "Bull Call Spread"
    if same_expiry and ra != rb:  # straddle / strangle
        if sa != sb:  # need equal signs
            return None
        same_strike = ka == kb
        if sa > 0:
            return "Long Straddle" if same_strike else "Long Strangle"
        return "Short Straddle" if same_strike else "Short Strangle"
    if not same_expiry and ra == rb:  # calendar / diagonal
        if sa == sb:  # need opposite signs
            return None
        return "Calendar Spread" if ka == kb else "Diagonal Spread"
    return None


def _match_options(opts: list[PortfolioPosition], custom: str) -> str:
    qmags = {abs(int(p.position)) for p in opts}
    if len(qmags) != 1:  # uneven copies / ratio -> not a clean structure
        return custom
    m = qmags.pop()
    reduced: list[_OptLeg] = []
    for p in opts:
        assert p.expiry is not None and p.strike is not None and p.right is not None
        reduced.append((p.expiry, p.strike, p.right, 1 if p.position > 0 else -1))
    label: str | None
    if len(reduced) == 1:
        label = _match_single(reduced[0])
    elif len(reduced) == 2:
        label = _match_two(reduced[0], reduced[1])
    else:
        label = None
    return _with_mult(label, m) if label is not None else custom


def identify_structure(legs: list[PortfolioPosition]) -> str:
    """Whole-group structure label for one underlying's held legs, or custom (N legs)."""
    legs = [p for p in legs if p.position != 0]
    n = len(legs)
    custom = f"custom ({n} leg{'' if n == 1 else 's'})"
    stock = [p for p in legs if p.sec_type == "STK"]
    opts = [
        p for p in legs
        if p.sec_type == "OPT" and p.expiry and p.strike is not None and p.right
    ]
    if len(stock) + len(opts) != n:  # an unrecognized sec_type leg
        return custom
    if stock:
        return _match_with_stock(stock, opts, custom)
    if not opts:
        return custom
    return _match_options(opts, custom)
