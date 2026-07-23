"""Curated default universe of liquid US optionable names (IBK-95).

A starting list, overridable via ScreenerSettings.universe. Dotted tickers
(e.g. BRK.B) are intentionally omitted to avoid contract-qualification edge
cases in Stage 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

DEFAULT_UNIVERSE: tuple[str, ...] = (
    # Index / sector / asset ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLY", "XLI",
    "XLP", "XLU", "GLD", "SLV", "TLT", "HYG", "EEM", "ARKK",
    # Mega / large-cap equities
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM",
    "V", "MA", "UNH", "HD", "PG", "XOM", "CVX", "LLY", "KO", "PEP", "COST",
    "WMT", "BAC", "WFC", "GS", "MS", "DIS", "NFLX", "CRM", "ADBE", "AMD",
    "INTC", "QCOM", "MU", "CSCO", "ORCL", "IBM", "PYPL", "UBER", "ABNB",
    "COIN", "PLTR", "SNOW", "SHOP", "BA", "CAT", "DE", "GE", "F", "GM",
    "T", "VZ", "CMCSA", "PFE", "MRK", "JNJ", "NKE", "MCD", "SBUX",
    # High-volume / retail-favorite movers
    "GME", "AMC", "SOFI", "RIOT", "MARA", "SMCI", "NIO",
)

# Cboe's short-dated stock/ETF schedule as of July 2026.  Friday eligibility is
# intentionally handled by the configured universe because all symbols in the
# production 0DTE universe have end-of-week options.  Midweek filtering keeps a
# high-HV Friday-only name from consuming a full-chain scan slot on a day when
# it cannot possibly produce an exact-0DTE candidate.
_DAILY_EXPIRY_SYMBOLS = frozenset({"SPY", "QQQ", "IWM"})
_MONDAY_WEDNESDAY_EXPIRY_SYMBOLS = frozenset(
    {
        "AAPL",
        "AMZN",
        "AVGO",
        "GOOGL",
        "META",
        "MSFT",
        "NVDA",
        "TSLA",
        "GLD",
        "IBIT",
        "SLV",
        "TLT",
    }
)
_MONDAY_EXPIRY_SYMBOLS = frozenset({"AMD", "MU", "SMH", "XLF"})
_WEDNESDAY_EXPIRY_SYMBOLS = frozenset({"UNG", "USO"})


def zero_dte_universe_for_session(
    universe: Sequence[str],
    session_date: date,
    *,
    end_of_week_expiry: bool = False,
) -> tuple[str, ...]:
    """Return configured symbols known to list an expiry on ``session_date``.

    ``end_of_week_expiry`` supports normal Fridays and Thursday-shifted weekly
    expirations before a Friday market holiday.  Ordering and configured scope
    are preserved.
    """
    weekday = session_date.weekday()
    if end_of_week_expiry or weekday == 4:
        eligible = None
    elif weekday in (1, 3):
        eligible = _DAILY_EXPIRY_SYMBOLS
    elif weekday == 0:
        eligible = (
            _DAILY_EXPIRY_SYMBOLS
            | _MONDAY_WEDNESDAY_EXPIRY_SYMBOLS
            | _MONDAY_EXPIRY_SYMBOLS
        )
    elif weekday == 2:
        eligible = (
            _DAILY_EXPIRY_SYMBOLS
            | _MONDAY_WEDNESDAY_EXPIRY_SYMBOLS
            | _WEDNESDAY_EXPIRY_SYMBOLS
        )
    else:
        return ()

    symbols = (symbol.upper() for symbol in universe)
    if eligible is None:
        return tuple(dict.fromkeys(symbols))
    return tuple(dict.fromkeys(symbol for symbol in symbols if symbol in eligible))
