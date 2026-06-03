"""Curated default universe of liquid US optionable names (IBK-95).

A starting list, overridable via ScreenerSettings.universe. Dotted tickers
(e.g. BRK.B) are intentionally omitted to avoid contract-qualification edge
cases in Stage 1.
"""

from __future__ import annotations

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
