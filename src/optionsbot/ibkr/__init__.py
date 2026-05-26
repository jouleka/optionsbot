"""IBKR integration layer.

Public surface; downstream code should import from here, not from
submodules directly.
"""

from optionsbot.ibkr.chains import ChainClient
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.history import HistoryClient
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.pacing import ConcurrencyLimiter, RateLimiter
from optionsbot.ibkr.types import (
    AccountSummary,
    OptionChainLeg,
    OptionQuote,
    PositionRecord,
    StockQuote,
)

__all__ = [
    "AccountSummary",
    "ChainClient",
    "ConcurrencyLimiter",
    "ContractResolver",
    "HistoryClient",
    "IBKRClient",
    "MarketDataClient",
    "OptionChainLeg",
    "OptionQuote",
    "PositionRecord",
    "RateLimiter",
    "StockQuote",
]
