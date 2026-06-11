"""IBKR integration layer.

Public surface; downstream code should import from here, not from
submodules directly.
"""

from optionsbot.ibkr.chains import ChainClient
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.history import HistoryClient
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.orders import OrderClient
from optionsbot.ibkr.pacing import ConcurrencyLimiter, RateLimiter
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.ibkr.types import (
    AccountSummary,
    CommissionUpdate,
    ExecutionFill,
    MarginPreview,
    OptionChainLeg,
    OptionQuote,
    OrderStatusUpdate,
    PlacedOrder,
    PositionRecord,
    StockQuote,
)

__all__ = [
    "AccountSummary",
    "ChainClient",
    "CommissionUpdate",
    "ConcurrencyLimiter",
    "ContractResolver",
    "ExecutionFill",
    "HistoryClient",
    "IBKRClient",
    "MarginPreview",
    "MarketDataClient",
    "OptionChainLeg",
    "OptionQuote",
    "OrderClient",
    "OrderStatusUpdate",
    "PlacedOrder",
    "PositionRecord",
    "PositionsClient",
    "RateLimiter",
    "StockQuote",
]
