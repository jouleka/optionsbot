"""Adapter dataclasses for ib_async types.

The rest of the codebase imports from here, not from ib_async directly.
All types are frozen + slots for cheap, hashable, immutable records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

OptionRight = Literal["C", "P"]


@dataclass(frozen=True, slots=True)
class StockQuote:
    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None  # (bid + ask) / 2 when both available
    ts: datetime
    delayed: bool  # True when the quote came from delayed market data


@dataclass(frozen=True, slots=True)
class OptionQuote:
    symbol: str
    expiry: str  # YYYYMMDD
    strike: float
    right: OptionRight
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None
    iv: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    open_interest: int | None
    volume: int | None
    ts: datetime
    delayed: bool


@dataclass(frozen=True, slots=True)
class OptionChainLeg:
    """One strike+right of an options chain. Same shape as OptionQuote but
    packaged for chain-level processing (rate-limited bulk fetch)."""
    symbol: str
    expiry: str
    strike: float
    right: OptionRight
    bid: float | None
    ask: float | None
    iv: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    open_interest: int | None
    volume: int | None


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """Flat position adapter. ib_async.Position carries `contract` as a
    nested object; we flatten the fields actually used by Covered Call
    eligibility checks and similar downstream consumers."""
    account: str
    symbol: str
    sec_type: str  # 'STK', 'OPT', 'FUT', etc.
    exchange: str
    currency: str
    position: float  # ib_async uses float; positions can be fractional for some products
    avg_cost: float


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    """Enriched open position from ``ib.portfolio()``: contract identity plus
    IBKR-computed market value and unrealized P&L. Used by the open-book read
    surface (IBK-112). ``PositionRecord`` (above) stays the lean shape used by
    Covered-Call eligibility; this is the richer view for position tracking."""

    account: str
    symbol: str
    sec_type: str  # 'OPT', 'STK', ...
    expiry: str | None  # YYYYMMDD for options; None for stock
    strike: float | None
    right: OptionRight | None  # 'C' / 'P'
    multiplier: int  # 100 for equity options, 1 for stock
    position: float  # signed; short = negative
    avg_cost: float  # IBKR averageCost (per contract, already x multiplier for options)
    market_price: float | None
    market_value: float | None
    unrealized_pnl: float | None
    realized_pnl: float | None


@dataclass(frozen=True, slots=True)
class AccountSummary:
    net_liquidation: Decimal | None
    buying_power: Decimal | None
    available_funds: Decimal | None
    currency: str
