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
    ts: datetime | None
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
    ts: datetime | None
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
    con_id: int | None = None


@dataclass(frozen=True, slots=True)
class AccountSummary:
    net_liquidation: Decimal | None
    buying_power: Decimal | None
    available_funds: Decimal | None
    currency: str
    # IBK-122: USD per 1 unit of the account base currency. ``None`` for a
    # non-USD account means the broker did not provide a usable conversion and
    # USD risk sizing must fail closed. USD accounts normalize the field to 1.
    fx_to_usd: Decimal | None = None

    def __post_init__(self) -> None:
        if self.currency.upper() == "USD" and self.fx_to_usd is None:
            object.__setattr__(self, "fx_to_usd", Decimal(1))

    @property
    def net_liquidation_usd(self) -> Decimal | None:
        if self.net_liquidation is None or not self.net_liquidation.is_finite():
            return None
        if self.currency.upper() == "USD":
            return self.net_liquidation
        if (
            self.fx_to_usd is None
            or not self.fx_to_usd.is_finite()
            or self.fx_to_usd <= 0
        ):
            return None
        return self.net_liquidation * self.fx_to_usd


@dataclass(frozen=True, slots=True)
class MarginPreview:
    """Parsed ``whatIfOrder`` result (IBK-125). IBKR returns margin figures
    as strings and uses DBL_MAX sentinels for unset values — both normalized
    to ``None`` here."""

    init_margin_change: float | None
    maint_margin_change: float | None
    equity_with_loan_change: float | None
    commission: float | None
    max_commission: float | None
    warning: str | None


@dataclass(frozen=True, slots=True)
class PlacedOrder:
    """Acknowledgement that an order was handed to ib_async (IBK-125)."""

    ib_order_id: int
    order_ref: str
    action: str  # 'BUY' | 'SELL'
    limit_price: float
    quantity: int
    # One immutable tuple per option leg, in submitted-leg order:
    # (IBKR conId, contract multiplier, currency). Empty only for legacy/mocked
    # acknowledgements that cannot prove exact fill attribution.
    leg_contracts: tuple[tuple[int, int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class OrderStatusUpdate:
    """One orderStatus event, flattened (IBK-125)."""

    ib_order_id: int
    perm_id: int | None
    order_ref: str | None
    status: str  # raw IBKR status string ('Submitted', 'Filled', ...)
    filled: float
    remaining: float
    avg_fill_price: float | None


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    """One execution event (IBK-125). Combo orders report one execution per
    LEG; ``sec_type`` lets consumers skip any BAG-level summary row."""

    ib_order_id: int
    order_ref: str | None
    exec_id: str
    side: str  # normalized 'BUY' | 'SELL' (IBKR reports 'BOT'/'SLD')
    price: float
    qty: int
    ts: datetime
    con_id: int | None
    sec_type: str  # 'OPT' leg vs 'BAG' summary
    # IBK-128: populated from the bundled commissionReport when translating
    # reqExecutions results (live events deliver commissions separately).
    commission: float | None = None


@dataclass(frozen=True, slots=True)
class CommissionUpdate:
    """commissionReport event, keyed to its fill by execId (IBK-125)."""

    exec_id: str
    commission: float
