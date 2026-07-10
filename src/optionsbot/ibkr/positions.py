"""Read-only positions and account summary.

``PositionsClient.get_positions()`` returns a list of flat
``PositionRecord`` objects derived from ``ib_async.IB.positions()``;
``get_account_summary()`` extracts ``NetLiquidation``, ``BuyingPower``
and ``AvailableFunds`` from ``ib_async.IB.accountSummaryAsync()`` and
wraps them in an ``AccountSummary``.

Both calls are TTL-cached (default 60s) per ``PositionsClient`` instance.

``ib.positions()`` is a passive read of already-received data and is
synchronous -- do NOT ``await`` it. ``ib.accountSummary()``, by contrast,
is a blocking wrapper that drives the event loop via ``IB._run`` (unsafe
inside our running loop), so we use the awaitable
``accountSummaryAsync()`` instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from optionsbot.ibkr._util import clean_float
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.types import (
    AccountSummary,
    OptionRight,
    PortfolioPosition,
    PositionRecord,
)

if TYPE_CHECKING:
    from ib_async import Position

log = logging.getLogger(__name__)

_DEFAULT_TTL = 60.0  # seconds
_ACCOUNT_TAGS = ("NetLiquidation", "BuyingPower", "AvailableFunds")


def _to_decimal(s: str | None) -> Decimal | None:
    if s is None:
        return None
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def _usd_per_base(rows: object, base_currency: str) -> Decimal | None:
    """USD per 1 unit of the account base currency, from the $LEDGER rows.

    IBKR reports an ``ExchangeRate`` row per held currency = base-currency
    units per 1 unit of that currency. With base=EUR the ``currency="USD"``
    row is EUR-per-USD, so USD-per-base = 1 / that. A missing or malformed
    non-USD conversion returns ``None`` so dollar risk sizing fails closed.
    """
    if base_currency.upper() == "USD":
        return Decimal(1)
    for row in rows:  # type: ignore[attr-defined]
        if getattr(row, "tag", None) == "ExchangeRate" and getattr(row, "currency", None) == "USD":
            rate = _to_decimal(getattr(row, "value", None))
            if rate is not None and rate.is_finite() and rate > 0:
                return Decimal(1) / rate
    log.error(
        "no usable USD ExchangeRate row for base %s; USD net-liq is unavailable",
        base_currency,
    )
    return None


def _norm_right(right: str | None) -> OptionRight | None:
    if not right:
        return None
    c = right[0].upper()
    return "C" if c == "C" else "P" if c == "P" else None


def _to_portfolio_position(item: object) -> PortfolioPosition:
    """Map an ``ib_async.PortfolioItem`` to our flat :class:`PortfolioPosition`.

    Option contract fields (expiry/strike/right) are captured only for OPT legs;
    stock legs leave them None and use a multiplier of 1.
    """
    c = item.contract  # type: ignore[attr-defined]
    sec_type = getattr(c, "secType", "") or ""
    is_opt = sec_type == "OPT"
    raw_strike = getattr(c, "strike", 0.0) or 0.0
    mult_raw = getattr(c, "multiplier", "") or ""
    try:
        multiplier = int(float(mult_raw)) if mult_raw else (100 if is_opt else 1)
    except (TypeError, ValueError):
        multiplier = 100 if is_opt else 1
    return PortfolioPosition(
        account=getattr(item, "account", "") or "",
        symbol=getattr(c, "symbol", "") or "",
        sec_type=sec_type,
        expiry=(getattr(c, "lastTradeDateOrContractMonth", "") or None) if is_opt else None,
        strike=(float(raw_strike) if raw_strike and raw_strike > 0 else None) if is_opt else None,
        right=_norm_right(getattr(c, "right", None)) if is_opt else None,
        multiplier=multiplier,
        position=float(item.position),  # type: ignore[attr-defined]
        avg_cost=float(item.averageCost),  # type: ignore[attr-defined]
        market_price=clean_float(getattr(item, "marketPrice", None)),
        market_value=clean_float(getattr(item, "marketValue", None)),
        unrealized_pnl=clean_float(getattr(item, "unrealizedPNL", None)),
        realized_pnl=clean_float(getattr(item, "realizedPNL", None)),
    )


class PositionsClient:
    def __init__(self, client: IBKRClient, cache_ttl_seconds: float = _DEFAULT_TTL) -> None:
        self._client = client
        self._ttl = cache_ttl_seconds
        self._positions_cache: tuple[float, list[PositionRecord]] | None = None
        self._portfolio_cache: tuple[float, list[PortfolioPosition]] | None = None
        self._summary_cache: tuple[float, AccountSummary] | None = None
        self._lock = asyncio.Lock()

    async def get_positions(self) -> list[PositionRecord]:
        async with self._lock:
            now = time.monotonic()
            if self._positions_cache and now - self._positions_cache[0] < self._ttl:
                return self._positions_cache[1]
            await self._client.ensure_connected()
            raw: list[Position] = self._client.ib.positions()  # sync; do NOT await
            out = [
                PositionRecord(
                    account=p.account,
                    symbol=p.contract.symbol,
                    sec_type=p.contract.secType,
                    exchange=getattr(p.contract, "exchange", "") or "",
                    currency=getattr(p.contract, "currency", "") or "",
                    position=float(p.position),
                    avg_cost=float(p.avgCost),
                )
                for p in raw
            ]
            self._positions_cache = (now, out)
            return out

    async def get_portfolio(self) -> list[PortfolioPosition]:
        """Enriched open positions via ``ib.portfolio()`` (IBKR-computed market value
        + unrealized P&L, matching TWS). TTL-cached like the other reads.

        Like ``ib.positions()``, ``ib.portfolio()`` is a passive read of already-
        streamed account-update data -- synchronous, do NOT await it.
        """
        async with self._lock:
            now = time.monotonic()
            if self._portfolio_cache and now - self._portfolio_cache[0] < self._ttl:
                return self._portfolio_cache[1]
            await self._client.ensure_connected()
            raw = self._client.ib.portfolio()  # sync; do NOT await
            out = [_to_portfolio_position(item) for item in raw]
            self._portfolio_cache = (now, out)
            return out

    async def get_account_summary(self) -> AccountSummary:
        async with self._lock:
            now = time.monotonic()
            if self._summary_cache and now - self._summary_cache[0] < self._ttl:
                return self._summary_cache[1]
            await self._client.ensure_connected()
            # Unlike ib.positions() (a passive read of already-received data,
            # safe to call sync), ib.accountSummary() is a BLOCKING wrapper that
            # drives the event loop via ib._run() -- calling it inside our
            # running loop raises "This event loop is already running". Use the
            # async variant and await it.
            rows = await self._client.ib.accountSummaryAsync()
            by_tag: dict[str, tuple[str, str]] = {}  # tag -> (value, currency)
            for row in rows:
                tag = getattr(row, "tag", None)
                if tag in _ACCOUNT_TAGS:
                    by_tag[tag] = (
                        getattr(row, "value", ""),
                        getattr(row, "currency", "USD") or "USD",
                    )
            currency = next((c for (_, c) in by_tag.values()), "USD")
            fx_to_usd = _usd_per_base(rows, currency)
            summary = AccountSummary(
                net_liquidation=_to_decimal(by_tag.get("NetLiquidation", (None, ""))[0]),
                buying_power=_to_decimal(by_tag.get("BuyingPower", (None, ""))[0]),
                available_funds=_to_decimal(by_tag.get("AvailableFunds", (None, ""))[0]),
                currency=currency,
                fx_to_usd=fx_to_usd,
            )
            self._summary_cache = (now, summary)
            return summary
