"""Read-only positions and account summary.

``PositionsClient.get_positions()`` returns a list of flat
``PositionRecord`` objects derived from ``ib_async.IB.positions()``;
``get_account_summary()`` extracts ``NetLiquidation``, ``BuyingPower``
and ``AvailableFunds`` from ``ib_async.IB.accountSummary()`` and wraps
them in an ``AccountSummary``.

Both calls are TTL-cached (default 60s) per ``PositionsClient`` instance.
The underlying ``ib.positions()`` / ``ib.accountSummary()`` are
synchronous methods on ib_async (they internally drive the asyncio
event loop via ``IB._run``); do NOT ``await`` them.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.types import AccountSummary, PositionRecord

if TYPE_CHECKING:
    from ib_async import Position


_DEFAULT_TTL = 60.0  # seconds
_ACCOUNT_TAGS = ("NetLiquidation", "BuyingPower", "AvailableFunds")


def _to_decimal(s: str | None) -> Decimal | None:
    if s is None:
        return None
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


class PositionsClient:
    def __init__(self, client: IBKRClient, cache_ttl_seconds: float = _DEFAULT_TTL) -> None:
        self._client = client
        self._ttl = cache_ttl_seconds
        self._positions_cache: tuple[float, list[PositionRecord]] | None = None
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
            summary = AccountSummary(
                net_liquidation=_to_decimal(by_tag.get("NetLiquidation", (None, ""))[0]),
                buying_power=_to_decimal(by_tag.get("BuyingPower", (None, ""))[0]),
                available_funds=_to_decimal(by_tag.get("AvailableFunds", (None, ""))[0]),
                currency=currency,
            )
            self._summary_cache = (now, summary)
            return summary
