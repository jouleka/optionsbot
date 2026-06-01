"""Contract qualification + caching.

``ContractResolver`` takes an ``IBKRClient`` and exposes ``stock``,
``option``, and ``qualify`` helpers. Qualified contracts are cached
in-memory for the life of the resolver -- IB Gateway is the source
of truth and contracts don't change mid-session.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.types import OptionRight

if TYPE_CHECKING:
    from ib_async import Contract


_CacheKey = tuple[str, str, str | None, float | None, str | None]


def _contract_cache_key(
    sec_type: str,
    symbol: str,
    expiry: str | None,
    strike: float | None,
    right: str | None,
) -> _CacheKey:
    return (sec_type, symbol, expiry, strike, right)


class ContractResolver:
    """Wraps qualification calls + caches qualified contracts.

    Caches survive the life of the resolver. For long-running daemons
    that may live across a market session boundary, callers can
    ``.clear_cache()`` between trading days if they ever observe stale
    contracts (rare for vanilla equity options).
    """

    def __init__(self, client: IBKRClient) -> None:
        self._client = client
        self._cache: dict[_CacheKey, Contract] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    async def stock(self, symbol: str) -> Contract:
        key = _contract_cache_key("STK", symbol, None, None, None)
        if key in self._cache:
            return self._cache[key]
        await self._client.ensure_connected()
        from ib_async import Stock
        contract = Stock(symbol, "SMART", "USD")
        qualified = await self._client.ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            raise ValueError(f"Could not qualify stock contract for symbol={symbol!r}")
        # qualifyContractsAsync's default (returnAll=False) returns list[Contract];
        # the union return type in stubs is for the returnAll=True variant.
        result = cast("Contract", qualified[0])
        self._cache[key] = result
        return result

    async def option(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: OptionRight,
        exchange: str = "SMART",
    ) -> Contract:
        key = _contract_cache_key("OPT", symbol, expiry, strike, right)
        if key in self._cache:
            return self._cache[key]
        await self._client.ensure_connected()
        from ib_async import Option
        contract = Option(symbol, expiry, strike, right, exchange)
        qualified = await self._client.ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            raise ValueError(
                f"Could not qualify option contract: symbol={symbol!r} expiry={expiry!r} "
                f"strike={strike!r} right={right!r}"
            )
        result = cast("Contract", qualified[0])
        self._cache[key] = result
        return result

    async def qualify_options(
        self,
        symbol: str,
        specs: Sequence[tuple[str, float, OptionRight]],
    ) -> dict[tuple[str, float, OptionRight], Contract]:
        """Qualify many option contracts in ONE qualifyContractsAsync call.

        Cache-preserving: cached specs are returned directly; only misses are
        qualified, in a single batched call (ib_async runs them concurrently).
        Unqualifiable specs (None in the positional result) are omitted.
        """
        result: dict[tuple[str, float, OptionRight], Contract] = {}
        miss_specs: list[tuple[str, float, OptionRight]] = []
        for spec in specs:
            expiry, strike, right = spec
            key = _contract_cache_key("OPT", symbol, expiry, strike, right)
            cached = self._cache.get(key)
            if cached is not None:
                result[spec] = cached
            else:
                miss_specs.append(spec)
        if miss_specs:
            await self._client.ensure_connected()
            from ib_async import Option
            contracts = [
                Option(symbol, expiry, strike, right, "SMART")
                for expiry, strike, right in miss_specs
            ]
            qualified = await self._client.ib.qualifyContractsAsync(*contracts)
            for spec, q in zip(miss_specs, qualified, strict=True):
                if q is None:
                    continue
                expiry, strike, right = spec
                key = _contract_cache_key("OPT", symbol, expiry, strike, right)
                contract = cast("Contract", q)
                self._cache[key] = contract
                result[spec] = contract
        return result

    async def qualify(self, contract: Contract) -> Contract:
        """Qualify an already-constructed Contract. Used for advanced flows."""
        await self._client.ensure_connected()
        qualified = await self._client.ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            raise ValueError(f"Could not qualify contract: {contract!r}")
        return cast("Contract", qualified[0])
