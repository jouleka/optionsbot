"""Contract qualification + caching.

``ContractResolver`` takes an ``IBKRClient`` and exposes ``stock``,
``option``, and ``qualify`` helpers. Qualified contracts are cached
in-memory for the life of the resolver -- IB Gateway is the source
of truth and contracts don't change mid-session.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, cast

from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.types import OptionRight

if TYPE_CHECKING:
    from ib_async import Contract

log = logging.getLogger(__name__)


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

    def prune_expired(self, today: date) -> int:
        """Evict cached OPT contracts whose expiry is strictly before ``today``.

        Bounds the long-lived cache: a daemon sharing one resolver across
        trading days otherwise accumulates expired-option entries forever
        (``listed_strikes`` primes the full per-expiry grid each scan). Returns
        the number of entries evicted.

        Fail-safe -- never evicts STK entries (``expiry is None``) or any expiry
        string that does not parse as ``YYYYMMDD``: it is safer to keep a
        maybe-live entry than to drop a live one, and the leak is slow. An
        option is tradeable through its expiry date, so ``expiry == today`` is
        kept. Never raises (the date parse is the only failure point and it is
        handled by keeping the entry).
        """
        stale: list[_CacheKey] = []
        for key in self._cache:
            sec_type, _symbol, expiry, _strike, _right = key
            if sec_type != "OPT" or expiry is None:
                continue
            try:
                expiry_date = datetime.strptime(expiry, "%Y%m%d").date()
            except (ValueError, TypeError):
                continue  # fail-safe: keep anything malformed/unparseable
            if expiry_date < today:
                stale.append(key)
        for key in stale:
            del self._cache[key]
        return len(stale)

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

    async def listed_strikes(
        self,
        symbol: str,
        expiry: str,
        exchange: str = "SMART",
    ) -> list[float]:
        """Return the strikes ACTUALLY listed for one specific expiry.

        ``reqSecDefOptParams`` reports the UNION of strikes across all of an
        underlying's expirations, so a strike present there may not exist for a
        given (especially far-dated) expiry -- a longer-dated month often lists a
        sparser grid (e.g. $5 vs the front week's $1). Qualifying those phantom
        strikes spams IBKR ``Error 200`` and wastes round trips.

        This enumerates the real per-expiry grid via a single
        ``reqContractDetails`` on a strike-less partial option, and primes the
        qualified-contract cache with every returned contract so a subsequent
        ``qualify_options`` for those legs is a pure cache hit (no extra round
        trip, no phantom-strike Error 200). Returns ``[]`` if the expiry has no
        listed contracts (caller may fall back to the union grid).
        """
        await self._client.ensure_connected()
        from ib_async import Option
        partial = Option(symbol, expiry, 0.0, "", exchange)
        try:
            details = await self._client.ib.reqContractDetailsAsync(partial)
        except Exception:  # noqa: BLE001 -- best-effort; caller falls back to union
            log.debug("listed_strikes(%s, %s) enumeration failed", symbol, expiry)
            return []
        strikes: set[float] = set()
        for cd in details or []:
            contract = getattr(cd, "contract", None)
            if contract is None:
                continue
            strike = getattr(contract, "strike", None)
            right = getattr(contract, "right", None)
            if strike is None or right not in ("C", "P"):
                continue
            strike = float(strike)
            strikes.add(strike)
            # Key with the REQUESTED symbol/expiry so the entry matches the key
            # qualify_options builds for the same spec (the contract echoes them).
            key = _contract_cache_key("OPT", symbol, expiry, strike, right)
            self._cache[key] = cast("Contract", contract)
        return sorted(strikes)

    async def qualify(self, contract: Contract) -> Contract:
        """Qualify an already-constructed Contract. Used for advanced flows."""
        await self._client.ensure_connected()
        qualified = await self._client.ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            raise ValueError(f"Could not qualify contract: {contract!r}")
        return cast("Contract", qualified[0])
