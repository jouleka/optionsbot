"""Tests for contract resolution + caching."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver, _contract_cache_key


@pytest.fixture()
def resolver(mock_ib) -> ContractResolver:
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    return ContractResolver(client)


def _qualified_stock(symbol: str = "SPY") -> MagicMock:
    """Mimic ib_async qualifying a Stock to itself with conId attached."""
    c = MagicMock()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    c.conId = 12345
    return c


def _qualified_option(
    symbol: str = "SPY",
    expiry: str = "20260619",
    strike: float = 400.0,
    right: str = "C",
) -> MagicMock:
    c = MagicMock()
    c.symbol = symbol
    c.secType = "OPT"
    c.lastTradeDateOrContractMonth = expiry
    c.strike = strike
    c.right = right
    c.exchange = "SMART"
    c.currency = "USD"
    c.multiplier = "100"
    c.conId = 98765
    return c


async def test_stock_qualifies_via_ib_async(resolver, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = [_qualified_stock("SPY")]
    qualified = await resolver.stock("SPY")
    assert qualified.symbol == "SPY"
    assert qualified.conId == 12345
    mock_ib.qualifyContractsAsync.assert_awaited_once()


async def test_stock_cache_hits_skip_qualification(resolver, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = [_qualified_stock("SPY")]
    await resolver.stock("SPY")
    await resolver.stock("SPY")
    assert mock_ib.qualifyContractsAsync.await_count == 1


async def test_option_qualifies_with_all_legs(resolver, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = [_qualified_option()]
    qualified = await resolver.option("SPY", "20260619", 400.0, "C")
    assert qualified.right == "C"
    assert qualified.strike == 400.0
    assert qualified.lastTradeDateOrContractMonth == "20260619"


async def test_option_cache_uses_full_key(resolver, mock_ib) -> None:
    # Two different strikes -> two qualifications. Same strike repeated -> cache hit.
    mock_ib.qualifyContractsAsync.side_effect = [
        [_qualified_option(strike=400.0)],
        [_qualified_option(strike=405.0)],
    ]
    await resolver.option("SPY", "20260619", 400.0, "C")
    await resolver.option("SPY", "20260619", 405.0, "C")
    await resolver.option("SPY", "20260619", 400.0, "C")  # cache hit
    assert mock_ib.qualifyContractsAsync.await_count == 2


async def test_stock_raises_when_ib_returns_empty(resolver, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = []
    with pytest.raises(ValueError, match="Could not qualify"):
        await resolver.stock("NOTREAL")


def test_contract_cache_key_is_deterministic() -> None:
    k1 = _contract_cache_key("STK", "SPY", None, None, None)
    k2 = _contract_cache_key("STK", "SPY", None, None, None)
    assert k1 == k2

    k_opt_call = _contract_cache_key("OPT", "SPY", "20260619", 400.0, "C")
    k_opt_put = _contract_cache_key("OPT", "SPY", "20260619", 400.0, "P")
    assert k_opt_call != k_opt_put
