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


async def test_option_raises_when_ib_returns_list_with_none(resolver, mock_ib) -> None:
    # ib_async returns [None] (a NON-empty list) for an unqualifiable
    # (expiry, strike) combo, not []. The guard must still reject it so the
    # caller gets a ValueError rather than a None contract.
    mock_ib.qualifyContractsAsync.return_value = [None]
    with pytest.raises(ValueError, match="Could not qualify"):
        await resolver.option("SPY", "20260619", 99999.0, "C")


async def test_stock_raises_when_ib_returns_list_with_none(resolver, mock_ib) -> None:
    mock_ib.qualifyContractsAsync.return_value = [None]
    with pytest.raises(ValueError, match="Could not qualify"):
        await resolver.stock("NOTREAL")


def test_contract_cache_key_is_deterministic() -> None:
    k1 = _contract_cache_key("STK", "SPY", None, None, None)
    k2 = _contract_cache_key("STK", "SPY", None, None, None)
    assert k1 == k2

    k_opt_call = _contract_cache_key("OPT", "SPY", "20260619", 400.0, "C")
    k_opt_put = _contract_cache_key("OPT", "SPY", "20260619", 400.0, "P")
    assert k_opt_call != k_opt_put


async def test_qualify_options_returns_contract_per_spec(resolver, mock_ib) -> None:
    specs = [("20260619", 400.0, "C"), ("20260619", 405.0, "C")]
    mock_ib.qualifyContractsAsync.return_value = [
        _qualified_option(strike=400.0, right="C"),
        _qualified_option(strike=405.0, right="C"),
    ]
    out = await resolver.qualify_options("SPY", specs)
    assert set(out.keys()) == set(specs)
    assert out[("20260619", 400.0, "C")].strike == 400.0
    mock_ib.qualifyContractsAsync.assert_awaited_once()


async def test_qualify_options_omits_unqualifiable(resolver, mock_ib) -> None:
    specs = [("20260619", 400.0, "C"), ("20260619", 99999.0, "C")]
    mock_ib.qualifyContractsAsync.return_value = [_qualified_option(strike=400.0), None]
    out = await resolver.qualify_options("SPY", specs)
    assert set(out.keys()) == {("20260619", 400.0, "C")}  # None slot omitted


async def test_qualify_options_serves_cache_without_requalifying(resolver, mock_ib) -> None:
    specs = [("20260619", 400.0, "C")]
    mock_ib.qualifyContractsAsync.return_value = [_qualified_option(strike=400.0)]
    await resolver.qualify_options("SPY", specs)
    await resolver.qualify_options("SPY", specs)  # cache hit -> no second call
    assert mock_ib.qualifyContractsAsync.await_count == 1


async def test_qualify_options_batches_misses_into_one_call(resolver, mock_ib) -> None:
    specs = [("20260619", float(k), "C") for k in (400, 405, 410)]
    mock_ib.qualifyContractsAsync.return_value = [
        _qualified_option(strike=float(k)) for k in (400, 405, 410)
    ]
    await resolver.qualify_options("SPY", specs)
    assert mock_ib.qualifyContractsAsync.await_count == 1
    assert len(mock_ib.qualifyContractsAsync.await_args.args) == 3  # all 3 in one call


async def test_qualify_options_only_qualifies_the_misses(resolver, mock_ib) -> None:
    """A mixed call qualifies ONLY the cache-miss specs; cached specs are
    returned without re-qualifying (the cache filter feeds the batch)."""
    seeded = ("20260619", 400.0, "C")
    fresh = ("20260619", 405.0, "C")
    mock_ib.qualifyContractsAsync.return_value = [_qualified_option(strike=400.0)]
    await resolver.qualify_options("SPY", [seeded])  # seed cache for 400C
    mock_ib.qualifyContractsAsync.reset_mock()
    mock_ib.qualifyContractsAsync.return_value = [_qualified_option(strike=405.0)]
    out = await resolver.qualify_options("SPY", [seeded, fresh])  # 400C cached, 405C miss
    assert set(out.keys()) == {seeded, fresh}  # both returned
    mock_ib.qualifyContractsAsync.assert_awaited_once()  # only the miss qualified
    assert len(mock_ib.qualifyContractsAsync.await_args.args) == 1  # just 405C
