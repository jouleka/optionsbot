"""Tests for options chain retrieval."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.chains import ChainClient
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.types import OptionChainLeg


def _expiry(days_from_today: int) -> str:
    d = date.today() + timedelta(days=days_from_today)
    return d.strftime("%Y%m%d")


def _opt_params(expirations: list[str], strikes: list[float]) -> list[MagicMock]:
    """Mock reqSecDefOptParamsAsync return shape (a list of OptionChain rows)."""
    p = MagicMock()
    p.exchange = "SMART"
    p.tradingClass = "SPY"
    p.multiplier = "100"
    p.expirations = expirations
    p.strikes = strikes
    return [p]


def _qualified_option(symbol="SPY", expiry="", strike=400.0, right="C"):
    c = MagicMock(
        symbol=symbol,
        secType="OPT",
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
        exchange="SMART",
        currency="USD",
    )
    c.conId = hash((expiry, strike, right)) & 0xFFFFFFFF
    return c


def _ticker(*, bid, ask, iv=0.2, delta=0.3, oi=100, vol=50) -> MagicMock:
    t = MagicMock()
    t.bid = bid
    t.ask = ask
    g = MagicMock(impliedVol=iv, delta=delta, gamma=0.01, theta=-0.02, vega=0.1)
    t.modelGreeks = g
    t.openInterest = oi
    t.volume = vol
    return t


def _qualify_side_effect(c: MagicMock) -> list[MagicMock]:
    return [
        _qualified_option(
            expiry=c.lastTradeDateOrContractMonth,
            strike=c.strike,
            right=c.right,
        )
    ]


@pytest.fixture()
def chain_client(mock_ib) -> ChainClient:
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    return ChainClient(client, max_concurrent=4, max_calls_per_window=50, window_seconds=10.0)


async def test_chain_filters_expiries_to_window(chain_client, mock_ib) -> None:
    # Three expiries: 10 DTE (out), 35 DTE (in), 70 DTE (out).
    expiries = [_expiry(10), _expiry(35), _expiry(70)]
    strikes = [395.0, 400.0, 405.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqTickersAsync.return_value = [_ticker(bid=5.0, ask=5.1)]
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    expiries_seen = {leg.expiry for leg in legs}
    assert expiries_seen == {_expiry(35)}


async def test_chain_emits_both_calls_and_puts(chain_client, mock_ib) -> None:
    expiries = [_expiry(35)]
    strikes = [400.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqTickersAsync.return_value = [_ticker(bid=5.0, ask=5.1)]
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    rights = {leg.right for leg in legs}
    assert rights == {"C", "P"}


async def test_chain_leg_count_matches_strike_times_2(chain_client, mock_ib) -> None:
    expiries = [_expiry(35)]
    strikes = [395.0, 400.0, 405.0]  # 3 strikes
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqTickersAsync.return_value = [_ticker(bid=5.0, ask=5.1)]
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert len(legs) == 3 * 2  # 3 strikes x 2 rights


async def test_chain_returns_empty_when_no_expiries_in_window(chain_client, mock_ib) -> None:
    expiries = [_expiry(70)]  # out of window
    strikes = [400.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqTickersAsync.return_value = []
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert legs == []


async def test_chain_leg_carries_greeks_and_oi(chain_client, mock_ib) -> None:
    expiries = [_expiry(35)]
    strikes = [400.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqTickersAsync.return_value = [
        _ticker(bid=5.0, ask=5.1, iv=0.25, delta=0.45, oi=1234, vol=99)
    ]
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    for leg in legs:
        assert isinstance(leg, OptionChainLeg)
        assert leg.iv == pytest.approx(0.25)
        assert leg.open_interest == 1234
        assert leg.volume == 99
