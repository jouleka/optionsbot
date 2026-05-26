"""Verify adapter dataclasses are frozen and hashable."""

from datetime import UTC, datetime

import pytest

from optionsbot.ibkr.types import (
    AccountSummary,
    OptionChainLeg,
    OptionQuote,
    PositionRecord,
    StockQuote,
)


def test_stock_quote_is_frozen() -> None:
    q = StockQuote(
        symbol="SPY",
        bid=400.0,
        ask=400.1,
        last=400.05,
        mid=400.05,
        ts=datetime(2026, 5, 26, tzinfo=UTC),
        delayed=True,
    )
    with pytest.raises(AttributeError):
        q.bid = 401.0  # type: ignore[misc]


def test_stock_quote_hashable() -> None:
    q = StockQuote("SPY", 400.0, 400.1, 400.05, 400.05, datetime(2026, 5, 26, tzinfo=UTC), True)
    hash(q)  # should not raise


def test_option_quote_round_trips() -> None:
    q = OptionQuote(
        symbol="SPY",
        expiry="20260619",
        strike=400.0,
        right="C",
        bid=5.0,
        ask=5.1,
        last=5.05,
        mid=5.05,
        iv=0.18,
        delta=0.5,
        gamma=0.02,
        theta=-0.04,
        vega=0.6,
        open_interest=1000,
        volume=50,
        ts=datetime(2026, 5, 26, tzinfo=UTC),
        delayed=False,
    )
    assert q.right == "C"


def test_option_chain_leg_omits_ts_and_delayed() -> None:
    # Sanity: chain legs don't carry per-leg timestamp (chain is a snapshot)
    from dataclasses import fields
    names = {f.name for f in fields(OptionChainLeg)}
    assert "ts" not in names
    assert "delayed" not in names


def test_position_record_is_flat() -> None:
    pos = PositionRecord(
        account="DU1234567",
        symbol="SPY",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        position=100.0,
        avg_cost=399.5,
    )
    assert pos.symbol == "SPY"
    assert pos.sec_type == "STK"


def test_account_summary_allows_none_fields() -> None:
    s = AccountSummary(
        net_liquidation=None,
        buying_power=None,
        available_funds=None,
        currency="USD",
    )
    assert s.currency == "USD"
