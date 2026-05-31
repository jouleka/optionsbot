"""Load IBKR-response JSON fixtures and build a MagicMock that emulates
``ib_async.IB`` for end-to-end replay tests.

The mock returns ib_async-shaped objects (MagicMocks with the right
attributes) for each low-level call (``connectAsync``,
``qualifyContractsAsync``, ``reqSecDefOptParamsAsync``, ``reqTickersAsync``,
``reqHistoricalDataAsync``, ``positions``, ``accountSummary``). The mock
honours the symbol mapping from the fixture so a single test instance can
simulate a multi-symbol scan.

No ``ib_async`` import is needed here: we emulate the shapes with
``MagicMock`` so the test tier has no hard dependency on ib_async internals.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def load_fixture(name: str) -> dict[str, Any]:
    """Load ``tests/fixtures/ibkr/<name>.json``."""
    path = Path(__file__).resolve().parents[1] / "fixtures" / "ibkr" / f"{name}.json"
    with path.open() as f:
        return json.load(f)


def _bar_mock(b: dict[str, Any]) -> MagicMock:
    m = MagicMock()
    m.date = date.fromisoformat(b["date"])
    m.open = b["open"]
    m.high = b["high"]
    m.low = b["low"]
    m.close = b["close"]
    m.volume = b["volume"]
    return m


def _ticker_mock(spot_data: dict[str, Any]) -> MagicMock:
    t = MagicMock()
    t.bid = spot_data.get("bid")
    t.ask = spot_data.get("ask")
    t.last = spot_data.get("last")
    t.time = None  # falls back to datetime.now in _ticker_ts
    if spot_data.get("modelGreeks") is None:
        t.modelGreeks = None
    else:
        g = MagicMock(
            impliedVol=spot_data["modelGreeks"].get("iv"),
            delta=spot_data["modelGreeks"].get("delta"),
            gamma=spot_data["modelGreeks"].get("gamma"),
            theta=spot_data["modelGreeks"].get("theta"),
            vega=spot_data["modelGreeks"].get("vega"),
        )
        t.modelGreeks = g
    t.openInterest = spot_data.get("openInterest")
    t.volume = spot_data.get("volume")
    return t


def _option_chain_ticker_mock(leg: dict[str, Any]) -> MagicMock:
    t = MagicMock()
    t.bid = leg.get("bid")
    t.ask = leg.get("ask")
    t.time = None
    g = MagicMock(
        impliedVol=leg.get("iv"),
        delta=leg.get("delta"),
        gamma=leg.get("gamma"),
        theta=leg.get("theta"),
        vega=leg.get("vega"),
    )
    t.modelGreeks = g
    t.openInterest = leg.get("openInterest", 0)
    t.volume = leg.get("volume", 0)
    return t


def _option_params_mock(params_data: dict[str, Any]) -> list[MagicMock]:
    p = MagicMock()
    p.exchange = params_data["exchange"]
    p.tradingClass = params_data["tradingClass"]
    p.multiplier = params_data["multiplier"]
    p.expirations = params_data["expirations"]
    p.strikes = params_data["strikes"]
    return [p]


def _account_summary_rows(data: list[dict[str, Any]]) -> list[MagicMock]:
    out = []
    for row in data:
        r = MagicMock()
        r.tag = row["tag"]
        r.value = row["value"]
        r.currency = row.get("currency", "USD")
        out.append(r)
    return out


def build_ib_mock(*fixtures: dict[str, Any]) -> MagicMock:
    """Build a MagicMock that emulates ``ib_async.IB`` for the given fixtures.

    ``fixtures`` is one or more loaded fixture dicts. The mock multiplexes
    by symbol -- if multiple fixtures are passed, calls keyed by the
    contract's ``symbol`` attribute route to the right fixture data.
    """
    by_symbol: dict[str, dict[str, Any]] = {f["symbol"]: f for f in fixtures}

    ib = MagicMock()
    ib.connectAsync = AsyncMock()
    ib.disconnect = MagicMock()
    ib.isConnected = MagicMock(return_value=True)
    ib.reqMarketDataType = MagicMock()
    ib.positions = MagicMock(return_value=[])
    ib.accountSummary = MagicMock(
        return_value=_account_summary_rows(
            next(iter(by_symbol.values()))["account_summary"]
        )
    )

    # qualifyContractsAsync: echo back the contract with a stable fake conId.
    async def _qualify(contract: Any) -> list[Any]:
        contract.conId = (
            abs(
                hash(
                    (
                        getattr(contract, "symbol", ""),
                        getattr(contract, "lastTradeDateOrContractMonth", ""),
                        getattr(contract, "strike", 0.0),
                        getattr(contract, "right", ""),
                    )
                )
            )
            & 0xFFFFFFFF
        )
        return [contract]

    ib.qualifyContractsAsync = AsyncMock(side_effect=_qualify)

    # reqSecDefOptParamsAsync: return option_params for the matching symbol.
    async def _opt_params(
        symbol: str,
        fut_fop_exchange: str,
        sec_type: str,
        con_id: int,
    ) -> list[MagicMock]:
        fx = by_symbol.get(symbol)
        if fx is None:
            return []
        return _option_params_mock(fx["option_params"])

    ib.reqSecDefOptParamsAsync = AsyncMock(side_effect=_opt_params)

    # reqHistoricalDataAsync: return bar mocks for the contract's symbol.
    async def _history(contract: Any, **kwargs: Any) -> list[MagicMock]:
        symbol = getattr(contract, "symbol", None)
        fx = by_symbol.get(symbol)
        if fx is None:
            return []
        return [_bar_mock(b) for b in fx["history"]]

    ib.reqHistoricalDataAsync = AsyncMock(side_effect=_history)

    # reqTickersAsync: for STK -> spot ticker; for OPT -> chain leg lookup.
    async def _tickers(contract: Any) -> list[MagicMock]:
        symbol = getattr(contract, "symbol", None)
        sec_type = getattr(contract, "secType", "STK")
        fx = by_symbol.get(symbol)
        if fx is None:
            return [_ticker_mock({})]
        if sec_type != "OPT":
            return [_ticker_mock(fx["spot"])]
        # OPT: look up by (expiry, strike, right)
        expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
        strike = getattr(contract, "strike", 0.0)
        right = getattr(contract, "right", "")
        for leg in fx["option_chain"]:
            if (
                leg["expiry"] == expiry
                and leg["strike"] == strike
                and leg["right"] == right
            ):
                return [_option_chain_ticker_mock(leg)]
        # Leg not in fixture -- return an empty-ish ticker rather than crashing.
        return [_option_chain_ticker_mock({})]

    ib.reqTickersAsync = AsyncMock(side_effect=_tickers)

    # reqMktData (STREAMING, sync): the chain client now subscribes to streaming
    # market data for option legs -- IBKR computes greeks only on the streaming
    # feed. Returns a SINGLE ticker (vs reqTickersAsync's list) with greeks
    # populated, so the leg fetch's greek-poll resolves on the first check.
    def _mkt_data(contract: Any, *args: Any) -> MagicMock:
        symbol = getattr(contract, "symbol", None)
        sec_type = getattr(contract, "secType", "STK")
        fx = by_symbol.get(symbol)
        if fx is None:
            return _ticker_mock({})
        if sec_type != "OPT":
            return _ticker_mock(fx["spot"])
        expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
        strike = getattr(contract, "strike", 0.0)
        right = getattr(contract, "right", "")
        for leg in fx["option_chain"]:
            if (
                leg["expiry"] == expiry
                and leg["strike"] == strike
                and leg["right"] == right
            ):
                return _option_chain_ticker_mock(leg)
        return _option_chain_ticker_mock({})

    ib.reqMktData = MagicMock(side_effect=_mkt_data)
    ib.cancelMktData = MagicMock()

    return ib
