"""Load IBKR-response JSON fixtures and build a MagicMock that emulates
``ib_async.IB`` for end-to-end replay tests.

The mock returns ib_async-shaped objects (MagicMocks with the right
attributes) for each low-level call (``connectAsync``,
``qualifyContractsAsync``, ``reqSecDefOptParamsAsync``, ``reqTickersAsync``,
``reqHistoricalDataAsync``, ``positions``, ``accountSummaryAsync``). The mock
honours the symbol mapping from the fixture so a single test instance can
simulate a multi-symbol scan.

No ``ib_async`` import is needed here: we emulate the shapes with
``MagicMock`` so the test tier has no hard dependency on ib_async internals.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Strategy viability is computed from option DTE against the real wall-clock
# (``strategies/base.py`` uses ``date.today()``). A fixture with a hardcoded
# expiry therefore rots: captured at ~``_FRONT_DTE`` days out, its DTE shrinks
# as real time passes until every strategy filters it out and the scan scores
# nothing. We re-anchor the fixture's nearest expiry to this many days from
# today on load, preserving the spacing of any back-month expiries, so the
# integration replay stays viable on any calendar day.
_FRONT_DTE = 45


def _relativize_expiries(fixture: dict[str, Any], front_dte: int = _FRONT_DTE) -> None:
    """Shift every expiry in ``fixture`` so its nearest one sits ``front_dte``
    days from today, mutating ``option_params.expirations``, ``option_chain``
    legs, and any ``positions`` legs in place (the lookup keys stay consistent
    because both the param list and the chain legs are rewritten together)."""
    params = fixture.get("option_params", {})
    expirations = params.get("expirations") or []
    if not expirations:
        return
    parsed = {e: datetime.strptime(e, "%Y%m%d").date() for e in set(expirations)}
    delta = timedelta(days=front_dte) - (min(parsed.values()) - date.today())
    remap = {old: (d + delta).strftime("%Y%m%d") for old, d in parsed.items()}

    params["expirations"] = [remap[e] for e in expirations]
    for leg in fixture.get("option_chain", []):
        if leg.get("expiry") in remap:
            leg["expiry"] = remap[leg["expiry"]]
    for pos in fixture.get("positions", []):
        if pos.get("expiry") in remap:
            pos["expiry"] = remap[pos["expiry"]]


def load_fixture(name: str, *, relativize_expiries: bool = True) -> dict[str, Any]:
    """Load ``tests/fixtures/ibkr/<name>.json``.

    By default the fixture's option expiries are re-anchored to today (see
    ``_relativize_expiries``) so replay tests do not rot as the calendar
    advances. Pass ``relativize_expiries=False`` for tests that assert on the
    raw on-disk expiry strings.
    """
    path = Path(__file__).resolve().parents[1] / "fixtures" / "ibkr" / f"{name}.json"
    with path.open() as f:
        fixture: dict[str, Any] = json.load(f)
    if relativize_expiries:
        _relativize_expiries(fixture)
    return fixture


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
    ib.accountSummaryAsync = AsyncMock(
        return_value=_account_summary_rows(
            next(iter(by_symbol.values()))["account_summary"]
        )
    )

    # qualifyContractsAsync(*contracts): echo each back with a stable fake conId,
    # positionally aligned with the input (matches ib_async's contract).
    async def _qualify(*contracts: Any) -> list[Any]:
        out: list[Any] = []
        for contract in contracts:
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
            out.append(contract)
        return out

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
