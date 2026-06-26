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


def _qualify_side_effect(*cs: MagicMock) -> list[MagicMock]:
    return [
        _qualified_option(
            expiry=c.lastTradeDateOrContractMonth,
            strike=c.strike,
            right=c.right,
        )
        for c in cs
    ]


@pytest.fixture()
def chain_client(mock_ib) -> ChainClient:
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    return ChainClient(client, secdef_retry_delay=0.0)


async def test_chain_filters_expiries_to_window(chain_client, mock_ib) -> None:
    # Three expiries: 10 DTE (out), 35 DTE (in), 70 DTE (out).
    expiries = [_expiry(10), _expiry(35), _expiry(70)]
    strikes = [395.0, 400.0, 405.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    expiries_seen = {leg.expiry for leg in legs}
    assert expiries_seen == {_expiry(35)}


async def test_chain_emits_both_calls_and_puts(chain_client, mock_ib) -> None:
    expiries = [_expiry(35)]
    strikes = [400.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    rights = {leg.right for leg in legs}
    assert rights == {"C", "P"}


async def test_chain_leg_count_matches_strike_times_2(chain_client, mock_ib) -> None:
    expiries = [_expiry(35)]
    strikes = [395.0, 400.0, 405.0]  # 3 strikes
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert len(legs) == 3 * 2  # 3 strikes x 2 rights


async def test_chain_returns_empty_when_no_expiries_in_window(chain_client, mock_ib) -> None:
    expiries = [_expiry(70)]  # out of window
    strikes = [400.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert legs == []


async def test_chain_leg_carries_greeks_and_oi(chain_client, mock_ib) -> None:
    expiries = [_expiry(35)]
    strikes = [400.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(
        bid=5.0, ask=5.1, iv=0.25, delta=0.45, oi=1234, vol=99
    )
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    for leg in legs:
        assert isinstance(leg, OptionChainLeg)
        assert leg.iv == pytest.approx(0.25)
        assert leg.open_interest == 1234
        assert leg.volume == 99


async def test_chain_windows_strikes_by_band(chain_client, mock_ib) -> None:
    # spot=400, band +/-10% => [360, 440]. 350 and 450 fall outside the band.
    expiries = [_expiry(35)]
    strikes = [350.0, 360.0, 400.0, 440.0, 450.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain(
        "SPY",
        dte_window=(25, 55),
        underlying_price=400.0,
        strike_band_pct=0.10,
        max_strikes_per_side=10,
    )
    assert {leg.strike for leg in legs} == {360.0, 400.0, 440.0}


async def test_chain_caps_strikes_per_side(chain_client, mock_ib) -> None:
    # Wide band, but cap=2/side => nearest 2 below + ATM + nearest 2 above.
    expiries = [_expiry(35)]
    strikes = [395.0, 396.0, 397.0, 398.0, 399.0, 400.0, 401.0, 402.0, 403.0, 404.0, 405.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain(
        "SPY",
        dte_window=(25, 55),
        underlying_price=400.0,
        strike_band_pct=1.0,
        max_strikes_per_side=2,
    )
    assert sorted({leg.strike for leg in legs}) == [398.0, 399.0, 400.0, 401.0, 402.0]


async def test_chain_uses_median_strike_when_no_underlying_price(chain_client, mock_ib) -> None:
    # No underlying_price -> reference = median of listed strikes (300).
    # band +/-10% of 300 = [270, 330] -> only 300 survives.
    expiries = [_expiry(35)]
    strikes = [100.0, 200.0, 300.0, 400.0, 500.0]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, strikes)
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain(
        "SPY",
        dte_window=(25, 55),
        underlying_price=None,
        strike_band_pct=0.10,
        max_strikes_per_side=10,
    )
    assert {leg.strike for leg in legs} == {300.0}


async def test_chain_retries_secdef_until_expiries_in_window(mock_ib) -> None:
    # First secdef result is degenerate (only an out-of-window expiry); the
    # retry returns the full set. get_chain should retry and then succeed.
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    chain_client = ChainClient(client, secdef_retries=2, secdef_retry_delay=0.0)
    bad = _opt_params([_expiry(5)], [400.0])  # 5 DTE -> outside (25, 55)
    good = _opt_params([_expiry(35)], [400.0])  # 35 DTE -> inside the window
    mock_ib.reqSecDefOptParamsAsync.side_effect = [bad, good]
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55), underlying_price=400.0)
    assert {leg.expiry for leg in legs} == {_expiry(35)}
    assert mock_ib.reqSecDefOptParamsAsync.await_count == 2


async def test_chain_returns_empty_after_secdef_retries_exhausted(mock_ib) -> None:
    # Every secdef result is degenerate -> give up after retries+1 attempts.
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    chain_client = ChainClient(client, secdef_retries=2, secdef_retry_delay=0.0)
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(5)], [400.0])
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55), underlying_price=400.0)
    assert legs == []
    assert mock_ib.reqSecDefOptParamsAsync.await_count == 3  # initial + 2 retries


async def test_chain_uses_streaming_not_snapshot(chain_client, mock_ib) -> None:
    """Greeks must be fetched via STREAMING reqMktData, never the snapshot
    reqTickersAsync (options snapshots return no greeks / are billed). The
    streaming subscription must also be cancelled."""
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(35)], [400.0])
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1, iv=0.22, delta=0.5)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert len(legs) == 2  # 1 strike x {C, P}
    assert legs[0].iv == pytest.approx(0.22)
    assert mock_ib.reqMktData.called
    mock_ib.reqTickersAsync.assert_not_awaited()
    assert mock_ib.cancelMktData.called  # streaming lines released


async def test_chain_caps_simultaneous_open_lines(mock_ib) -> None:
    """Never hold more streaming lines open at once than max_market_data_lines."""
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    chain_client = ChainClient(client, max_market_data_lines=2, secdef_retry_delay=0.0)
    # 3 strikes x 2 rights = 6 legs; cap = 2 lines.
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(35)], [395.0, 400.0, 405.0])
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect

    open_lines = 0
    max_open = 0

    def _sub(contract, *a, **k):
        nonlocal open_lines, max_open
        open_lines += 1
        max_open = max(max_open, open_lines)
        return _ticker(bid=5.0, ask=5.1)

    def _cancel(contract):
        nonlocal open_lines
        open_lines -= 1

    mock_ib.reqMktData.side_effect = _sub
    mock_ib.cancelMktData.side_effect = _cancel

    legs = await chain_client.get_chain("SPY", dte_window=(25, 55), underlying_price=400.0)
    assert len(legs) == 6
    assert max_open <= 2


async def test_chain_releases_every_streaming_line(chain_client, mock_ib) -> None:
    """Every subscribed line is cancelled (cancel count == subscribe count)."""
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(35)], [395.0, 400.0, 405.0])
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert len(legs) == 6
    assert mock_ib.reqMktData.call_count == 6
    assert mock_ib.cancelMktData.call_count == 6


async def test_chain_wait_is_batched_not_per_leg(chain_client, mock_ib, monkeypatch) -> None:
    """With greeks present immediately, the batched wait issues zero poll-sleeps
    regardless of leg count -- proving the wait is per-chunk, not per-leg."""
    from unittest.mock import AsyncMock

    sleep_spy = AsyncMock()
    monkeypatch.setattr("optionsbot.ibkr.chains.asyncio.sleep", sleep_spy)
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(
        [_expiry(35)], [390.0, 395.0, 400.0, 405.0, 410.0]
    )
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1, iv=0.2, delta=0.3)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert len(legs) == 10  # 5 strikes x 2 rights
    sleep_spy.assert_not_awaited()


async def test_chain_leg_without_greeks_returns_none_iv(mock_ib) -> None:
    """A leg whose greeks never populate is still returned (iv/delta=None) and
    the chunk wait ends at timeout rather than hanging."""
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    chain_client = ChainClient(
        client, secdef_retry_delay=0.0, greek_wait_timeout=0.05, greek_poll_interval=0.01
    )
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(35)], [400.0])
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect

    def _sub(contract, *a, **k):
        if contract.right == "C":
            return _ticker(bid=5.0, ask=5.1, iv=0.3, delta=0.5)
        t = _ticker(bid=4.0, ask=4.1)
        t.modelGreeks = None  # puts never get greeks
        return t

    mock_ib.reqMktData.side_effect = _sub
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    by_right = {leg.right: leg for leg in legs}
    assert by_right["C"].iv == pytest.approx(0.3)
    assert by_right["P"].iv is None
    assert by_right["P"].delta is None


async def test_chain_skips_leg_that_fails_to_qualify(chain_client, mock_ib) -> None:
    """A leg whose contract can't be qualified is dropped; the rest still
    subscribe, and every opened line is released."""
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(35)], [400.0])

    def _qualify(*cs):
        # Stock qualifies (right == ""); puts are unqualifiable (None at that
        # positional slot -> omitted by qualify_options).
        return [
            None
            if getattr(c, "right", "") == "P"
            else _qualified_option(
                expiry=c.lastTradeDateOrContractMonth, strike=c.strike, right=c.right
            )
            for c in cs
        ]

    mock_ib.qualifyContractsAsync.side_effect = _qualify
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55))
    assert {leg.right for leg in legs} == {"C"}
    assert mock_ib.reqMktData.call_count == 1
    assert mock_ib.cancelMktData.call_count == 1


async def test_chain_greek_wait_exits_on_plateau(mock_ib, monkeypatch) -> None:
    """Once greek coverage plateaus (some legs have greeks, others never will),
    the per-chunk wait early-exits after ~greek_stable_polls polls instead of
    waiting out the full greek_timeout."""
    from unittest.mock import AsyncMock

    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    chain_client = ChainClient(client, secdef_retry_delay=0.0, greek_stable_polls=3)
    # 2 strikes x 2 rights = 4 legs; calls get greeks, puts never -> coverage
    # plateaus at 2/4.
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(35)], [400.0, 405.0])
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect

    def _sub(contract, *a, **k):
        if contract.right == "C":
            return _ticker(bid=5.0, ask=5.1, iv=0.2, delta=0.4)
        t = _ticker(bid=4.0, ask=4.1)
        t.modelGreeks = None
        return t

    mock_ib.reqMktData.side_effect = _sub

    sleep_spy = AsyncMock()
    monkeypatch.setattr("optionsbot.ibkr.chains.asyncio.sleep", sleep_spy)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55), underlying_price=400.0)

    # Calls carry greeks; puts don't.
    assert {leg.right for leg in legs if leg.iv is not None} == {"C"}
    # Plateau break fires after exactly greek_stable_polls (3) sleeps in this
    # scenario; <= 4 leaves one count of headroom. Far below the all-or-timeout
    # path's greek_timeout/greek_poll = 10/0.5 = 20 polls.
    assert sleep_spy.await_count <= 4


async def test_chain_qualifies_once_per_chunk(mock_ib) -> None:
    """Each chunk qualifies its contracts in a single batched call, not per leg."""
    mock_ib.isConnected.return_value = True
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    chain_client = ChainClient(client, max_market_data_lines=2, secdef_retry_delay=0.0)
    # 3 strikes x 2 rights = 6 legs; cap=2 -> 3 chunks.
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([_expiry(35)], [395.0, 400.0, 405.0])
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55), underlying_price=400.0)
    assert len(legs) == 6
    # 1 stock qualify + 3 chunk batch-qualifies = 4 (per-leg would be 1 + 6 = 7).
    assert mock_ib.qualifyContractsAsync.await_count == 4


async def test_chain_fetches_front_and_back_month(chain_client, mock_ib) -> None:
    """With back_dte_gap set, fetch a near-target front + the nearest back-month
    (>= front+gap, pulled from the FULL list, even outside the DTE window)."""
    # In-window (25-55): 30, 45, 49. Back-month (full list, outside window): 75.
    expiries = [_expiry(30), _expiry(45), _expiry(49), _expiry(75)]
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(expiries, [400.0])
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55), back_dte_gap=30)
    assert {leg.expiry for leg in legs} == {_expiry(45), _expiry(75)}


# --- per-expiry real strike selection (IBK-147) ----------------------------


def _contract_details(*, expiry: str, strike: float, right: str) -> MagicMock:
    cd = MagicMock()
    cd.contract = _qualified_option(expiry=expiry, strike=strike, right=right)
    return cd


def _listed_details(strikes_by_expiry: dict[str, list[float]]):
    """reqContractDetailsAsync side_effect: return per-expiry ContractDetails for
    a strike-less partial option (the per-expiry ground truth)."""

    def _details(contract, *a, **k):
        expiry = contract.lastTradeDateOrContractMonth
        return [
            _contract_details(expiry=expiry, strike=s, right=r)
            for s in strikes_by_expiry.get(expiry, [])
            for r in ("C", "P")
        ]

    return _details


async def test_chain_uses_per_expiry_listed_strikes(chain_client, mock_ib) -> None:
    """reqSecDefOptParams returns the UNION of strikes across all expiries, but a
    far-dated month lists a sparser grid. get_chain must select each expiry's band
    from its OWN listed strikes -- so a strike that exists only on the front month
    never produces a (non-existent) back-month leg (the Error 200 root cause)."""
    front, back = _expiry(45), _expiry(80)
    # Union from reqSecDefOptParams: dense -- includes 528, which exists ONLY on
    # the front month (a $1 strike absent from the back month's $5 grid).
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params(
        [front, back], [525.0, 528.0, 530.0, 535.0]
    )
    mock_ib.reqContractDetailsAsync.side_effect = _listed_details(
        {
            front: [525.0, 528.0, 530.0, 535.0],  # dense $1 grid
            back: [525.0, 530.0, 535.0],  # sparse $5 grid -- NO 528
        }
    )
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain(
        "SPY",
        dte_window=(25, 55),
        underlying_price=530.0,
        strike_band_pct=0.10,
        max_strikes_per_side=10,
        back_dte_gap=30,
    )
    front_strikes = {leg.strike for leg in legs if leg.expiry == front}
    back_strikes = {leg.strike for leg in legs if leg.expiry == back}
    assert 528.0 in front_strikes  # exists on the front -> kept
    assert 528.0 not in back_strikes  # absent on the back -> never requested
    assert back_strikes == {525.0, 530.0, 535.0}


async def test_chain_enumerates_strikes_once_per_expiry(chain_client, mock_ib) -> None:
    front, back = _expiry(45), _expiry(80)
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([front, back], [400.0])
    mock_ib.reqContractDetailsAsync.side_effect = _listed_details(
        {front: [400.0], back: [400.0]}
    )
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    await chain_client.get_chain("SPY", dte_window=(25, 55), back_dte_gap=30)
    assert mock_ib.reqContractDetailsAsync.await_count == 2  # one enumeration per expiry


async def test_chain_primes_cache_so_qualify_options_is_a_hit(chain_client, mock_ib) -> None:
    """listed_strikes returns fully-qualified contracts; priming the resolver cache
    makes _fetch_chunk's qualify_options a pure cache hit -> the only
    qualifyContractsAsync call is the underlying stock (no per-leg re-qualify)."""
    exp = _expiry(35)
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([exp], [400.0, 405.0])
    mock_ib.reqContractDetailsAsync.side_effect = _listed_details({exp: [400.0, 405.0]})
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain("SPY", dte_window=(25, 55), underlying_price=400.0)
    assert len(legs) == 4  # 2 strikes x 2 rights
    # 1 qualify for the stock; the option legs are served from the primed cache.
    assert mock_ib.qualifyContractsAsync.await_count == 1


async def test_chain_falls_back_to_union_when_enumeration_empty(chain_client, mock_ib) -> None:
    """If per-expiry enumeration yields nothing (transient/unavailable), fall back
    to the union strike grid -- no worse than the pre-fix behavior -- rather than
    dropping the expiry entirely."""
    exp = _expiry(35)
    mock_ib.reqSecDefOptParamsAsync.return_value = _opt_params([exp], [395.0, 400.0, 405.0])
    mock_ib.reqContractDetailsAsync.return_value = []  # enumeration empty
    mock_ib.qualifyContractsAsync.side_effect = _qualify_side_effect
    mock_ib.reqMktData.return_value = _ticker(bid=5.0, ask=5.1)
    legs = await chain_client.get_chain(
        "SPY",
        dte_window=(25, 55),
        underlying_price=400.0,
        strike_band_pct=0.10,
        max_strikes_per_side=10,
    )
    assert sorted({leg.strike for leg in legs}) == [395.0, 400.0, 405.0]
