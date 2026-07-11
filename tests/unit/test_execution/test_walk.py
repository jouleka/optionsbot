"""Tests for the price-walk math + runner (IBK-127)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Engine, insert, select, update

from optionsbot.config import Settings
from optionsbot.execution.orders import get_order, record_order_quotes
from optionsbot.execution.walk import (
    combo_bid_ask,
    liquidity_issues,
    next_walk_target,
    price_increment_for,
    run_price_walk,
    slippage_budget,
)
from optionsbot.ibkr.types import OptionQuote
from optionsbot.storage.schema import order_quotes, orders

NOW = datetime(2026, 6, 11, 15, 30, tzinfo=UTC)

LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
]


def _quote(
    strike: float, right: str, *, bid: float | None, ask: float | None,
    mid: float | None = None, oi: int | None = None,
    ts: datetime | None = NOW, delayed: bool = True,
) -> OptionQuote:
    computed_mid = mid if mid is not None else (
        (bid + ask) / 2 if bid is not None and ask is not None else None
    )
    return OptionQuote(
        symbol="SPY", expiry="20260717", strike=strike, right=right,  # type: ignore[arg-type]
        bid=bid, ask=ask, last=None, mid=computed_mid, iv=None, delta=None,
        gamma=None, theta=None, vega=None, open_interest=oi, volume=None,
        ts=ts, delayed=delayed,
    )


GOOD_QUOTES = {
    ("20260717", 580.0, "P"): _quote(580.0, "P", bid=1.55, ask=1.65, oi=500),
    ("20260717", 575.0, "P"): _quote(575.0, "P", bid=0.35, ask=0.45, oi=400),
}


# --- pure math --------------------------------------------------------------------


def test_price_increment_spx_nickels() -> None:
    assert price_increment_for("SPX") == 0.05
    assert price_increment_for("SPXW") == 0.05
    assert price_increment_for("SPY") == 0.01


def test_combo_bid_ask_synthetic_nbbo() -> None:
    # sell 580P (bid 1.55/ask 1.65), buy 575P (bid 0.35/ask 0.45):
    # worst receive = 1.55 - 0.45 = 1.10; best = 1.65 - 0.35 = 1.30
    result = combo_bid_ask(LEGS, GOOD_QUOTES)
    assert result is not None
    bid, ask = result
    assert bid == pytest.approx(1.10)
    assert ask == pytest.approx(1.30)


def test_combo_bid_ask_none_when_leg_missing() -> None:
    quotes = {("20260717", 580.0, "P"): GOOD_QUOTES[("20260717", 580.0, "P")]}
    assert combo_bid_ask(LEGS, quotes) is None


def test_slippage_budget_min_of_frac_and_abs() -> None:
    # spread 0.20: 25% = 0.05 < abs cap 0.10 -> 0.05
    assert slippage_budget(1.10, 1.30, frac=0.25, abs_cap=0.10, increment=0.01) == (
        pytest.approx(0.05)
    )
    # wide spread 1.00: 25% = 0.25 > abs cap -> 0.10
    assert slippage_budget(0.5, 1.5, frac=0.25, abs_cap=0.10, increment=0.01) == (
        pytest.approx(0.10)
    )
    # tiny spread: never below one increment
    assert slippage_budget(1.0, 1.02, frac=0.25, abs_cap=0.10, increment=0.01) == (
        pytest.approx(0.01)
    )


def test_walk_targets_credit_descend_toward_floor() -> None:
    # decision mid +1.20 credit, budget 0.08, 4 steps: 1.18, 1.16, 1.14, 1.12
    prev = 1.20
    seen = []
    for step in (1, 2, 3, 4):
        prev = next_walk_target(
            decision_mid=1.20, current_mid=1.20, prev_target=prev, step=step,
            max_steps=4, budget=0.08, increment=0.01,
        )
        seen.append(prev)
    assert seen == [pytest.approx(1.18), pytest.approx(1.16),
                    pytest.approx(1.14), pytest.approx(1.12)]


def test_walk_targets_debit_pay_more() -> None:
    # decision mid -2.30 (debit), budget 0.08: net descends => paying more.
    target = next_walk_target(
        decision_mid=-2.30, current_mid=-2.30, prev_target=-2.30, step=4,
        max_steps=4, budget=0.08, increment=0.01,
    )
    assert target == pytest.approx(-2.38)


def test_walk_reanchors_to_current_mid() -> None:
    # Market improved (mid rose to 1.30): step 1 target re-anchors UP but
    # never above the previous target (monotonic toward marketable).
    target = next_walk_target(
        decision_mid=1.20, current_mid=1.30, prev_target=1.18, step=2,
        max_steps=4, budget=0.08, increment=0.01,
    )
    assert target == pytest.approx(1.18)  # clamped by prev_target

    # Market dropped hard: candidate would breach the decision-anchored
    # floor (1.20 - 0.08 = 1.12) -> clamped at the floor.
    target = next_walk_target(
        decision_mid=1.20, current_mid=1.00, prev_target=1.14, step=3,
        max_steps=4, budget=0.08, increment=0.01,
    )
    assert target == pytest.approx(1.12)


def test_walk_target_stale_quote_falls_back_to_decision_anchor() -> None:
    target = next_walk_target(
        decision_mid=1.20, current_mid=None, prev_target=1.18, step=2,
        max_steps=4, budget=0.08, increment=0.01,
    )
    assert target == pytest.approx(1.16)


def test_walk_target_rounds_to_increment() -> None:
    target = next_walk_target(
        decision_mid=1.237, current_mid=1.237, prev_target=1.237, step=1,
        max_steps=3, budget=0.10, increment=0.05,
    )
    assert (target / 0.05) == pytest.approx(round(target / 0.05))


# --- liquidity gates -----------------------------------------------------------------


_GATES = {"leg_spread_frac": 0.40, "leg_spread_floor": 0.20}


def test_liquidity_ok_on_good_quotes() -> None:
    assert liquidity_issues(LEGS, GOOD_QUOTES, min_open_interest=100, **_GATES) == []


def test_liquidity_flags_wide_spread() -> None:
    # 1.00/1.80 on a 1.40 mid = $0.80 spread = 57% of mid, over the 40% cap.
    quotes = dict(GOOD_QUOTES)
    quotes[("20260717", 580.0, "P")] = _quote(580.0, "P", bid=1.00, ask=1.80)
    issues = liquidity_issues(LEGS, quotes, min_open_interest=0, **_GATES)
    assert issues and "spread" in issues[0]


def test_liquidity_proportional_lets_pricey_liquid_leg_through() -> None:
    # A $0.55 spread on a $14 option = 4% of mid — fine, though it would have
    # failed the old absolute $0.50 cap. This is the juicy-trade rescue.
    quotes = dict(GOOD_QUOTES)
    quotes[("20260717", 580.0, "P")] = _quote(580.0, "P", bid=13.75, ask=14.30)
    assert liquidity_issues(LEGS, quotes, min_open_interest=0, **_GATES) == []


def test_liquidity_flags_missing_and_crossed() -> None:
    quotes = dict(GOOD_QUOTES)
    quotes[("20260717", 575.0, "P")] = _quote(575.0, "P", bid=None, ask=0.45)
    issues = liquidity_issues(LEGS, quotes, min_open_interest=0, **_GATES)
    assert issues and "no bid/ask" in issues[0]

    quotes[("20260717", 575.0, "P")] = _quote(575.0, "P", bid=0.50, ask=0.40)
    issues = liquidity_issues(LEGS, quotes, min_open_interest=0, **_GATES)
    assert issues and "crossed" in issues[0]


def test_liquidity_oi_gate_only_when_enabled_and_present() -> None:
    quotes = {
        ("20260717", 580.0, "P"): _quote(580.0, "P", bid=1.55, ask=1.65, oi=10),
        ("20260717", 575.0, "P"): _quote(575.0, "P", bid=0.35, ask=0.45, oi=None),
    }
    issues = liquidity_issues(LEGS, quotes, min_open_interest=100, **_GATES)
    assert len(issues) == 1  # low OI flagged; None OI tolerated
    assert "open interest" in issues[0]
    assert liquidity_issues(LEGS, quotes, min_open_interest=0, **_GATES) == []


def test_combo_spread_issue_economic_gate() -> None:
    from optionsbot.execution.walk import combo_spread_issue

    # GOOD_QUOTES combo: bid 1.10 / ask 1.30 -> spread 0.20. Net credit 1.20.
    # 0.20/1.20 = 17% < 35% cap -> fine.
    assert combo_spread_issue(LEGS, GOOD_QUOTES, 1.20, max_frac=0.35) is None
    # Same spread but tiny credit 0.40 -> 50% > cap -> rejected.
    issue = combo_spread_issue(LEGS, GOOD_QUOTES, 0.40, max_frac=0.35)
    assert issue is not None and "net premium" in issue
    # Wide combo (the XLK case): legs 1.00/2.60 + 0.10/2.06 -> combo spread huge.
    wide = {
        ("20260717", 580.0, "P"): _quote(580.0, "P", bid=1.00, ask=2.60),
        ("20260717", 575.0, "P"): _quote(575.0, "P", bid=0.10, ask=2.06),
    }
    assert combo_spread_issue(LEGS, wide, 1.20, max_frac=0.35) is not None


# --- journal --------------------------------------------------------------------------


def test_record_order_quotes_round_trips(tmp_db: Engine) -> None:
    with tmp_db.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread", legs_json=[],
            quantity=1, status="staged", staged_ts=NOW, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
    record_order_quotes(
        tmp_db, order_id, kind="decision", step=0, ts=NOW, combo_bid=1.10,
        combo_ask=1.30, combo_mid=1.20, target_net=1.20, limit_price=-1.20,
        legs=[{"strike": 580.0, "bid": 1.55, "ask": 1.65, "delayed": True}],
    )
    with tmp_db.connect() as conn:
        row = conn.execute(select(order_quotes)).one()
    assert row.order_id == order_id
    assert row.kind == "decision"
    assert row.combo_mid == pytest.approx(1.20)
    assert row.legs_json[0]["strike"] == 580.0


# --- walk runner ------------------------------------------------------------------------


def _walk_order(engine: Engine, status: str = "submitted") -> int:
    with engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=LEGS, quantity=1, status=status, staged_ts=NOW,
            submitted_ts=NOW, ib_order_id=11, limit_price=-1.20, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(order_ref=f"obot-{order_id}"))
    return order_id


def _walk_settings() -> Settings:
    s = Settings()
    s.execution.walk_step_seconds = 0  # no sleeping in tests
    s.execution.walk_max_steps = 3
    s.execution.walk_final_rest_seconds = 0
    return s


def _md(
    mids: dict[tuple[float, str], tuple[float, float]],
    *,
    delayed: bool = False,
) -> MagicMock:
    md = MagicMock()

    async def snap(symbol: str, expiry: str, strike: float, right: str) -> OptionQuote:
        bid, ask = mids[(strike, right)]
        return _quote(
            strike, right, bid=bid, ask=ask, ts=datetime.now(UTC), delayed=delayed
        )

    md.get_option_snapshot = AsyncMock(side_effect=snap)
    return md


def _tracker_confirms_cancel(engine: Engine, order_id: int) -> AsyncMock:
    """Simulate the tracker: the broker's Cancelled event lands right after
    the cancel request and moves the row terminal."""

    async def _cancel(ib_order_id: int) -> None:
        with engine.begin() as conn:
            conn.execute(
                update(orders).where(orders.c.id == order_id)
                .values(status="cancelled", terminal_ts=NOW)
            )

    return AsyncMock(side_effect=_cancel)


async def test_walk_exhaustion_requests_cancel_tracker_confirms(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
    md = _md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)})

    await run_price_walk(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, symbol="SPY", legs=LEGS, order_id=order_id, ib_order_id=11,
        decision_mid=1.20, budget=0.09, increment=0.01,
    )
    # 3 steps: 1.17, 1.14, 1.11 -> limits -1.17, -1.14, -1.11
    limits = [c.kwargs["new_limit_price"] for c in order_client.modify_price.call_args_list]
    assert limits == [pytest.approx(-1.17), pytest.approx(-1.14), pytest.approx(-1.11)]
    order_client.cancel.assert_awaited_once_with(11)
    record = get_order(tmp_db, order_id)
    assert record is not None
    # The TRACKER moved it terminal; the walk only annotated the intent.
    assert record.status == "cancelled"
    assert "walk exhausted" in (record.last_error or "")
    assert record.reprice_count == 3
    with tmp_db.connect() as conn:
        journal = conn.execute(select(order_quotes)).fetchall()
    assert len(journal) == 3  # one row per executed step
    assert {r.kind for r in journal} == {"step"}


async def test_walk_does_not_modify_from_delayed_quotes(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
    md = _md(
        {(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)},
        delayed=True,
    )

    await run_price_walk(
        engine=tmp_db,
        settings=_walk_settings(),
        order_client=order_client,
        md=md,
        symbol="SPY",
        legs=LEGS,
        order_id=order_id,
        ib_order_id=11,
        decision_mid=1.20,
        budget=0.09,
        increment=0.01,
    )

    order_client.modify_price.assert_not_awaited()
    order_client.cancel.assert_awaited_once_with(11)


async def test_walk_does_not_modify_without_quote_timestamps(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
    md = MagicMock()

    async def snapshot(
        symbol: str, expiry: str, strike: float, right: str
    ) -> OptionQuote:
        bid, ask = ({580.0: (1.55, 1.65), 575.0: (0.35, 0.45)})[strike]
        return _quote(strike, right, bid=bid, ask=ask, ts=None, delayed=False)

    md.get_option_snapshot = AsyncMock(side_effect=snapshot)

    await run_price_walk(
        engine=tmp_db,
        settings=_walk_settings(),
        order_client=order_client,
        md=md,
        symbol="SPY",
        legs=LEGS,
        order_id=order_id,
        ib_order_id=11,
        decision_mid=1.20,
        budget=0.09,
        increment=0.01,
    )

    order_client.modify_price.assert_not_awaited()
    order_client.cancel.assert_awaited_once_with(11)


@pytest.mark.parametrize("clock_offset_seconds", [46, -1])
async def test_walk_rejects_quotes_outside_exit_window_after_async_refresh(
    tmp_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
    clock_offset_seconds: int,
) -> None:
    from datetime import timedelta

    import optionsbot.execution.walk as walk_module

    class Clock(datetime):
        current = datetime(2026, 7, 10, 14, 0, tzinfo=UTC)

        @classmethod
        def now(cls, tz: object = None) -> Clock:
            return cls.fromtimestamp(cls.current.timestamp(), tz=UTC)

    quote_ts = Clock(2026, 7, 10, 14, 0, tzinfo=UTC)
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
    md = MagicMock()

    async def snapshot(
        symbol: str, expiry: str, strike: float, right: str
    ) -> OptionQuote:
        Clock.current = quote_ts + timedelta(seconds=clock_offset_seconds)
        bid, ask = ({580.0: (1.55, 1.65), 575.0: (0.35, 0.45)})[strike]
        return _quote(strike, right, bid=bid, ask=ask, ts=quote_ts, delayed=False)

    md.get_option_snapshot = AsyncMock(side_effect=snapshot)
    monkeypatch.setattr(walk_module, "datetime", Clock)
    settings = _walk_settings()
    settings.execution.entry_quote_max_age_seconds = 120
    settings.execution.exit_quote_max_age_seconds = 45

    await run_price_walk(
        engine=tmp_db,
        settings=settings,
        order_client=order_client,
        md=md,
        symbol="SPY",
        legs=LEGS,
        order_id=order_id,
        ib_order_id=11,
        decision_mid=1.20,
        budget=0.09,
        increment=0.01,
    )

    order_client.modify_price.assert_not_awaited()
    order_client.cancel.assert_awaited_once_with(11)


async def test_walk_does_not_modify_when_quote_refresh_fails(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
    md = MagicMock()
    md.get_option_snapshot = AsyncMock(side_effect=RuntimeError("quote unavailable"))

    await run_price_walk(
        engine=tmp_db,
        settings=_walk_settings(),
        order_client=order_client,
        md=md,
        symbol="SPY",
        legs=LEGS,
        order_id=order_id,
        ib_order_id=11,
        decision_mid=1.20,
        budget=0.09,
        increment=0.01,
    )

    order_client.modify_price.assert_not_awaited()
    order_client.cancel.assert_awaited_once_with(11)


async def test_walk_does_not_modify_from_incomplete_live_quotes(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
    md = MagicMock()

    async def snapshot(
        symbol: str, expiry: str, strike: float, right: str
    ) -> OptionQuote:
        if strike == 580.0:
            return _quote(
                strike,
                right,
                bid=None,
                ask=1.65,
                ts=datetime.now(UTC),
                delayed=False,
            )
        return _quote(
            strike,
            right,
            bid=0.35,
            ask=0.45,
            ts=datetime.now(UTC),
            delayed=False,
        )

    md.get_option_snapshot = AsyncMock(side_effect=snapshot)

    await run_price_walk(
        engine=tmp_db,
        settings=_walk_settings(),
        order_client=order_client,
        md=md,
        symbol="SPY",
        legs=LEGS,
        order_id=order_id,
        ib_order_id=11,
        decision_mid=1.20,
        budget=0.09,
        increment=0.01,
    )

    order_client.modify_price.assert_not_awaited()
    order_client.cancel.assert_awaited_once_with(11)


async def test_walk_debit_sends_ascending_positive_limits(tmp_db: Engine) -> None:
    # Debit structure: decision mid −2.00 (we pay). Walking = paying more, so
    # the BAG limits sent must be POSITIVE and ASCENDING (+2.03, +2.06, +2.09).
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = AsyncMock()
    md = _md({(580.0, "P"): (1.00, 1.10), (575.0, "P"): (3.00, 3.10)})  # net mid −2.00

    debit_legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 580.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
         "strike": 575.0, "right": "P", "quantity": 1},
    ]
    await run_price_walk(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, symbol="SPY", legs=debit_legs, order_id=order_id, ib_order_id=11,
        decision_mid=-2.00, budget=0.09, increment=0.01,
    )
    limits = [c.kwargs["new_limit_price"] for c in order_client.modify_price.call_args_list]
    assert limits == [pytest.approx(2.03), pytest.approx(2.06), pytest.approx(2.09)]
    assert limits == sorted(limits)  # strictly toward marketable


async def test_walk_stops_when_order_fills_midway(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    calls = {"n": 0}

    async def modify(ib_order_id: int, *, new_limit_price: float) -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # fill arrives after the second reprice
            with tmp_db.begin() as conn:
                conn.execute(update(orders).where(orders.c.id == order_id)
                             .values(status="filled", terminal_ts=NOW))

    order_client.modify_price = AsyncMock(side_effect=modify)
    order_client.cancel = AsyncMock()
    md = _md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)})

    await run_price_walk(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, symbol="SPY", legs=LEGS, order_id=order_id, ib_order_id=11,
        decision_mid=1.20, budget=0.09, increment=0.01,
    )
    assert order_client.modify_price.await_count == 2  # stopped after the fill
    order_client.cancel.assert_not_awaited()
    assert get_order(tmp_db, order_id).status == "filled"  # type: ignore[union-attr]


async def test_walk_noop_when_order_already_terminal(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db, status="cancelled")
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = AsyncMock()
    md = _md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)})
    await run_price_walk(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, symbol="SPY", legs=LEGS, order_id=order_id, ib_order_id=11,
        decision_mid=1.20, budget=0.09, increment=0.01,
    )
    order_client.modify_price.assert_not_awaited()
    order_client.cancel.assert_not_awaited()


async def test_walk_survives_modify_errors(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock(side_effect=RuntimeError("flaky"))
    order_client.cancel = _tracker_confirms_cancel(tmp_db, order_id)
    md = _md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)})
    await run_price_walk(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, symbol="SPY", legs=LEGS, order_id=order_id, ib_order_id=11,
        decision_mid=1.20, budget=0.09, increment=0.01,
    )
    # Errors tolerated per-step; the walk still completes and requests cancel.
    order_client.cancel.assert_awaited_once()
    assert get_order(tmp_db, order_id).status == "cancelled"  # type: ignore[union-attr]
