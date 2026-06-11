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
) -> OptionQuote:
    computed_mid = mid if mid is not None else (
        (bid + ask) / 2 if bid is not None and ask is not None else None
    )
    return OptionQuote(
        symbol="SPY", expiry="20260717", strike=strike, right=right,  # type: ignore[arg-type]
        bid=bid, ask=ask, last=None, mid=computed_mid, iv=None, delta=None,
        gamma=None, theta=None, vega=None, open_interest=oi, volume=None,
        ts=NOW, delayed=True,
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


def test_liquidity_ok_on_good_quotes() -> None:
    assert liquidity_issues(
        LEGS, GOOD_QUOTES, max_leg_spread=0.50, min_open_interest=100
    ) == []


def test_liquidity_flags_wide_spread() -> None:
    quotes = dict(GOOD_QUOTES)
    quotes[("20260717", 580.0, "P")] = _quote(580.0, "P", bid=1.00, ask=1.80)
    issues = liquidity_issues(LEGS, quotes, max_leg_spread=0.50, min_open_interest=0)
    assert issues and "spread" in issues[0]


def test_liquidity_flags_missing_and_crossed() -> None:
    quotes = dict(GOOD_QUOTES)
    quotes[("20260717", 575.0, "P")] = _quote(575.0, "P", bid=None, ask=0.45)
    issues = liquidity_issues(LEGS, quotes, max_leg_spread=0.50, min_open_interest=0)
    assert issues and "no bid/ask" in issues[0]

    quotes[("20260717", 575.0, "P")] = _quote(575.0, "P", bid=0.50, ask=0.40)
    issues = liquidity_issues(LEGS, quotes, max_leg_spread=0.50, min_open_interest=0)
    assert issues and "crossed" in issues[0]


def test_liquidity_oi_gate_only_when_enabled_and_present() -> None:
    quotes = {
        ("20260717", 580.0, "P"): _quote(580.0, "P", bid=1.55, ask=1.65, oi=10),
        ("20260717", 575.0, "P"): _quote(575.0, "P", bid=0.35, ask=0.45, oi=None),
    }
    issues = liquidity_issues(LEGS, quotes, max_leg_spread=0.50, min_open_interest=100)
    assert len(issues) == 1  # low OI flagged; None OI tolerated
    assert "open interest" in issues[0]
    assert liquidity_issues(LEGS, quotes, max_leg_spread=0.50, min_open_interest=0) == []


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


def _md(mids: dict[tuple[float, str], tuple[float, float]]) -> MagicMock:
    md = MagicMock()

    async def snap(symbol: str, expiry: str, strike: float, right: str) -> OptionQuote:
        bid, ask = mids[(strike, right)]
        return _quote(strike, right, bid=bid, ask=ask)

    md.get_option_snapshot = AsyncMock(side_effect=snap)
    return md


async def test_walk_exhaustion_cancels_and_abandons(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()
    order_client.cancel = AsyncMock()
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
    assert record.status == "abandoned"
    assert record.reprice_count == 3
    with tmp_db.connect() as conn:
        journal = conn.execute(select(order_quotes)).fetchall()
    assert len(journal) == 3  # one row per executed step
    assert {r.kind for r in journal} == {"step"}


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
    order_client.cancel = AsyncMock()
    md = _md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)})
    await run_price_walk(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, symbol="SPY", legs=LEGS, order_id=order_id, ib_order_id=11,
        decision_mid=1.20, budget=0.09, increment=0.01,
    )
    # Errors tolerated per-step; the walk still completes and abandons.
    order_client.cancel.assert_awaited_once()
    assert get_order(tmp_db, order_id).status == "abandoned"  # type: ignore[union-attr]
