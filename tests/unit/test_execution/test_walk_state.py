"""Tests for walk-state persistence + resume (Work-stream D1)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import Engine, insert, update

from optionsbot.config import Settings
from optionsbot.execution.orders import (
    clear_walk_state,
    get_order,
    load_walk_states,
    upsert_walk_state,
)
from optionsbot.execution.walk import resume_walks, run_price_walk
from optionsbot.ibkr.types import OptionQuote
from optionsbot.storage.schema import orders

NOW = datetime(2026, 6, 11, 15, 30, tzinfo=UTC)

LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
]


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


def test_upsert_then_load_round_trips(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    upsert_walk_state(
        tmp_db, order_id, ib_order_id=11, symbol="SPY", legs=LEGS,
        decision_mid=1.20, budget=0.09, increment=0.01, step=2,
        prev_target=1.14, ts=NOW,
    )
    states = load_walk_states(tmp_db)
    assert len(states) == 1
    ws = states[0]
    assert ws.order_id == order_id
    assert ws.ib_order_id == 11
    assert ws.symbol == "SPY"
    assert ws.legs == LEGS
    assert ws.decision_mid == 1.20
    assert ws.budget == 0.09
    assert ws.increment == 0.01
    assert ws.step == 2
    assert ws.prev_target == 1.14


def test_upsert_is_idempotent_on_order_id(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    upsert_walk_state(
        tmp_db, order_id, ib_order_id=11, symbol="SPY", legs=LEGS,
        decision_mid=1.20, budget=0.09, increment=0.01, step=1,
        prev_target=1.17, ts=NOW,
    )
    upsert_walk_state(
        tmp_db, order_id, ib_order_id=11, symbol="SPY", legs=LEGS,
        decision_mid=1.20, budget=0.09, increment=0.01, step=3,
        prev_target=1.11, ts=NOW,
    )
    states = load_walk_states(tmp_db)
    assert len(states) == 1  # one row per order, last write wins
    assert states[0].step == 3
    assert states[0].prev_target == 1.11


def test_clear_removes_the_row(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    upsert_walk_state(
        tmp_db, order_id, ib_order_id=11, symbol="SPY", legs=LEGS,
        decision_mid=1.20, budget=0.09, increment=0.01, step=1,
        prev_target=1.17, ts=NOW,
    )
    clear_walk_state(tmp_db, order_id)
    assert load_walk_states(tmp_db) == []


def test_load_skips_terminal_orders(tmp_db: Engine) -> None:
    # A walk_state row whose order already went terminal must not be resumed.
    order_id = _walk_order(tmp_db)
    upsert_walk_state(
        tmp_db, order_id, ib_order_id=11, symbol="SPY", legs=LEGS,
        decision_mid=1.20, budget=0.09, increment=0.01, step=1,
        prev_target=1.17, ts=NOW,
    )
    with tmp_db.begin() as conn:
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(status="filled", terminal_ts=NOW))
    assert get_order(tmp_db, order_id).status == "filled"  # type: ignore[union-attr]
    assert load_walk_states(tmp_db) == []


def _quote(
    strike: float, right: str, *, bid: float, ask: float,
) -> OptionQuote:
    return OptionQuote(
        symbol="SPY", expiry="20260717", strike=strike, right=right,  # type: ignore[arg-type]
        bid=bid, ask=ask, last=None, mid=(bid + ask) / 2, iv=None, delta=None,
        gamma=None, theta=None, vega=None, open_interest=None, volume=None,
        ts=datetime.now(UTC), delayed=False,
    )


def _walk_settings() -> Settings:
    s = Settings()
    s.execution.walk_step_seconds = 0
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


async def test_walk_persists_state_each_step_then_clears(tmp_db: Engine) -> None:
    order_id = _walk_order(tmp_db)
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()

    async def _cancel(ib_order_id: int) -> None:
        with tmp_db.begin() as conn:
            conn.execute(update(orders).where(orders.c.id == order_id)
                         .values(status="cancelled", terminal_ts=NOW))
    order_client.cancel = AsyncMock(side_effect=_cancel)
    md = _md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)})

    await run_price_walk(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, symbol="SPY", legs=LEGS, order_id=order_id, ib_order_id=11,
        decision_mid=1.20, budget=0.09, increment=0.01,
    )
    # Walk ran to exhaustion + cancel: the walk-state row must be cleaned up.
    assert load_walk_states(tmp_db) == []


async def test_resume_repumps_a_persisted_walk_after_restart(tmp_db: Engine) -> None:
    # Simulate: a walk persisted state at step 1, then the daemon died.
    order_id = _walk_order(tmp_db)
    upsert_walk_state(
        tmp_db, order_id, ib_order_id=11, symbol="SPY", legs=LEGS,
        decision_mid=1.20, budget=0.09, increment=0.01, step=1,
        prev_target=1.17, ts=NOW,
    )
    order_client = MagicMock()
    order_client.modify_price = AsyncMock()

    async def _cancel(ib_order_id: int) -> None:
        with tmp_db.begin() as conn:
            conn.execute(update(orders).where(orders.c.id == order_id)
                         .values(status="cancelled", terminal_ts=NOW))
    order_client.cancel = AsyncMock(side_effect=_cancel)
    md = _md({(580.0, "P"): (1.55, 1.65), (575.0, "P"): (0.35, 0.45)})
    sent: list[str] = []

    async def notify(text: str) -> None:
        sent.append(text)

    walk_tasks: set[asyncio.Task[None]] = set()
    n = await resume_walks(
        engine=tmp_db, settings=_walk_settings(), order_client=order_client,
        md=md, walk_tasks=walk_tasks, notify=notify,
    )
    assert n == 1
    # Drain the spawned walk task(s) so assertions see the final state.
    if walk_tasks:
        await asyncio.gather(*walk_tasks, return_exceptions=True)
    # Resume continues from step 2: it must NOT replay step 1's reprice and
    # must drive the order toward marketable, then exhaust + clear state.
    assert order_client.modify_price.await_count >= 1
    assert load_walk_states(tmp_db) == []
    assert any("re-attached" in m or "resum" in m.lower() for m in sent)
