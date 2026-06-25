"""Tests for the exit runner tick (IBK-129)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import insert, select, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.exit_runner import force_close_entry, run_exits_tick
from optionsbot.execution.orders import record_fill
from optionsbot.execution.state import load_state, trip_kill
from optionsbot.ibkr.types import OptionQuote, PlacedOrder, PortfolioPosition
from optionsbot.storage.schema import orders

NOW = datetime.now(UTC)
FAR = (NOW + timedelta(days=36)).strftime("%Y%m%d")
NEAR = (NOW + timedelta(days=2)).strftime("%Y%m%d")


@pytest.fixture(autouse=True)
def _force_market_open():
    """``run_exits_tick`` short-circuits when NYSE is closed (exit_runner.py).

    Every test in this module asserts open-market exit behavior, so without
    pinning the gate they only pass during live US trading hours and fail (or
    pass vacuously with ``positions=0``) the rest of the day. Force the gate
    open so the suite is deterministic regardless of wall-clock time.
    """
    with patch("optionsbot.daemon.exit_runner.is_market_open", return_value=True):
        yield


def _legs(expiry: str) -> list[dict[str, object]]:
    return [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": expiry,
         "strike": 580.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": expiry,
         "strike": 575.0, "right": "P", "quantity": 1},
    ]


def _filled_entry(context: DaemonContext, *, expiry: str = FAR) -> int:
    engine = context.engine
    with engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=_legs(expiry), quantity=1, status="filled", staged_ts=NOW,
            submitted_ts=NOW, terminal_ts=NOW, ib_order_id=11, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == order_id)
                     .values(order_ref=f"obot-{order_id}"))
    record_fill(engine, order_id, exec_id=f"x{order_id}a", side="SELL",
                price=1.60, qty=1, ts=NOW)
    record_fill(engine, order_id, exec_id=f"x{order_id}b", side="BUY",
                price=0.40, qty=1, ts=NOW)
    return order_id


def _quote(strike: float, right: str, mid: float) -> OptionQuote:
    return OptionQuote(
        symbol="SPY", expiry=FAR, strike=strike, right=right,  # type: ignore[arg-type]
        bid=round(mid - 0.05, 4), ask=round(mid + 0.05, 4), last=None, mid=mid,
        iv=None, delta=None, gamma=None, theta=None, vega=None,
        open_interest=None, volume=None, ts=NOW, delayed=True,
    )


def _wire(context: DaemonContext, mids: dict[tuple[float, str], float]) -> MagicMock:
    context.settings.execution.enabled = True
    context.settings.execution.walk_max_steps = 0  # no walk task in tests
    order_client = MagicMock()
    order_client.place_combo_limit = AsyncMock(
        side_effect=lambda *a, **k: PlacedOrder(
            ib_order_id=99, order_ref=k["order_ref"], action="BUY",
            limit_price=k["limit_price"], quantity=k["quantity"],
        )
    )
    context.order_client = order_client

    md = MagicMock()
    md.get_option_snapshot = AsyncMock(
        side_effect=lambda symbol, expiry, strike, right: _quote(
            strike, right, mids[(strike, right)]
        )
    )
    context._test_md = md  # type: ignore[attr-defined]
    return order_client


async def test_take_profit_fires_closing_order(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    # Entry credit 1.20; structure now reopens at 0.50 -> kept 58% -> close.
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 1
    call = order_client.place_combo_limit.call_args
    # Flipped close: we PAY ~0.50/unit -> BUY-bag positive limit.
    assert call.kwargs["limit_price"] > 0
    with daemon_context.engine.connect() as conn:
        close = conn.execute(
            select(orders).where(orders.c.intent == "close")
        ).one()
    assert close.closes_order_id == entry_id
    assert close.status == "submitted"
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("closing" in m.lower() for m in sent)


async def test_no_trigger_no_close(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()


async def test_active_close_blocks_duplicate(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)
    assert first.closes_submitted == 1
    assert second.closes_submitted == 0
    assert order_client.place_combo_limit.await_count == 1


async def test_kill_switch_blocks_exits(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    trip_kill(daemon_context.engine, "halt")
    summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()


async def test_expiry_guard_forces_close(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context, expiry=NEAR)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()


async def test_half_closed_position_hands_off_instead_of_reclosing(
    daemon_context: DaemonContext,
) -> None:
    # Opus IBK-129 critical: a close that PARTIALLY filled then died must
    # never be auto-restaged at full quantity (over-close = wrong-way risk).
    entry_id = _filled_entry(daemon_context)
    with daemon_context.engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="close", closes_order_id=entry_id, symbol="SPY",
            strategy="bull_put_spread", legs_json=_legs(FAR), quantity=1,
            status="abandoned", staged_ts=NOW, submitted_ts=NOW,
            terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        close_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == close_id)
                     .values(order_ref=f"obot-{close_id}"))
    record_fill(daemon_context.engine, close_id, exec_id="half1", side="BUY",
                price=0.80, qty=1, ts=NOW)

    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)
    assert first.closes_submitted == 0
    assert second.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    warns = [
        c.args[0] for c in daemon_context.telegram.send_message.await_args_list
        if "HALF-CLOSED" in c.args[0]
    ]
    assert len(warns) == 1  # escalated exactly once


async def test_missing_quote_skips_and_retries_later(
    daemon_context: DaemonContext,
) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {})
    md = MagicMock()
    md.get_option_snapshot = AsyncMock(side_effect=RuntimeError("no data"))
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 0
    assert summary.errors >= 0  # tick survives
    order_client.place_combo_limit.assert_not_awaited()


async def test_stale_quote_suppresses_take_profit_and_alerts(
    daemon_context: DaemonContext,
) -> None:
    # TP would fire (kept 58% of credit), but the quotes are older than the
    # freshness threshold -> the quote-priced exit is suppressed and a
    # staleness alert is sent exactly once. No close is placed.
    _filled_entry(daemon_context)
    daemon_context.settings.execution.exit_quote_max_age_seconds = 30
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})

    stale_ts = NOW - timedelta(seconds=120)
    md = MagicMock()

    def _stale_quote(symbol: str, expiry: str, strike: float, right: str) -> OptionQuote:
        mid = {(580.0, "P"): 0.80, (575.0, "P"): 0.30}[(strike, right)]
        return OptionQuote(
            symbol="SPY", expiry=FAR, strike=strike, right=right,  # type: ignore[arg-type]
            bid=round(mid - 0.05, 4), ask=round(mid + 0.05, 4), last=None, mid=mid,
            iv=None, delta=None, gamma=None, theta=None, vega=None,
            open_interest=None, volume=None, ts=stale_ts, delayed=True,
        )

    md.get_option_snapshot = AsyncMock(side_effect=_stale_quote)
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)

    assert first.closes_submitted == 0
    assert second.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    stale_alerts = [
        c.args[0] for c in daemon_context.telegram.send_message.await_args_list
        if "stale" in c.args[0].lower()
    ]
    assert len(stale_alerts) == 1  # escalated exactly once


async def test_non_atomic_close_fails_safe_and_halts(
    daemon_context: DaemonContext,
) -> None:
    # Phase 0 C2: if the staged close is not the exact inverse of the entry
    # (here: a leg went missing), the runner must NOT place it. It fails safe:
    # trips the kill switch, alerts, places nothing.
    from optionsbot.daemon import exit_runner as er
    from optionsbot.execution.orders import stage_close_order as real_stage
    from optionsbot.execution.state import load_state

    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})

    def _drop_a_leg(engine, entry, *, now=None):  # type: ignore[no-untyped-def]
        close = real_stage(engine, entry, now=now)
        # Simulate a staging defect: only one option leg survives -> a
        # single-leg route that would strand the other side.
        object.__setattr__(close, "legs", close.legs[:1])
        return close

    with (
        patch("optionsbot.daemon.exit_runner._exec_md",
              return_value=daemon_context._test_md),  # type: ignore[attr-defined]
        patch.object(er, "stage_close_order", _drop_a_leg),
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    assert load_state(daemon_context.engine).killed is True
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("naked" in m.lower() or "atomic" in m.lower() for m in sent)


async def test_stale_quote_still_allows_expiry_guard(
    daemon_context: DaemonContext,
) -> None:
    # Even with stale quotes, the time-based expiry guard must still force a
    # close (assignment/pin risk is not a function of quote freshness).
    _filled_entry(daemon_context, expiry=NEAR)
    daemon_context.settings.execution.exit_quote_max_age_seconds = 30
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})

    stale_ts = NOW - timedelta(seconds=600)
    md = MagicMock()

    def _stale_quote(symbol: str, expiry: str, strike: float, right: str) -> OptionQuote:
        mid = {(580.0, "P"): 1.40, (575.0, "P"): 0.30}[(strike, right)]
        return OptionQuote(
            symbol="SPY", expiry=NEAR, strike=strike, right=right,  # type: ignore[arg-type]
            bid=round(mid - 0.05, 4), ask=round(mid + 0.05, 4), last=None, mid=mid,
            iv=None, delta=None, gamma=None, theta=None, vega=None,
            open_interest=None, volume=None, ts=stale_ts, delayed=True,
        )

    md.get_option_snapshot = AsyncMock(side_effect=_stale_quote)
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()


def _half_closed_entry_with_abandoned_close(
    context: DaemonContext, *, expiry: str = FAR
) -> int:
    """Create a filled open entry whose only close is terminal+partial-filled
    (abandoned with one fill) — the condition that triggers the post-close
    naked-short sweep in ``run_exits_tick``."""
    entry_id = _filled_entry(context, expiry=expiry)
    engine = context.engine
    with engine.begin() as conn:
        pk = conn.execute(insert(orders).values(
            intent="close", closes_order_id=entry_id, symbol="SPY",
            strategy="bull_put_spread", legs_json=_legs(expiry), quantity=1,
            status="abandoned", staged_ts=NOW, submitted_ts=NOW,
            terminal_ts=NOW, reprice_count=0,
        )).inserted_primary_key
        assert pk is not None
        close_id = int(pk[0])
        conn.execute(update(orders).where(orders.c.id == close_id)
                     .values(order_ref=f"obot-{close_id}"))
    # Partial fill on the abandoned close — this is what _half_closed() detects.
    record_fill(engine, close_id, exec_id="half1", side="BUY",
                price=0.80, qty=1, ts=NOW)
    return entry_id


def _residual_short_position(expiry: str = FAR) -> PortfolioPosition:
    """A broker PortfolioPosition representing the sold 580P leg still open
    (position < 0 → naked short) after the close failed to fully fill."""
    return PortfolioPosition(
        account="DU123456",
        symbol="SPY",
        sec_type="OPT",
        expiry=expiry,
        strike=580.0,
        right="P",
        multiplier=100,
        position=-1.0,  # still short after partial close
        avg_cost=1.60,
        market_price=None,
        market_value=None,
        unrealized_pnl=None,
        realized_pnl=None,
    )


async def test_post_close_naked_short_trips_kill_and_alerts_p1(
    daemon_context: DaemonContext,
) -> None:
    """Integration: run_exits_tick → assert_no_naked_short_after_close → P1.

    A half-closed entry (abandoned close with a partial fill, no active close)
    whose broker positions still show a residual SHORT 580P leg must cause the
    exit tick to:
      1. Trip the kill switch.
      2. Send exactly one "🛑"/"P1" Telegram alert.
      3. On a second tick for the SAME entry, NOT re-trip or re-alert (naked_leg_halted
         set deduplicates).

    Discriminating: if the ``await assert_no_naked_short_after_close(context, entry)``
    call were removed from ``run_exits_tick``, ``load_state(...).killed`` would remain
    False and no P1 telegram message would be sent, causing both ``assert killed is True``
    and the P1-alert-count assertion to fail.
    """
    entry_id = _half_closed_entry_with_abandoned_close(daemon_context)

    # Wire exec_ibkr so the sweep doesn't short-circuit on "no exec connection".
    exec_ibkr_mock = MagicMock()
    daemon_context.exec_ibkr = exec_ibkr_mock

    # _wire sets execution.enabled=True and provides the market-data mock.
    _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})

    residual = _residual_short_position(FAR)

    with (
        patch("optionsbot.daemon.exit_runner._exec_md",
              return_value=daemon_context._test_md),  # type: ignore[attr-defined]
        patch(
            "optionsbot.ibkr.positions.PositionsClient",
            autospec=True,
        ) as MockPositionsClient,
    ):
        mock_pc_instance = MagicMock()
        mock_pc_instance.get_portfolio = AsyncMock(return_value=[residual])
        MockPositionsClient.return_value = mock_pc_instance

        # --- First tick: naked short detected ---
        first = await run_exits_tick(daemon_context)
        # Kill switch must be tripped.
        assert load_state(daemon_context.engine).killed is True
        # Exactly one P1 alert must have been sent.
        all_messages = [
            c.args[0] for c in daemon_context.telegram.send_message.await_args_list
        ]
        p1_alerts = [m for m in all_messages if "P1" in m or "🛑" in m]
        assert len(p1_alerts) == 1, f"expected 1 P1 alert on first tick, got: {p1_alerts}"
        # The entry is now tracked in the dedup set.
        assert entry_id in daemon_context.naked_leg_halted

        # --- Second tick: naked_leg_halted dedup suppresses re-alert ---
        # After the kill is tripped, can_execute() blocks the whole tick (gate check
        # runs before any sweep). Reset killed so the tick can reach the sweep again —
        # the only dedup barrier we want to exercise here is naked_leg_halted.
        with daemon_context.engine.begin() as conn:
            from optionsbot.storage.schema import execution_state as _es
            conn.execute(_es.update().values(killed=0))

        alert_count_before = len(daemon_context.telegram.send_message.await_args_list)
        _second = await run_exits_tick(daemon_context)
        new_p1_alerts = [
            c.args[0]
            for c in daemon_context.telegram.send_message.await_args_list[alert_count_before:]
            if "P1" in c.args[0] or "🛑" in c.args[0]
        ]
        assert new_p1_alerts == [], (
            f"second tick must not re-alert via naked_leg_halted, got: {new_p1_alerts}"
        )

        # Broker was queried on both ticks (sweep still reads portfolio each tick;
        # naked_leg_halted suppresses the alert but not the broker read itself).
        assert mock_pc_instance.get_portfolio.await_count == 2

        # No close order was submitted on either tick.
        assert first.closes_submitted == 0


# ---- /close: human-initiated force close (force_close_entry) ----


async def test_force_close_fires_without_a_trigger(daemon_context: DaemonContext) -> None:
    # The whole point of /close: it bypasses evaluate_exit. Use mids where the
    # take-profit rule would NOT fire (same mids as test_no_trigger_no_close) and
    # assert a close is placed anyway.
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        msg = await force_close_entry(daemon_context, entry_id)
    order_client.place_combo_limit.assert_awaited_once()
    with daemon_context.engine.connect() as conn:
        close = conn.execute(select(orders).where(orders.c.intent == "close")).one()
    assert close.closes_order_id == entry_id
    assert close.status == "submitted"
    assert str(entry_id) in msg


async def test_force_close_unknown_id_is_reported(daemon_context: DaemonContext) -> None:
    _wire(daemon_context, {})
    msg = await force_close_entry(daemon_context, 999)
    assert "unknown" in msg.lower()


async def test_force_close_rejects_non_open_entry(daemon_context: DaemonContext) -> None:
    # A cancelled (non-filled) order is not a closable position.
    entry_id = _filled_entry(daemon_context)
    with daemon_context.engine.begin() as conn:
        conn.execute(update(orders).where(orders.c.id == entry_id)
                     .values(status="cancelled"))
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    msg = await force_close_entry(daemon_context, entry_id)
    order_client.place_combo_limit.assert_not_awaited()
    assert "filled open" in msg.lower()


async def test_force_close_blocks_when_already_closing(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await force_close_entry(daemon_context, entry_id)
        second = await force_close_entry(daemon_context, entry_id)
    assert order_client.place_combo_limit.await_count == 1
    assert str(entry_id) in first
    assert "already closing" in second.lower()


async def test_force_close_respects_kill_switch(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    trip_kill(daemon_context.engine, "halt")
    msg = await force_close_entry(daemon_context, entry_id)
    order_client.place_combo_limit.assert_not_awaited()
    assert "kill" in msg.lower()


async def test_force_close_refused_when_market_closed(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner.is_market_open", return_value=False):
        msg = await force_close_entry(daemon_context, entry_id)
    order_client.place_combo_limit.assert_not_awaited()
    assert "market" in msg.lower()


async def test_force_close_without_order_client(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    # order_client stays None (fixture default) -> not configured.
    msg = await force_close_entry(daemon_context, entry_id)
    assert "not configured" in msg.lower()


async def test_force_close_with_stale_quotes_places_and_skips_deferred_alert(
    daemon_context: DaemonContext,
) -> None:
    # P1 (Opus review): a forced /close with STALE quotes must STILL place the
    # close (priced off entry-net; the walk re-anchors) AND must NOT emit the
    # "TP/stop deferred" staleness alert — that message is false on the forced path.
    entry_id = _filled_entry(daemon_context)
    daemon_context.settings.execution.exit_quote_max_age_seconds = 30
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})

    stale_ts = NOW - timedelta(seconds=120)
    md = MagicMock()

    def _stale_quote(symbol: str, expiry: str, strike: float, right: str) -> OptionQuote:
        mid = {(580.0, "P"): 0.80, (575.0, "P"): 0.30}[(strike, right)]
        return OptionQuote(
            symbol="SPY", expiry=FAR, strike=strike, right=right,  # type: ignore[arg-type]
            bid=round(mid - 0.05, 4), ask=round(mid + 0.05, 4), last=None, mid=mid,
            iv=None, delta=None, gamma=None, theta=None, vega=None,
            open_interest=None, volume=None, ts=stale_ts, delayed=True,
        )

    md.get_option_snapshot = AsyncMock(side_effect=_stale_quote)
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        msg = await force_close_entry(daemon_context, entry_id)

    order_client.place_combo_limit.assert_awaited_once()  # close DID place
    assert str(entry_id) in msg
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert not any("stale" in m.lower() or "deferred" in m.lower() for m in sent)


async def test_force_close_on_half_closed_hands_off(daemon_context: DaemonContext) -> None:
    # A half-closed position (abandoned close with a partial fill) must NOT be
    # re-closed by /close either — the _half_closed guard hands off to the human.
    entry_id = _half_closed_entry_with_abandoned_close(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        msg = await force_close_entry(daemon_context, entry_id)
    order_client.place_combo_limit.assert_not_awaited()
    assert "no close placed" in msg.lower()


async def test_close_command_replay_is_idempotent_end_to_end(
    daemon_context: DaemonContext,
) -> None:
    # Replay-safety (Opus follow-up): dispatching /close twice — as a restart
    # backlog replay would — must place EXACTLY ONE close. Exercises the full
    # seam: dispatch -> _cmd_close -> force_close_entry -> open_close_for guard.
    from optionsbot.daemon.commands import dispatch

    daemon_context.settings.telegram.chat_id = "5356256463"
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md",
               return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await dispatch(daemon_context, f"/close {entry_id}")
        second = await dispatch(daemon_context, f"/close {entry_id}")
    assert order_client.place_combo_limit.await_count == 1  # exactly one close placed
    assert str(entry_id) in first[0].text
    assert "already closing" in second[0].text.lower()
