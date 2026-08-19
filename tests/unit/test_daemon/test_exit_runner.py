"""Tests for the exit runner tick (IBK-129)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import insert, select, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.exit_runner import (
    _exec_md,
    _manage_entry,
    _opening_range_exit_plan,
    _settle_cleared_expirations,
    assert_no_naked_short_after_close,
    force_close_entry,
    run_exits_tick,
)
from optionsbot.daemon.market_hours import nyse_session_date
from optionsbot.execution.equity_guard import capture_day_start_net_liq
from optionsbot.execution.exit_requests import HermesLossCapDecision
from optionsbot.execution.orders import get_order, record_fill, set_fill_commission
from optionsbot.execution.state import load_state, trip_kill
from optionsbot.ibkr.types import OptionQuote, PlacedOrder, PortfolioPosition
from optionsbot.storage.schema import (
    exit_requests,
    orders,
    pick_outcomes,
    position_settlements,
    snapshots,
    strategy_scores,
)

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
        {
            "symbol": "SPY",
            "side": "sell",
            "sec_type": "OPT",
            "expiry": expiry,
            "strike": 580.0,
            "right": "P",
            "quantity": 1,
            "con_id": 580001,
            "multiplier": 100,
            "currency": "USD",
        },
        {
            "symbol": "SPY",
            "side": "buy",
            "sec_type": "OPT",
            "expiry": expiry,
            "strike": 575.0,
            "right": "P",
            "quantity": 1,
            "con_id": 575001,
            "multiplier": 100,
            "currency": "USD",
        },
    ]


def _filled_entry(context: DaemonContext, *, expiry: str = FAR) -> int:
    engine = context.engine
    with engine.begin() as conn:
        pk = conn.execute(
            insert(orders).values(
                intent="open",
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=_legs(expiry),
                quantity=1,
                status="filled",
                staged_ts=NOW,
                submitted_ts=NOW,
                terminal_ts=NOW,
                reprice_count=0,
            )
        ).inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(
            update(orders)
            .where(orders.c.id == order_id)
            .values(order_ref=f"obot-{order_id}", ib_order_id=10 + order_id)
        )
    record_fill(
        engine,
        order_id,
        exec_id=f"x{order_id}a",
        side="SELL",
        price=1.60,
        qty=1,
        ts=NOW,
        leg_con_id=580001,
    )
    record_fill(
        engine,
        order_id,
        exec_id=f"x{order_id}b",
        side="BUY",
        price=0.40,
        qty=1,
        ts=NOW,
        leg_con_id=575001,
    )
    set_fill_commission(engine, f"x{order_id}a", 0.65)
    set_fill_commission(engine, f"x{order_id}b", 0.65)
    return order_id


def _quote(
    strike: float,
    right: str,
    mid: float,
    *,
    delayed: bool | None = False,
    ts: datetime | None = None,
) -> OptionQuote:
    return OptionQuote(
        symbol="SPY",
        expiry=FAR,
        strike=strike,
        right=right,  # type: ignore[arg-type]
        bid=round(mid - 0.05, 4),
        ask=round(mid + 0.05, 4),
        last=None,
        mid=mid,
        iv=None,
        delta=None,
        gamma=None,
        theta=None,
        vega=None,
        open_interest=None,
        volume=None,
        ts=ts or datetime.now(UTC),
        delayed=delayed,  # type: ignore[arg-type]
    )


def _wire(
    context: DaemonContext,
    mids: dict[tuple[float, str], float],
    *,
    delayed: bool | None = False,
    quote_ts: datetime | None = None,
) -> MagicMock:
    context.settings.execution.enabled = True
    context.settings.execution.walk_max_steps = 0  # no walk task in tests
    order_client = MagicMock()
    order_client.place_combo_limit = AsyncMock(
        side_effect=lambda *a, **k: PlacedOrder(
            ib_order_id=99,
            order_ref=k["order_ref"],
            action="BUY",
            limit_price=k["limit_price"],
            quantity=k["quantity"],
            leg_contracts=((580001, 100, "USD"), (575001, 100, "USD")),
        )
    )
    context.order_client = order_client

    md = MagicMock()
    md.get_option_snapshot = AsyncMock(
        side_effect=lambda symbol, expiry, strike, right: _quote(
            strike, right, mids[(strike, right)], delayed=delayed, ts=quote_ts
        )
    )
    context._test_md = md  # type: ignore[attr-defined]
    return order_client


def _capture_loss_baseline(context: DaemonContext, net_liq: float = 10_000.0) -> None:
    capture_day_start_net_liq(
        context.engine,
        net_liq,
        session=nyse_session_date(NOW).isoformat(),
    )


def test_exit_quotes_use_the_single_daemon_market_data_session(
    daemon_context: DaemonContext,
) -> None:
    """The execution client owns orders only; it must not compete for live quotes."""
    daemon_context.ibkr = MagicMock()
    daemon_context.exec_ibkr = MagicMock()

    md = _exec_md(daemon_context)

    assert md is not None
    assert md.client is daemon_context.ibkr
    assert md.client is not daemon_context.exec_ibkr


async def test_opening_range_cost_floor_reaches_exit_evaluator(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=NOW,
                    spot=600.0,
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    legs_json=_legs(FAR),
                    suggestion_json={
                        "estimated_round_trip_cost": 16.40,
                        "opening_range_fvg": {
                            "status": "entry_confirmed",
                            "stop_pct": 0.15,
                            "target_pct": 0.30,
                        },
                    },
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            update(orders)
            .where(orders.c.id == entry_id)
            .values(strategy_score_id=score_id)
        )
    entry = get_order(daemon_context.engine, entry_id)
    assert entry is not None
    plan = _opening_range_exit_plan(daemon_context.engine, entry)
    assert plan is not None
    assert plan.estimated_round_trip_cost_per_unit == pytest.approx(0.164)
    _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})

    with patch("optionsbot.daemon.exit_runner.evaluate_exit", return_value=None) as evaluate:
        submitted = await _manage_entry(
            daemon_context,
            daemon_context._test_md,  # type: ignore[attr-defined]
            entry,
            NOW,
        )

    assert submitted == 0
    assert evaluate.call_args.kwargs["debit_stop_pct_override"] == 0.15
    assert evaluate.call_args.kwargs["debit_take_profit_pct_override"] == 0.30
    assert evaluate.call_args.kwargs["debit_round_trip_cost_override"] == pytest.approx(
        0.164
    )


async def test_cleared_all_otm_expiration_is_settled_from_terminal_outcome(
    daemon_context: DaemonContext,
) -> None:
    expiry = (nyse_session_date(NOW) - timedelta(days=1)).strftime("%Y%m%d")
    entry_id = _filled_entry(daemon_context, expiry=expiry)
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=NOW,
                    spot=600.0,
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    legs_json=_legs(expiry),
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            update(orders)
            .where(orders.c.id == entry_id)
            .values(strategy_score_id=score_id)
        )
        conn.execute(
            insert(pick_outcomes).values(
                strategy_score_id=score_id,
                symbol="SPY",
                strategy="bull_put_spread",
                expiry=expiry,
                entry_spot=590.0,
                terminal_spot=600.0,
                realized_pnl=120.0,
                win=1,
                evaluated_at=NOW,
            )
        )
    entry = get_order(daemon_context.engine, entry_id)
    assert entry is not None
    daemon_context.exec_ibkr = MagicMock()

    with patch(
        "optionsbot.ibkr.positions.PositionsClient",
        autospec=True,
    ) as positions_client:
        positions_client.return_value.get_portfolio = AsyncMock(return_value=[])
        settled = await _settle_cleared_expirations(
            daemon_context,
            [entry],
            NOW,
        )

    assert settled == 1
    with daemon_context.engine.connect() as conn:
        row = conn.execute(
            select(position_settlements).where(
                position_settlements.c.entry_order_id == entry_id
            )
        ).one()
    assert row.pnl == pytest.approx(118.70)
    assert row.settled_at.replace(tzinfo=UTC) == NOW


async def test_cleared_itm_expiration_settles_and_halts_on_assignment(
    daemon_context: DaemonContext,
) -> None:
    expiry = (nyse_session_date(NOW) - timedelta(days=1)).strftime("%Y%m%d")
    entry_id = _filled_entry(daemon_context, expiry=expiry)
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=NOW,
                    spot=577.0,
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    legs_json=_legs(expiry),
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            update(orders)
            .where(orders.c.id == entry_id)
            .values(strategy_score_id=score_id)
        )
        conn.execute(
            insert(pick_outcomes).values(
                strategy_score_id=score_id,
                symbol="SPY",
                strategy="bull_put_spread",
                expiry=expiry,
                entry_spot=590.0,
                terminal_spot=577.0,
                realized_pnl=-180.0,
                win=0,
                evaluated_at=NOW,
            )
        )
    entry = get_order(daemon_context.engine, entry_id)
    assert entry is not None
    daemon_context.exec_ibkr = MagicMock()
    assigned_stock = PortfolioPosition(
        account="DU123",
        symbol="SPY",
        sec_type="STK",
        expiry=None,
        strike=None,
        right=None,
        multiplier=1,
        position=100.0,
        avg_cost=580.0,
        market_price=577.0,
        market_value=57_700.0,
        unrealized_pnl=-300.0,
        realized_pnl=0.0,
        con_id=756733,
    )

    with patch(
        "optionsbot.ibkr.positions.PositionsClient",
        autospec=True,
    ) as positions_client:
        positions_client.return_value.get_portfolio = AsyncMock(
            return_value=[assigned_stock]
        )
        settled = await _settle_cleared_expirations(
            daemon_context,
            [entry],
            NOW,
        )

    assert settled == 1
    assert load_state(daemon_context.engine).killed
    assert "expected +100 SPY shares" in str(load_state(daemon_context.engine).reason)
    with daemon_context.engine.connect() as conn:
        row = conn.execute(
            select(position_settlements).where(
                position_settlements.c.entry_order_id == entry_id
            )
        ).one()
    assert row.kind == "expired_intrinsic"
    assert row.pnl == pytest.approx(-181.30)


async def test_exit_quote_set_is_serialized_on_daemon_ibkr_lock(
    daemon_context: DaemonContext,
) -> None:
    _filled_entry(daemon_context)
    _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    md = daemon_context._test_md  # type: ignore[attr-defined]

    def _locked_quote(
        symbol: str, expiry: str, strike: float, right: str
    ) -> OptionQuote:
        assert daemon_context.ibkr_lock.locked()
        mid = {(580.0, "P"): 1.40, (575.0, "P"): 0.30}[(strike, right)]
        return _quote(strike, right, mid)

    md.get_option_snapshot = AsyncMock(side_effect=_locked_quote)
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        summary = await run_exits_tick(daemon_context)

    assert summary.errors == 0
    assert md.get_option_snapshot.await_count == 2


def _queue_exit_request(
    context: DaemonContext,
    position_id: int,
    *,
    catalyst_type: str = "downgrade_upgrade",
) -> int:
    with context.engine.begin() as conn:
        pk = conn.execute(
            insert(exit_requests).values(
                position_id=position_id,
                requested_at=NOW,
                catalyst_type=catalyst_type,
                confidence=0.85,
                sources_json=["source A", "source B"],
                reason="corroborated adverse catalyst",
                status="requested",
            )
        ).inserted_primary_key
    assert pk is not None
    return int(pk[0])


async def test_take_profit_fires_closing_order(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    # Entry credit 1.20; structure now reopens at 0.50 -> kept 58% -> close.
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 1
    call = order_client.place_combo_limit.call_args
    # Flipped close: we PAY ~0.50/unit -> BUY-bag positive limit.
    assert call.kwargs["limit_price"] > 0
    with daemon_context.engine.connect() as conn:
        close = conn.execute(select(orders).where(orders.c.intent == "close")).one()
    assert close.closes_order_id == entry_id
    assert close.status == "submitted"
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("closing" in m.lower() for m in sent)


async def test_no_trigger_no_close(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with (
        patch(
            "optionsbot.daemon.exit_runner._exec_md",
            return_value=daemon_context._test_md,  # type: ignore[attr-defined]
        ),
        patch("optionsbot.daemon.exit_runner.log.info") as info,
    ):
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    decision = next(
        call for call in info.call_args_list if "exit decision entry_id=" in call.args[0]
    )
    assert decision.args[4] == "hold"
    assert decision.args[-1] == "ready"


@pytest.mark.parametrize("delivery_state", [True, None])
async def test_delayed_or_unknown_quote_never_triggers_take_profit(
    daemon_context: DaemonContext, delivery_state: bool | None
) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(
        daemon_context,
        {(580.0, "P"): 0.80, (575.0, "P"): 0.30},
        delayed=delivery_state,
    )

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()


@pytest.mark.parametrize("delivery_state", [True, None])
async def test_delayed_or_unknown_quote_never_triggers_soft_stop(
    daemon_context: DaemonContext, delivery_state: bool | None
) -> None:
    _filled_entry(daemon_context)
    daemon_context.settings.execution.exit_stop_enabled = True
    order_client = _wire(
        daemon_context,
        {(580.0, "P"): 4.00, (575.0, "P"): 0.30},
        delayed=delivery_state,
    )

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()


async def test_soft_stop_submission_emits_operational_event(
    daemon_context: DaemonContext,
) -> None:
    _filled_entry(daemon_context)
    daemon_context.settings.execution.exit_stop_enabled = True
    daemon_context.events = MagicMock()
    order_client = _wire(
        daemon_context,
        {(580.0, "P"): 4.00, (575.0, "P"): 0.30},
    )

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()
    daemon_context.events.emit.assert_called_once()
    assert daemon_context.events.emit.call_args.args[0] == "stop-hit"
    with daemon_context.engine.connect() as conn:
        close = conn.execute(
            select(orders.c.last_error)
            .where(orders.c.intent == "close")
            .where(orders.c.closes_order_id.is_not(None))
        ).one()
    assert close.last_error is not None
    assert close.last_error.startswith("exit trigger: soft stop")


async def test_request_exit_adverse_loser_submits_audited_close(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    # Entry credit 1.20; current debit 1.60 -> -0.40, beyond the 25% adverse gate.
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()
    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
    assert row.status == "submitted"
    assert row.close_order_id is not None
    assert "adverse" in row.decision_reason


@pytest.mark.parametrize("delivery_state", [True, None])
async def test_delayed_or_unknown_quote_cannot_corroborate_hermes_exit(
    daemon_context: DaemonContext, delivery_state: bool | None
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(
        daemon_context,
        {(580.0, "P"): 1.90, (575.0, "P"): 0.30},
        delayed=delivery_state,
    )

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
    assert row.status == "refused"
    assert "no live quote" in row.decision_reason


@pytest.mark.parametrize(
    ("confidence", "sources", "reason"),
    [
        (float("inf"), ["source A", "source B"], "material adverse move"),
        (0.90, "ab", "material adverse move"),
        (0.90, ["same", "same"], "material adverse move"),
        (0.90, ["", "source B"], "material adverse move"),
        (0.90, ["Reuters", "reuters"], "material adverse move"),
        (0.90, ["source A", "source B"], "   "),
    ],
)
async def test_persisted_exit_request_evidence_must_be_finite_and_well_formed(
    daemon_context: DaemonContext,
    confidence: float,
    sources: object,
    reason: str,
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(exit_requests)
            .where(exit_requests.c.id == request_id)
            .values(confidence=confidence, sources_json=sources, reason=reason)
        )
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
    assert row.status == "refused"
    assert "evidence" in row.decision_reason


async def test_request_exit_is_bound_before_broker_placement(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})

    async def _place(*args: object, **kwargs: object) -> PlacedOrder:
        with daemon_context.engine.connect() as conn:
            request = conn.execute(
                select(exit_requests).where(exit_requests.c.id == request_id)
            ).one()
            close = conn.execute(select(orders).where(orders.c.id == request.close_order_id)).one()
        assert request.status == "requested"
        assert request.close_order_id is not None
        assert close.status == "submitting"
        return PlacedOrder(
            ib_order_id=99,
            order_ref="obot-bound-before-place",
            action="BUY",
            limit_price=1.60,
            quantity=1,
            leg_contracts=((580001, 100, "USD"), (575001, 100, "USD")),
        )

    order_client.place_combo_limit.side_effect = _place
    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()


async def test_request_exit_rechecks_loss_cap_immediately_before_placement(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})
    allowed = HermesLossCapDecision(True, True, 0.0, 200.0, "within cap")
    denied = HermesLossCapDecision(False, True, -250.0, 200.0, "Hermes loss cap breached")
    enforce = AsyncMock(side_effect=[allowed, denied])

    with (
        patch(
            "optionsbot.daemon.exit_runner._exec_md",
            return_value=daemon_context._test_md,  # type: ignore[attr-defined]
        ),
        patch("optionsbot.daemon.exit_runner._enforce_hermes_loss_cap", enforce),
    ):
        summary = await run_exits_tick(daemon_context)

    assert enforce.await_count == 2
    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        request = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
        close = conn.execute(select(orders).where(orders.c.id == request.close_order_id)).one()
    assert request.status == "refused"
    assert close.status == "skipped"


async def test_kill_tripped_by_final_hermes_gate_does_not_stop_deterministic_close(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context, expiry=NEAR)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})
    allowed = HermesLossCapDecision(True, True, 0.0, 200.0, "within cap")
    denied = HermesLossCapDecision(False, True, -250.0, 200.0, "Hermes loss cap breached")
    calls = 0

    async def _enforce(*args: object, **kwargs: object) -> HermesLossCapDecision:
        nonlocal calls
        calls += 1
        if calls == 2:
            trip_kill(daemon_context.engine, denied.reason, now=NOW)
            return denied
        return allowed

    with (
        patch(
            "optionsbot.daemon.exit_runner._exec_md",
            return_value=daemon_context._test_md,  # type: ignore[attr-defined]
        ),
        patch("optionsbot.daemon.exit_runner._enforce_hermes_loss_cap", _enforce),
    ):
        summary = await run_exits_tick(daemon_context)

    assert calls == 2
    assert load_state(daemon_context.engine).killed is True
    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()
    with daemon_context.engine.connect() as conn:
        request = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
        closes = (
            conn.execute(select(orders.c.status).where(orders.c.closes_order_id == entry_id))
            .scalars()
            .all()
        )
    assert request.status == "refused"
    assert closes == ["skipped", "submitted"]


async def test_request_exit_losing_eligibility_after_binding_is_not_placed(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})
    allowed = HermesLossCapDecision(True, True, 0.0, 200.0, "within cap")
    calls = 0

    async def _enforce(*args: object, **kwargs: object) -> HermesLossCapDecision:
        nonlocal calls
        calls += 1
        if calls == 2:
            with daemon_context.engine.begin() as conn:
                conn.execute(
                    update(exit_requests)
                    .where(exit_requests.c.id == request_id)
                    .values(status="refused", decision_reason="concurrent refusal")
                )
        return allowed

    with (
        patch(
            "optionsbot.daemon.exit_runner._exec_md",
            return_value=daemon_context._test_md,  # type: ignore[attr-defined]
        ),
        patch("optionsbot.daemon.exit_runner._enforce_hermes_loss_cap", _enforce),
    ):
        summary = await run_exits_tick(daemon_context)

    assert calls == 2
    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        request = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
        close = conn.execute(select(orders).where(orders.c.id == request.close_order_id)).one()
    assert request.status == "refused"
    assert close.status == "skipped"


async def test_request_exit_completion_rowcount_failure_halts_after_placement(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})

    async def _place(*args: object, **kwargs: object) -> PlacedOrder:
        with daemon_context.engine.begin() as conn:
            conn.execute(
                update(exit_requests)
                .where(exit_requests.c.id == request_id)
                .values(status="refused", decision_reason="concurrent refusal")
            )
        return PlacedOrder(
            ib_order_id=99,
            order_ref="obot-completion-race",
            action="BUY",
            limit_price=1.60,
            quantity=1,
            leg_contracts=((580001, 100, "USD"), (575001, 100, "USD")),
        )

    order_client.place_combo_limit.side_effect = _place
    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()
    assert load_state(daemon_context.engine).killed is True
    with daemon_context.engine.connect() as conn:
        request = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
    assert request.status == "refused"
    assert request.close_order_id is not None


async def test_request_exit_completion_exception_halts_after_placement(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    _queue_exit_request(daemon_context, entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})

    with (
        patch(
            "optionsbot.daemon.exit_runner._exec_md",
            return_value=daemon_context._test_md,  # type: ignore[attr-defined]
        ),
        patch(
            "optionsbot.daemon.exit_runner._finish_bound_exit_request",
            side_effect=RuntimeError("simulated completion write failure"),
        ),
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert state.reason is not None and "completion" in state.reason


async def test_close_ledger_failure_after_broker_placement_halts(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon import exit_runner as er

    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    real_transition = er.transition

    def _fail_submitted(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        new_status = args[2] if len(args) > 2 else kwargs.get("new_status")
        if new_status == "submitted":
            raise RuntimeError("simulated post-placement ledger race")
        return real_transition(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch(
            "optionsbot.daemon.exit_runner._exec_md",
            return_value=daemon_context._test_md,  # type: ignore[attr-defined]
        ),
        patch.object(er, "transition", _fail_submitted),
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert state.reason is not None and "ledger" in state.reason


async def test_close_placement_exception_halts_with_claim_preserved(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    order_client.place_combo_limit.side_effect = TimeoutError("broker acknowledgement lost")

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_awaited_once()
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert state.reason is not None and "unknown" in state.reason
    with daemon_context.engine.connect() as conn:
        close = conn.execute(select(orders).where(orders.c.closes_order_id == entry_id)).one()
    assert close.status == "submitting"


async def test_close_ack_without_qualified_contracts_halts(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(
        daemon_context,
        {(580.0, "P"): 0.80, (575.0, "P"): 0.30},
    )
    order_client.place_combo_limit.side_effect = None
    order_client.place_combo_limit.return_value = PlacedOrder(
        ib_order_id=100,
        order_ref="obot-empty-close-contracts",
        action="BUY",
        limit_price=0.50,
        quantity=1,
        leg_contracts=(),
    )
    order_client.cancel = AsyncMock()

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    assert load_state(daemon_context.engine).killed is True
    order_client.cancel.assert_awaited_once_with(100)
    with daemon_context.engine.connect() as conn:
        close = conn.execute(select(orders).where(orders.c.closes_order_id == entry_id)).one()
    assert close.status == "submitting"


async def test_request_exit_refuses_winning_headline_without_deterministic_exit(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(
        daemon_context,
        entry_id,
        catalyst_type="headline_news",
    )
    _capture_loss_baseline(daemon_context)
    # Current debit 0.80 is a winner but below the deterministic 50% TP threshold.
    order_client = _wire(daemon_context, {(580.0, "P"): 1.10, (575.0, "P"): 0.30})

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
    assert row.status == "refused"
    assert "winner" in row.decision_reason


async def test_cumulative_hermes_realized_loss_trips_kill_before_another_exit(
    daemon_context: DaemonContext,
) -> None:
    engine = daemon_context.engine
    closed_entry_id = _filled_entry(daemon_context)
    with engine.begin() as conn:
        close_pk = conn.execute(
            insert(orders).values(
                intent="close",
                closes_order_id=closed_entry_id,
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=[
                    {**leg, "side": "buy" if leg["side"] == "sell" else "sell"}
                    for leg in _legs(FAR)
                ],
                quantity=1,
                status="filled",
                staged_ts=NOW,
                submitted_ts=NOW,
                terminal_ts=NOW,
                order_ref="hermes-loss-close",
                ib_order_id=77,
                reprice_count=0,
            )
        ).inserted_primary_key
    assert close_pk is not None
    close_id = int(close_pk[0])
    # Entry +$120, close -$370 => Hermes-driven realized P&L -$250.
    record_fill(
        engine,
        close_id,
        exec_id="hermes-loss-a",
        side="BUY",
        price=3.80,
        qty=1,
        ts=NOW,
        leg_con_id=580001,
    )
    record_fill(
        engine,
        close_id,
        exec_id="hermes-loss-b",
        side="SELL",
        price=0.10,
        qty=1,
        ts=NOW,
        leg_con_id=575001,
    )
    set_fill_commission(engine, "hermes-loss-a", 0.65)
    set_fill_commission(engine, "hermes-loss-b", 0.65)
    with engine.begin() as conn:
        conn.execute(
            insert(exit_requests).values(
                position_id=closed_entry_id,
                requested_at=NOW,
                catalyst_type="downgrade_upgrade",
                confidence=0.90,
                sources_json=["source A", "source B"],
                reason="prior Hermes close",
                status="submitted",
                decision_reason="corroborated adverse move",
                processed_at=NOW,
                close_order_id=close_id,
            )
        )

    _filled_entry(daemon_context)  # another open position remains at risk
    _capture_loss_baseline(daemon_context, 10_000.0)
    daemon_context.settings.execution.max_daily_loss_pct = 0.02
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    state = load_state(engine)
    assert state.killed is True
    assert state.reason is not None and "Hermes" in state.reason


async def test_cumulative_hermes_loss_trips_kill_after_last_position_closes(
    daemon_context: DaemonContext,
) -> None:
    engine = daemon_context.engine
    closed_entry_id = _filled_entry(daemon_context)
    with engine.begin() as conn:
        close_pk = conn.execute(
            insert(orders).values(
                intent="close",
                closes_order_id=closed_entry_id,
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=[
                    {**leg, "side": "buy" if leg["side"] == "sell" else "sell"}
                    for leg in _legs(FAR)
                ],
                quantity=1,
                status="filled",
                staged_ts=NOW,
                submitted_ts=NOW,
                terminal_ts=NOW,
                order_ref="hermes-final-loss-close",
                ib_order_id=78,
                reprice_count=0,
            )
        ).inserted_primary_key
    assert close_pk is not None
    close_id = int(close_pk[0])
    record_fill(
        engine,
        close_id,
        exec_id="hermes-final-loss-a",
        side="BUY",
        price=3.80,
        qty=1,
        ts=NOW,
        leg_con_id=580001,
    )
    record_fill(
        engine,
        close_id,
        exec_id="hermes-final-loss-b",
        side="SELL",
        price=0.10,
        qty=1,
        ts=NOW,
        leg_con_id=575001,
    )
    set_fill_commission(engine, "hermes-final-loss-a", 0.65)
    set_fill_commission(engine, "hermes-final-loss-b", 0.65)
    with engine.begin() as conn:
        conn.execute(
            insert(exit_requests).values(
                position_id=closed_entry_id,
                requested_at=NOW,
                catalyst_type="downgrade_upgrade",
                confidence=0.90,
                sources_json=["source A", "source B"],
                reason="Hermes closed the final position",
                # Crash after the broker close filled but before the request's
                # terminal status update; the bound close still owns attribution.
                status="requested",
                decision_reason="corroborated adverse move",
                processed_at=NOW,
                close_order_id=close_id,
            )
        )

    _capture_loss_baseline(daemon_context, 10_000.0)
    daemon_context.settings.execution.max_daily_loss_pct = 0.02
    daemon_context.order_client = MagicMock()

    summary = await run_exits_tick(daemon_context)

    assert summary.positions == 0
    state = load_state(engine)
    assert state.killed is True
    assert state.reason is not None and "Hermes" in state.reason


async def test_incomplete_hermes_realized_accounting_refuses_next_exit(
    daemon_context: DaemonContext,
) -> None:
    engine = daemon_context.engine
    closed_entry_id = _filled_entry(daemon_context)
    with engine.begin() as conn:
        close_id = int(
            conn.execute(
                insert(orders).values(
                    intent="close",
                    closes_order_id=closed_entry_id,
                    symbol="SPY",
                    strategy="bull_put_spread",
                    legs_json=[
                        {**leg, "side": "buy" if leg["side"] == "sell" else "sell"}
                        for leg in _legs(FAR)
                    ],
                    quantity=1,
                    status="filled",
                    staged_ts=NOW,
                    submitted_ts=NOW,
                    terminal_ts=NOW,
                    order_ref="hermes-incomplete-close",
                    ib_order_id=79,
                    reprice_count=0,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(exit_requests).values(
                position_id=closed_entry_id,
                requested_at=NOW,
                catalyst_type="downgrade_upgrade",
                confidence=0.90,
                sources_json=["source A", "source B"],
                reason="prior Hermes close with incomplete accounting",
                status="submitted",
                decision_reason="corroborated adverse move",
                processed_at=NOW,
                close_order_id=close_id,
            )
        )
    record_fill(
        engine,
        close_id,
        exec_id="hermes-incomplete-a",
        side="BUY",
        price=3.80,
        qty=1,
        ts=NOW,
    )
    set_fill_commission(engine, "hermes-incomplete-a", 0.65)

    next_entry_id = _filled_entry(daemon_context)
    request_id = _queue_exit_request(daemon_context, next_entry_id)
    _capture_loss_baseline(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.90, (575.0, "P"): 0.30})

    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    with engine.connect() as conn:
        request = conn.execute(select(exit_requests).where(exit_requests.c.id == request_id)).one()
    assert request.status == "refused"
    assert "accounting" in request.decision_reason


async def test_active_close_blocks_duplicate(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)
    assert first.closes_submitted == 1
    assert second.closes_submitted == 0
    assert order_client.place_combo_limit.await_count == 1


async def test_kill_switch_allows_deterministic_protective_exits(
    daemon_context: DaemonContext,
) -> None:
    _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    trip_kill(daemon_context.engine, "halt")
    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()


async def test_expiry_guard_forces_close(daemon_context: DaemonContext) -> None:
    _filled_entry(daemon_context, expiry=NEAR)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        summary = await run_exits_tick(daemon_context)
    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()


async def test_expired_contracts_are_never_submitted(
    daemon_context: DaemonContext,
) -> None:
    expiry = (nyse_session_date(NOW) - timedelta(days=1)).strftime("%Y%m%d")
    entry_id = _filled_entry(daemon_context, expiry=expiry)
    entry = get_order(daemon_context.engine, entry_id)
    assert entry is not None
    order_client = _wire(
        daemon_context,
        {(580.0, "P"): 1.40, (575.0, "P"): 0.30},
    )

    submitted = await _manage_entry(
        daemon_context,
        daemon_context._test_md,  # type: ignore[attr-defined]
        entry,
        NOW,
    )

    assert submitted == 0
    order_client.place_combo_limit.assert_not_awaited()


async def test_delayed_expiry_guard_is_quote_blind_and_does_not_walk(
    daemon_context: DaemonContext,
) -> None:
    _filled_entry(daemon_context, expiry=NEAR)
    order_client = _wire(
        daemon_context,
        {(580.0, "P"): 4.00, (575.0, "P"): 0.30},
        delayed=True,
    )
    daemon_context.settings.execution.walk_max_steps = 2

    with (
        patch(
            "optionsbot.daemon.exit_runner._exec_md",
            return_value=daemon_context._test_md,  # type: ignore[attr-defined]
        ),
        patch("optionsbot.execution.walk.run_price_walk", new_callable=AsyncMock) as walk,
    ):
        summary = await run_exits_tick(daemon_context)
        await asyncio.sleep(0)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()
    assert order_client.place_combo_limit.await_args.kwargs["limit_price"] == pytest.approx(1.20)
    walk.assert_not_awaited()


async def test_half_closed_position_hands_off_instead_of_reclosing(
    daemon_context: DaemonContext,
) -> None:
    # Opus IBK-129 critical: a close that PARTIALLY filled then died must
    # never be auto-restaged at full quantity (over-close = wrong-way risk).
    entry_id = _filled_entry(daemon_context)
    with daemon_context.engine.begin() as conn:
        pk = conn.execute(
            insert(orders).values(
                intent="close",
                closes_order_id=entry_id,
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=_legs(FAR),
                quantity=1,
                status="abandoned",
                staged_ts=NOW,
                submitted_ts=NOW,
                terminal_ts=NOW,
                reprice_count=0,
            )
        ).inserted_primary_key
        assert pk is not None
        close_id = int(pk[0])
        conn.execute(
            update(orders).where(orders.c.id == close_id).values(order_ref=f"obot-{close_id}")
        )
    record_fill(
        daemon_context.engine, close_id, exec_id="half1", side="BUY", price=0.80, qty=1, ts=NOW
    )

    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)
    assert first.closes_submitted == 0
    assert second.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    warns = [
        c.args[0]
        for c in daemon_context.telegram.send_message.await_args_list
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
            symbol="SPY",
            expiry=FAR,
            strike=strike,
            right=right,  # type: ignore[arg-type]
            bid=round(mid - 0.05, 4),
            ask=round(mid + 0.05, 4),
            last=None,
            mid=mid,
            iv=None,
            delta=None,
            gamma=None,
            theta=None,
            vega=None,
            open_interest=None,
            volume=None,
            ts=stale_ts,
            delayed=False,
        )

    md.get_option_snapshot = AsyncMock(side_effect=_stale_quote)
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        first = await run_exits_tick(daemon_context)
        second = await run_exits_tick(daemon_context)

    assert first.closes_submitted == 0
    assert second.closes_submitted == 0
    order_client.place_combo_limit.assert_not_awaited()
    stale_alerts = [
        c.args[0]
        for c in daemon_context.telegram.send_message.await_args_list
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
        patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md),  # type: ignore[attr-defined]
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
            symbol="SPY",
            expiry=NEAR,
            strike=strike,
            right=right,  # type: ignore[arg-type]
            bid=round(mid - 0.05, 4),
            ask=round(mid + 0.05, 4),
            last=None,
            mid=mid,
            iv=None,
            delta=None,
            gamma=None,
            theta=None,
            vega=None,
            open_interest=None,
            volume=None,
            ts=stale_ts,
            delayed=False,
        )

    md.get_option_snapshot = AsyncMock(side_effect=_stale_quote)
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=md):
        summary = await run_exits_tick(daemon_context)

    assert summary.closes_submitted == 1
    order_client.place_combo_limit.assert_awaited_once()


def _half_closed_entry_with_abandoned_close(context: DaemonContext, *, expiry: str = FAR) -> int:
    """Create a filled open entry whose only close is terminal+partial-filled
    (abandoned with one fill) — the condition that triggers the post-close
    naked-short sweep in ``run_exits_tick``."""
    entry_id = _filled_entry(context, expiry=expiry)
    engine = context.engine
    with engine.begin() as conn:
        pk = conn.execute(
            insert(orders).values(
                intent="close",
                closes_order_id=entry_id,
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=_legs(expiry),
                quantity=1,
                status="abandoned",
                staged_ts=NOW,
                submitted_ts=NOW,
                terminal_ts=NOW,
                reprice_count=0,
            )
        ).inserted_primary_key
        assert pk is not None
        close_id = int(pk[0])
        conn.execute(
            update(orders).where(orders.c.id == close_id).values(order_ref=f"obot-{close_id}")
        )
    # Partial fill on the abandoned close — this is what _half_closed() detects.
    record_fill(engine, close_id, exec_id="half1", side="BUY", price=0.80, qty=1, ts=NOW)
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
        con_id=1580,
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
        patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md),  # type: ignore[attr-defined]
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
        all_messages = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
        p1_alerts = [m for m in all_messages if "P1" in m or "🛑" in m]
        assert len(p1_alerts) == 1, f"expected 1 P1 alert on first tick, got: {p1_alerts}"
        # The entry is now tracked in the dedup set.
        assert entry_id in daemon_context.naked_leg_halted

        # --- Second tick: naked_leg_halted dedup suppresses re-alert ---
        # IBK-145: the sweep runs even with the kill tripped (it reads the broker,
        # places no orders), so we do NOT reset killed here. The only dedup barrier
        # under test is naked_leg_halted, which suppresses the repeat P1 alert.
        assert load_state(daemon_context.engine).killed is True
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


async def test_post_close_portfolio_read_failure_trips_kill(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    entry = get_order(daemon_context.engine, entry_id)
    assert entry is not None
    daemon_context.exec_ibkr = MagicMock()

    with patch(
        "optionsbot.ibkr.positions.PositionsClient",
        autospec=True,
    ) as mock_positions_client:
        mock_positions_client.return_value.get_portfolio = AsyncMock(
            side_effect=RuntimeError("portfolio unavailable")
        )
        clean = await assert_no_naked_short_after_close(daemon_context, entry)

    assert clean is False
    assert load_state(daemon_context.engine).killed is True
    daemon_context.telegram.send_message.assert_awaited()


@pytest.mark.parametrize("snapshot", [None, [object()]])
async def test_post_close_malformed_portfolio_snapshot_trips_kill(
    daemon_context: DaemonContext,
    snapshot: object,
) -> None:
    entry_id = _filled_entry(daemon_context)
    entry = get_order(daemon_context.engine, entry_id)
    assert entry is not None
    daemon_context.exec_ibkr = MagicMock()

    with patch(
        "optionsbot.ibkr.positions.PositionsClient",
        autospec=True,
    ) as mock_positions_client:
        mock_positions_client.return_value.get_portfolio = AsyncMock(return_value=snapshot)
        clean = await assert_no_naked_short_after_close(daemon_context, entry)

    assert clean is False
    assert load_state(daemon_context.engine).killed is True


# ---- /close: human-initiated force close (force_close_entry) ----


async def test_force_close_fires_without_a_trigger(daemon_context: DaemonContext) -> None:
    # The whole point of /close: it bypasses evaluate_exit. Use mids where the
    # take-profit rule would NOT fire (same mids as test_no_trigger_no_close) and
    # assert a close is placed anyway.
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
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
        conn.execute(update(orders).where(orders.c.id == entry_id).values(status="cancelled"))
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    msg = await force_close_entry(daemon_context, entry_id)
    order_client.place_combo_limit.assert_not_awaited()
    assert "filled open" in msg.lower()


async def test_force_close_blocks_when_already_closing(daemon_context: DaemonContext) -> None:
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await force_close_entry(daemon_context, entry_id)
        second = await force_close_entry(daemon_context, entry_id)
    assert order_client.place_combo_limit.await_count == 1
    assert str(entry_id) in first
    assert "already closing" in second.lower()


async def test_force_close_can_reduce_risk_while_killed(
    daemon_context: DaemonContext,
) -> None:
    entry_id = _filled_entry(daemon_context)
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    trip_kill(daemon_context.engine, "halt")
    with patch(
        "optionsbot.daemon.exit_runner._exec_md",
        return_value=daemon_context._test_md,  # type: ignore[attr-defined]
    ):
        msg = await force_close_entry(daemon_context, entry_id)
    order_client.place_combo_limit.assert_awaited_once()
    assert "close requested" in msg.lower()


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
            symbol="SPY",
            expiry=FAR,
            strike=strike,
            right=right,  # type: ignore[arg-type]
            bid=round(mid - 0.05, 4),
            ask=round(mid + 0.05, 4),
            last=None,
            mid=mid,
            iv=None,
            delta=None,
            gamma=None,
            theta=None,
            vega=None,
            open_interest=None,
            volume=None,
            ts=stale_ts,
            delayed=False,
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
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
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
    with patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md):  # type: ignore[attr-defined]
        first = await dispatch(daemon_context, f"/close {entry_id}")
        second = await dispatch(daemon_context, f"/close {entry_id}")
    assert order_client.place_combo_limit.await_count == 1  # exactly one close placed
    assert str(entry_id) in first[0].text
    assert "already closing" in second[0].text.lower()


# ---- IBK-142: exit-tick safety guards run outside market hours ----


async def test_naked_short_sweep_runs_when_market_closed(
    daemon_context: DaemonContext,
) -> None:
    # IBK-142: the post-close naked-short P1 sweep is detect-and-halt (no order
    # placement), so it must run OUTSIDE market hours — a partial close near the
    # bell can strand a short leg overnight/over a weekend, and the sweep must
    # trip the kill + alert before the next session, not stay dormant.
    entry_id = _half_closed_entry_with_abandoned_close(daemon_context)
    daemon_context.exec_ibkr = MagicMock()
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    residual = _residual_short_position(FAR)

    with (
        patch("optionsbot.daemon.exit_runner.is_market_open", return_value=False),
        patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md),  # type: ignore[attr-defined]
        patch("optionsbot.ibkr.positions.PositionsClient", autospec=True) as MockPC,
    ):
        mock_pc = MagicMock()
        mock_pc.get_portfolio = AsyncMock(return_value=[residual])
        MockPC.return_value = mock_pc
        summary = await run_exits_tick(daemon_context)

    # The sweep ran despite the closed market: kill tripped + exactly one P1 alert.
    assert load_state(daemon_context.engine).killed is True
    p1 = [
        c.args[0]
        for c in daemon_context.telegram.send_message.await_args_list
        if "P1" in c.args[0] or "🛑" in c.args[0]
    ]
    assert len(p1) == 1
    assert entry_id in daemon_context.naked_leg_halted
    # Order placement stays market-gated: no close orders were placed.
    order_client.place_combo_limit.assert_not_awaited()
    assert summary.closes_submitted == 0


async def test_naked_short_sweep_runs_when_kill_switched(
    daemon_context: DaemonContext,
) -> None:
    # IBK-145: the detect-and-halt sweep must run even when the execution
    # interlock is tripped -- a stranded naked short is most dangerous exactly
    # when the account is halted. With can_execute() false (kill switch), the
    # sweep still reads the broker, keeps the kill tripped, and sends the P1 alert,
    # while order placement stays blocked.
    entry_id = _half_closed_entry_with_abandoned_close(daemon_context)
    daemon_context.exec_ibkr = MagicMock()
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    residual = _residual_short_position(FAR)
    trip_kill(daemon_context.engine, "operator halt")  # interlock tripped BEFORE the tick

    with (
        patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md),  # type: ignore[attr-defined]
        patch("optionsbot.ibkr.positions.PositionsClient", autospec=True) as MockPC,
    ):
        mock_pc = MagicMock()
        mock_pc.get_portfolio = AsyncMock(return_value=[residual])
        MockPC.return_value = mock_pc
        summary = await run_exits_tick(daemon_context)

    assert load_state(daemon_context.engine).killed is True
    p1 = [
        c.args[0]
        for c in daemon_context.telegram.send_message.await_args_list
        if "P1" in c.args[0] or "🛑" in c.args[0]
    ]
    assert len(p1) == 1  # sweep alerted despite the kill
    assert entry_id in daemon_context.naked_leg_halted
    order_client.place_combo_limit.assert_not_awaited()  # placement still interlocked
    assert summary.closes_submitted == 0
    assert summary.positions == 1  # real count even while killed


async def test_naked_short_sweep_runs_when_md_unavailable(
    daemon_context: DaemonContext,
) -> None:
    # IBK-145: the sweep reads broker positions, not market data, so it must run
    # even when the exec quote source is down (md is None) -- the md gate only
    # blocks order placement.
    entry_id = _half_closed_entry_with_abandoned_close(daemon_context)
    daemon_context.exec_ibkr = MagicMock()
    order_client = _wire(daemon_context, {(580.0, "P"): 0.80, (575.0, "P"): 0.30})
    residual = _residual_short_position(FAR)

    with (
        patch("optionsbot.daemon.exit_runner._exec_md", return_value=None),  # md down
        patch("optionsbot.ibkr.positions.PositionsClient", autospec=True) as MockPC,
    ):
        mock_pc = MagicMock()
        mock_pc.get_portfolio = AsyncMock(return_value=[residual])
        MockPC.return_value = mock_pc
        summary = await run_exits_tick(daemon_context)

    assert load_state(daemon_context.engine).killed is True
    p1 = [
        c.args[0]
        for c in daemon_context.telegram.send_message.await_args_list
        if "P1" in c.args[0] or "🛑" in c.args[0]
    ]
    assert len(p1) == 1
    assert entry_id in daemon_context.naked_leg_halted
    order_client.place_combo_limit.assert_not_awaited()
    assert summary.closes_submitted == 0


async def test_market_closed_skips_order_placement(
    daemon_context: DaemonContext,
) -> None:
    # IBK-142 complement: with the market closed, a position that WOULD hit the
    # expiry guard gets NO close placed (order placement stays gated), even though
    # the tick now runs past the old early-return.
    _filled_entry(daemon_context, expiry=NEAR)
    order_client = _wire(daemon_context, {(580.0, "P"): 1.40, (575.0, "P"): 0.30})
    with (
        patch("optionsbot.daemon.exit_runner.is_market_open", return_value=False),
        patch("optionsbot.daemon.exit_runner._exec_md", return_value=daemon_context._test_md),  # type: ignore[attr-defined]
    ):
        summary = await run_exits_tick(daemon_context)
    order_client.place_combo_limit.assert_not_awaited()
    assert summary.closes_submitted == 0
