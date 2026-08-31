"""Prospective managed-outcome capture stays causal, conservative, and inert."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError

import optionsbot.daemon.managed_capture as managed_capture_mod
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.managed_capture import (
    canonical_legs,
    opportunity_key,
    record_snapshot_bot_dispositions,
    register_snapshot_opportunities,
    run_managed_capture_tick,
)
from optionsbot.ibkr.types import OptionQuote
from optionsbot.managed_contract import DEFAULT_MANAGED_OUTCOME_POLICY_VERSION
from optionsbot.storage.schema import (
    managed_context_reviews,
    managed_model_evaluations,
    managed_models,
    managed_opportunities,
    managed_opportunity_marks,
    orders,
    snapshots,
    strategy_scores,
)

SESSION = "2026-05-27"
DETECTED = datetime(2026, 5, 27, 14, 0, tzinfo=UTC)


def _leg(
    *,
    strike: float = 500.0,
    side: str = "buy",
    right: str = "C",
) -> dict[str, Any]:
    return {
        "symbol": "SPY",
        "side": side,
        "sec_type": "OPT",
        "expiry": "20260527",
        "strike": strike,
        "right": right,
        "quantity": 1,
    }


def _seed_snapshot(
    context: DaemonContext,
    *,
    signal_id: str = "2026-05-27:SPY:bull:fvg_retest:formed:respected",
    strategies: list[tuple[str, list[dict[str, Any]]]] | None = None,
    suggestions: dict[str, dict[str, Any]] | None = None,
    detected: datetime = DETECTED,
) -> int:
    if strategies is None:
        strategies = [("long_call", [_leg()])]
    plan = {
        "schema_version": "managed_signal_plan_v1",
        "status": "entry_confirmed",
        "source": "trusted_daemon",
        "admission_enabled": True,
        "signal_id": signal_id,
        "session": SESSION,
        "direction": "bull",
        "generator": "opening_range_fvg",
        "setup_type": "fvg_retest",
        "option_expiry": "20260527",
        "stop_pct": 0.15,
        "target_r": 1.5,
        "target_pct": 0.225,
    }
    with context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=detected,
                    spot=500.0,
                    iv_rank=0.5,
                    hv20=0.2,
                    iv_hv_ratio=1.5,
                    expected_move=3.0,
                    regime_dir="bull",
                    regime_iv="neutral",
                    raw_json={"opening_range_fvg": plan, "relative_volume": 1.3},
                )
            ).inserted_primary_key[0]
        )
        for strategy, legs in strategies:
            suggestion = {
                "defined_risk": True,
                "credit_or_debit": -110.0,
                "max_loss": 110.0,
                "max_profit": None,
                "expected_value": None,
                "managed_target_hit_probability": None,
                "managed_signal_plan": plan,
            }
            if suggestions is not None:
                suggestion.update(suggestions.get(strategy, {}))
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy=strategy,
                    score=75.0,
                    rationale="shadow test",
                    legs_json=legs,
                    suggestion_json=suggestion,
                )
            )
    return snapshot_id


def _quote(
    now: datetime,
    *,
    bid: float,
    ask: float,
    strike: float = 500.0,
    delayed: bool = False,
) -> OptionQuote:
    return OptionQuote(
        symbol="SPY",
        expiry="20260527",
        strike=strike,
        right="C",
        bid=bid,
        ask=ask,
        last=None,
        mid=(bid + ask) / 2.0,
        iv=0.3,
        delta=0.5,
        gamma=0.02,
        theta=-0.1,
        vega=0.05,
        open_interest=1000,
        volume=100,
        ts=now,
        delayed=delayed,
    )


class _QuoteFeed:
    def __init__(self, quotes: dict[tuple[str, str, float, str], OptionQuote]) -> None:
        self.quotes = quotes
        self.calls: list[tuple[str, str, float, str]] = []

    async def get_option_snapshot(
        self, symbol: str, expiry: str, strike: float, right: str
    ) -> OptionQuote:
        spec = (symbol, expiry, strike, right)
        self.calls.append(spec)
        return self.quotes[spec]


class _ConcurrentQuoteFeed(_QuoteFeed):
    def __init__(
        self,
        quotes: dict[tuple[str, str, float, str], OptionQuote],
        *,
        delay: float = 0.01,
        slow_first: bool = False,
    ) -> None:
        super().__init__(quotes)
        self.delay = delay
        self.slow_first = slow_first
        self.active = 0
        self.max_active = 0

    async def get_option_snapshot(
        self, symbol: str, expiry: str, strike: float, right: str
    ) -> OptionQuote:
        spec = (symbol, expiry, strike, right)
        call_index = len(self.calls)
        self.calls.append(spec)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(1.0 if self.slow_first and call_index == 0 else self.delay)
            return self.quotes[spec]
        finally:
            self.active -= 1


def _feed(now: datetime, *, bid: float, ask: float, delayed: bool = False) -> _QuoteFeed:
    quote = _quote(now, bid=bid, ask=ask, delayed=delayed)
    return _QuoteFeed({("SPY", "20260527", 500.0, "C"): quote})


def _only_opportunity(context: DaemonContext) -> Any:
    with context.engine.connect() as conn:
        return conn.execute(select(managed_opportunities)).one()


def _record_candidate_dispositions(context: DaemonContext, snapshot_id: int) -> None:
    """Complete the scan gate before a test starts the shadow path."""

    with context.engine.connect() as conn:
        detected_at = conn.scalar(select(snapshots.c.ts).where(snapshots.c.id == snapshot_id))
        strategies = conn.scalars(
            select(strategy_scores.c.strategy).where(
                strategy_scores.c.snapshot_id == snapshot_id
            )
        ).all()
    assert isinstance(detected_at, datetime)
    recorded = record_snapshot_bot_dispositions(
        context.engine,
        snapshot_id,
        {strategy: ("candidate", "scan_admission_passed") for strategy in strategies},
        policy_version=context.settings.managed_learning.outcome_policy_version,
        decided_at=detected_at.replace(tzinfo=UTC),
        account_value_usd=100_000.0,
    )
    assert recorded == len(strategies)


def test_canonical_identity_is_order_independent_and_does_not_retarget(
    daemon_context: DaemonContext,
) -> None:
    first = [_leg(strike=505.0, side="sell"), _leg(strike=500.0)]
    assert canonical_legs(first) == canonical_legs(list(reversed(first)))
    key = opportunity_key(
        "signal",
        "bull_call_spread",
        policy_version=DEFAULT_MANAGED_OUTCOME_POLICY_VERSION,
    )
    assert key == opportunity_key(
        "signal",
        "bull_call_spread",
        policy_version=DEFAULT_MANAGED_OUTCOME_POLICY_VERSION,
    )
    assert key != opportunity_key(
        "signal",
        "bull_call_spread",
        policy_version="different-policy",
    )

    signal = "stable-signal"
    snapshot_one = _seed_snapshot(
        daemon_context,
        signal_id=signal,
        strategies=[("long_call", [_leg(strike=500.0)])],
    )
    assert (
        register_snapshot_opportunities(
            daemon_context.engine, daemon_context.settings, snapshot_one
        )
        == 1
    )
    snapshot_two = _seed_snapshot(
        daemon_context,
        signal_id=signal,
        strategies=[("long_call", [_leg(strike=501.0)])],
        detected=datetime(2026, 5, 27, 14, 2, tzinfo=UTC),
    )
    assert (
        register_snapshot_opportunities(
            daemon_context.engine, daemon_context.settings, snapshot_two
        )
        == 0
    )
    row = _only_opportunity(daemon_context)
    assert row.legs_json[0]["strike"] == 500.0
    assert row.strategy_score_id is not None
    assert row.admission_eligible == 1
    assert row.shadow_only == 0
    assert row.bot_action is None
    assert row.baseline_action == "hold"
    assert "managed_expected_value_unavailable" in row.baseline_reason
    assert row.policy_version == daemon_context.settings.managed_learning.outcome_policy_version
    assert (
        row.features_json["feature_schema_version"]
        == daemon_context.settings.managed_learning.feature_schema_version
    )


def test_registration_revalidates_policy_identity_after_runtime_setting_drift(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    daemon_context.settings.validation.managed_capture_quote_max_age_seconds = 60

    with pytest.raises(
        ValueError,
        match="does not identify the configured managed-capture semantics",
    ):
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )


def test_registration_keeps_both_structures_and_persists_capacity_failure(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.validation.managed_capture_max_active = 1
    snapshot_id = _seed_snapshot(
        daemon_context,
        strategies=[
            ("long_call", [_leg()]),
            (
                "bull_call_spread",
                [_leg(), _leg(strike=505.0, side="sell")],
            ),
        ],
    )
    assert (
        register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
        == 2
    )
    with daemon_context.engine.connect() as conn:
        rows = conn.execute(
            select(
                managed_opportunities.c.strategy,
                managed_opportunities.c.status,
                managed_opportunities.c.resolution_reason,
            ).order_by(managed_opportunities.c.id)
        ).fetchall()
    assert [(row.strategy, row.status) for row in rows] == [
        ("long_call", "pending_entry"),
        ("bull_call_spread", "unobservable"),
    ]
    assert rows[1].resolution_reason == "managed_capture_capacity_reached"


def test_capacity_reclaims_a_surplus_structure_for_an_independent_signal(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.validation.managed_capture_max_active = 2
    first_signal = "2026-05-27:SPY:bull:first"
    first_snapshot = _seed_snapshot(
        daemon_context,
        signal_id=first_signal,
        strategies=[
            ("long_call", [_leg(strike=500.0)]),
            ("bull_call_spread", [_leg(strike=501.0)]),
        ],
    )
    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            first_snapshot,
        )
        == 2
    )

    second_signal = "2026-05-27:SPY:bull:second"
    second_snapshot = _seed_snapshot(
        daemon_context,
        signal_id=second_signal,
        strategies=[("long_call", [_leg(strike=502.0)])],
        detected=datetime(2026, 5, 27, 14, 1, tzinfo=UTC),
    )
    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            second_snapshot,
        )
        == 1
    )

    with daemon_context.engine.connect() as conn:
        rows = conn.execute(
            select(
                managed_opportunities.c.signal_id,
                managed_opportunities.c.strategy,
                managed_opportunities.c.status,
                managed_opportunities.c.resolution_reason,
            ).order_by(managed_opportunities.c.id)
        ).fetchall()
    active = [row for row in rows if row.status in {"pending_entry", "active"}]
    assert [row.signal_id for row in active] == [first_signal, second_signal]
    assert rows[0].strategy == "long_call"
    assert rows[0].status == "pending_entry"
    assert rows[1].strategy == "bull_call_spread"
    assert rows[1].status == "unobservable"
    assert (
        rows[1].resolution_reason == "managed_capture_capacity_reallocated_for_independent_signal"
    )


@pytest.mark.asyncio
async def test_capacity_reallocation_censors_an_started_surplus_path(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.validation.managed_capture_max_active = 2
    first_signal = "2026-05-27:SPY:bull:started-first"
    first_snapshot = _seed_snapshot(
        daemon_context,
        signal_id=first_signal,
        strategies=[
            ("long_call", [_leg(strike=500.0)]),
            ("bull_call_spread", [_leg(strike=501.0)]),
        ],
    )
    register_snapshot_opportunities(
        daemon_context.engine,
        daemon_context.settings,
        first_snapshot,
    )
    _record_candidate_dispositions(daemon_context, first_snapshot)
    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_QuoteFeed(
            {
                ("SPY", "20260527", strike, "C"): _quote(
                    entry_at,
                    bid=1.0,
                    ask=1.1,
                    strike=strike,
                )
                for strike in (500.0, 501.0)
            }
        ),  # type: ignore[arg-type]
    )

    second_signal = "2026-05-27:SPY:bull:started-second"
    second_snapshot = _seed_snapshot(
        daemon_context,
        signal_id=second_signal,
        strategies=[("long_call", [_leg(strike=502.0)])],
        detected=datetime(2026, 5, 27, 14, 1, tzinfo=UTC),
    )
    register_snapshot_opportunities(
        daemon_context.engine,
        daemon_context.settings,
        second_snapshot,
    )

    with daemon_context.engine.connect() as conn:
        rows = conn.execute(
            select(managed_opportunities).order_by(managed_opportunities.c.id)
        ).fetchall()
    assert rows[0].status == "active"
    assert rows[1].status == "censored"
    assert rows[1].outcome == "censored"
    assert rows[1].training_eligible == 0
    assert (
        rows[1].resolution_reason == "managed_capture_capacity_reallocated_for_independent_signal"
    )
    assert rows[2].status == "pending_entry"


def test_registration_uses_each_rows_shadow_plan_identity_and_timeout(
    daemon_context: DaemonContext,
) -> None:
    def _plan(
        signal_id: str,
        generator: str,
        direction: str,
        expires_at: str,
        efficiency: float,
    ) -> dict[str, Any]:
        return {
            "schema_version": "managed_signal_plan_v1",
            "status": "shadow_confirmed",
            "source": "trusted_daemon",
            "authority": "shadow_research_only_no_order_or_halt_authority",
            "admission_enabled": False,
            "signal_id": signal_id,
            "session": SESSION,
            "direction": direction,
            "generator": generator,
            "setup_type": generator,
            "option_expiry": "20260527",
            "thesis_expires_at": expires_at,
            "stop_pct": 0.15,
            "target_r": 1.5,
            "target_pct": 0.225,
            "hypothesis": {
                "direction": direction,
                "generator": generator,
                "reference_price": 500.0,
                "invalidation_level": 502.0 if direction == "bear" else 498.0,
                "features": {
                    "momentum": {"directional_efficiency": efficiency},
                },
            },
        }

    call_plan = _plan(
        "shadow-opening-momentum",
        "opening_momentum_continuation",
        "bull",
        "2026-05-27T15:00:00+00:00",
        0.8,
    )
    put_plan = _plan(
        "shadow-failed-breakout",
        "failed_breakout_reversal",
        "bear",
        "2026-05-27T18:00:00+00:00",
        0.6,
    )
    snapshot_id = _seed_snapshot(
        daemon_context,
        strategies=[
            ("shadow_grid_v1:long_call_d50:a", [_leg()]),
            (
                "shadow_grid_v1:long_put_d50:b",
                [_leg(right="P")],
            ),
        ],
        suggestions={
            "shadow_grid_v1:long_call_d50:a": {
                "shadow_only": True,
                "managed_signal_plan": call_plan,
            },
            "shadow_grid_v1:long_put_d50:b": {
                "shadow_only": True,
                "managed_signal_plan": put_plan,
            },
        },
    )

    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )
        == 2
    )
    with daemon_context.engine.connect() as conn:
        rows = conn.execute(
            select(managed_opportunities).order_by(managed_opportunities.c.signal_id)
        ).fetchall()
    assert [row.signal_id for row in rows] == [
        "shadow-failed-breakout",
        "shadow-opening-momentum",
    ]
    assert [row.direction for row in rows] == ["bear", "bull"]
    assert [row.setup_type for row in rows] == [
        "failed_breakout_reversal",
        "opening_momentum_continuation",
    ]
    assert rows[0].timeout_at == datetime(2026, 5, 27, 18, 0)
    assert rows[1].timeout_at == datetime(2026, 5, 27, 15, 0)
    assert rows[0].baseline_action == "hold"
    assert rows[1].baseline_action == "hold"
    assert [row.admission_eligible for row in rows] == [0, 0]
    assert [row.shadow_only for row in rows] == [1, 1]
    assert [row.bot_action for row in rows] == ["hold", "hold"]
    assert all(row.bot_decided_at is not None for row in rows)
    assert (
        rows[0].features_json["suggestion"]["managed_signal_plan"]["hypothesis"]["features"][
            "momentum"
        ]["directional_efficiency"]
        == 0.6
    )
    assert (
        rows[1].features_json["suggestion"]["managed_signal_plan"]["hypothesis"]["features"][
            "momentum"
        ]["directional_efficiency"]
        == 0.8
    )


def test_bot_scan_disposition_is_complete_and_first_write_wins(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    decided_at = datetime(2026, 5, 27, 14, 0, 1, tzinfo=UTC)
    assert (
        record_snapshot_bot_dispositions(
            daemon_context.engine,
            snapshot_id,
            {"long_call": ("hold", "non_positive_edge(expected_value=unavailable)")},
            policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
            decided_at=decided_at,
        )
        == 1
    )
    assert (
        record_snapshot_bot_dispositions(
            daemon_context.engine,
            snapshot_id,
            {"long_call": ("candidate", "attempted_rewrite")},
            policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
            decided_at=datetime(2026, 5, 27, 14, 0, 2, tzinfo=UTC),
            account_value_usd=100_000.0,
        )
        == 0
    )

    row = _only_opportunity(daemon_context)
    assert row.bot_action == "hold"
    assert row.bot_reason == "non_positive_edge(expected_value=unavailable)"
    assert row.bot_decided_at == decided_at.replace(tzinfo=None)
    with pytest.raises(IntegrityError, match="managed opportunities are immutable"):
        with daemon_context.engine.begin() as conn:
            conn.execute(delete(managed_opportunities).where(managed_opportunities.c.id == row.id))


@pytest.mark.asyncio
async def test_shadow_grid_candidate_is_held_captured_and_feasibility_labeled(
    daemon_context: DaemonContext,
) -> None:
    strategy = f"shadow_grid_v1:bull_call_spread_d50_25:{'b' * 64}"
    legs = [_leg(), _leg(strike=501.0, side="sell")]
    snapshot_id = _seed_snapshot(
        daemon_context,
        strategies=[(strategy, legs)],
        suggestions={
            strategy: {
                "shadow_only": True,
                "shadow_reason": "shadow_structure_pending_promoted_base_model",
                "managed_marketable_entry_net": -0.90,
                "managed_marketable_basis_dollars": 90.0,
                "managed_commission_estimate": 2.8,
                "estimated_round_trip_cost": 22.8,
                "structure_kind": "debit_vertical",
                "structure_leg_count": 2,
                "structure_width": 1.0,
                "structure_net_delta": 0.25,
                "structure_target_scenario_pnl_dollars": 8.0,
                "thesis_entry_spot": 500.0,
                "thesis_invalidation_spot": 499.0,
                "thesis_target_spot": 501.5,
                "premium_target_feasible": False,
            }
        },
    )
    # Registration must use the frozen decision-time economics even if a hot
    # reload changes the current commission setting between scan and import.
    daemon_context.settings.execution.opening_range_commission_per_contract = 1.25
    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )
        == 1
    )
    pending = _only_opportunity(daemon_context)
    assert pending.strategy == strategy
    assert pending.baseline_action == "hold"
    assert pending.baseline_reason == "shadow_structure_pending_promoted_base_model"
    assert pending.status == "pending_entry"
    assert pending.commission_estimate == pytest.approx(2.8)
    assert pending.features_json["suggestion"]["premium_target_feasible"] == 0
    assert pending.features_json["suggestion"]["managed_marketable_basis_dollars"] == 90.0

    # The optimizer's feasibility flag is evidence, not a silent prefilter.
    # Capture obtains a real entry NBBO, then applies the versioned structural
    # feasibility rule and retains the terminal reason outside training.
    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    feed = _QuoteFeed(
        {
            ("SPY", "20260527", 500.0, "C"): _quote(entry_at, bid=1.00, ask=1.10),
            ("SPY", "20260527", 501.0, "C"): _quote(entry_at, bid=0.20, ask=0.30, strike=501.0),
        }
    )
    summary = await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=feed,  # type: ignore[arg-type]
    )
    assert summary.usable_marks == 1
    labeled = _only_opportunity(daemon_context)
    assert labeled.status == "unobservable"
    assert labeled.resolution_reason == "target_not_reachable_after_commissions"
    assert labeled.entry_net == pytest.approx(-0.90)
    assert labeled.training_eligible == 0


@pytest.mark.asyncio
async def test_marketable_nbbo_target_label_and_no_execution_writes(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    assert (
        register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
        == 1
    )
    _record_candidate_dispositions(daemon_context, snapshot_id)

    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    first = await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_feed(entry_at, bid=1.00, ask=1.10),  # type: ignore[arg-type]
    )
    assert first.usable_marks == 1
    active = _only_opportunity(daemon_context)
    assert active.status == "active"
    assert active.entry_net == pytest.approx(-1.10)  # long opened at the ask
    assert active.gross_pnl is None

    target_at = datetime(2026, 5, 27, 14, 0, 20, tzinfo=UTC)
    second = await run_managed_capture_tick(
        daemon_context,
        now=target_at,
        md=_feed(target_at, bid=1.40, ask=1.50),  # type: ignore[arg-type]
    )
    assert second.resolved == 1
    result = _only_opportunity(daemon_context)
    assert result.status == "resolved"
    assert result.outcome == "target"
    assert result.exit_net == pytest.approx(-1.40)  # long liquidated at the bid
    assert result.gross_pnl == pytest.approx(30.0)
    assert result.net_pnl == pytest.approx(28.6)
    assert result.training_eligible == 1
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(orders)) == 0
        stored_probability = conn.scalar(select(strategy_scores.c.suggestion_json))[
            "managed_target_hit_probability"
        ]
        assert stored_probability is None


@pytest.mark.asyncio
async def test_marketable_nbbo_stop_label(daemon_context: DaemonContext) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_feed(entry_at, bid=1.00, ask=1.10),  # type: ignore[arg-type]
    )
    stop_at = datetime(2026, 5, 27, 14, 0, 20, tzinfo=UTC)
    result = await run_managed_capture_tick(
        daemon_context,
        now=stop_at,
        md=_feed(stop_at, bid=0.80, ask=0.90),  # type: ignore[arg-type]
    )

    assert result.resolved == 1
    row = _only_opportunity(daemon_context)
    assert row.outcome == "stop"
    assert row.gross_pnl == pytest.approx(-30.0)
    assert row.resolution_reason == "first_observed_stop_boundary"
    assert row.training_eligible == 1


@pytest.mark.asyncio
async def test_delayed_mark_then_large_gap_censors_threshold_order(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_feed(entry_at, bid=1.00, ask=1.10),  # type: ignore[arg-type]
    )
    bad_at = datetime(2026, 5, 27, 14, 0, 20, tzinfo=UTC)
    bad = await run_managed_capture_tick(
        daemon_context,
        now=bad_at,
        md=_feed(bad_at, bid=1.20, ask=1.30, delayed=True),  # type: ignore[arg-type]
    )
    assert bad.unusable_marks == 1
    target_at = datetime(2026, 5, 27, 14, 1, 5, tzinfo=UTC)
    final = await run_managed_capture_tick(
        daemon_context,
        now=target_at,
        md=_feed(target_at, bid=1.40, ask=1.50),  # type: ignore[arg-type]
    )
    assert final.censored == 1
    row = _only_opportunity(daemon_context)
    assert row.status == "censored"
    assert row.outcome == "censored"
    assert row.training_eligible == 0
    assert row.resolution_reason == "ambiguous_gap_before_target_observation"
    with daemon_context.engine.connect() as conn:
        marks = conn.execute(
            select(
                managed_opportunity_marks.c.usable,
                managed_opportunity_marks.c.issue,
            ).order_by(managed_opportunity_marks.c.id)
        ).fetchall()
    assert [mark.usable for mark in marks] == [1, 0, 1]
    assert marks[1].issue == "delayed_or_unknown_quote"


@pytest.mark.asyncio
async def test_timeout_is_force_exit_policy_not_expiry_close(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    entry_at = datetime(2026, 5, 27, 19, 29, 35, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_feed(entry_at, bid=1.00, ask=1.10),  # type: ignore[arg-type]
    )
    timeout_at = datetime(2026, 5, 27, 19, 30, 5, tzinfo=UTC)
    result = await run_managed_capture_tick(
        daemon_context,
        now=timeout_at,
        md=_feed(timeout_at, bid=1.05, ask=1.15),  # type: ignore[arg-type]
    )
    assert result.resolved == 1
    row = _only_opportunity(daemon_context)
    assert row.outcome == "timeout"
    assert row.resolution_reason == "scheduled_zero_dte_force_exit"
    # The shadow entry itself happened after the immutable 90-minute cutoff,
    # so the outcome is retained but excluded from the promotion dataset.
    assert row.training_eligible == 0


@pytest.mark.asyncio
async def test_slow_first_contract_does_not_starve_other_signals(
    daemon_context: DaemonContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strikes = (500.0, 501.0, 502.0)
    for index, strike in enumerate(strikes):
        snapshot_id = _seed_snapshot(
            daemon_context,
            signal_id=f"2026-05-27:SPY:bull:signal-{index}",
            strategies=[("long_call", [_leg(strike=strike)])],
        )
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )
        _record_candidate_dispositions(daemon_context, snapshot_id)
    now = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    feed = _ConcurrentQuoteFeed(
        {
            ("SPY", "20260527", strike, "C"): _quote(
                now,
                bid=1.0,
                ask=1.1,
                strike=strike,
            )
            for strike in strikes
        },
        slow_first=True,
    )
    monkeypatch.setattr(
        managed_capture_mod,
        "_PER_QUOTE_TIMEOUT_SECONDS",
        0.05,
    )

    summary = await run_managed_capture_tick(
        daemon_context,
        now=now,
        md=feed,  # type: ignore[arg-type]
    )

    assert len(feed.calls) == 3
    assert feed.max_active == 3
    assert summary.quote_errors == 1
    assert summary.usable_marks == 2
    assert summary.unusable_marks == 1
    assert not daemon_context.ibkr_lock.locked()
    with daemon_context.engine.connect() as conn:
        statuses = (
            conn.execute(
                select(managed_opportunities.c.status).order_by(managed_opportunities.c.id)
            )
            .scalars()
            .all()
        )
    assert statuses.count("active") == 2
    assert statuses.count("pending_entry") == 1


@pytest.mark.asyncio
async def test_quote_fetch_respects_line_and_concurrency_caps(
    daemon_context: DaemonContext,
) -> None:
    strikes = tuple(500.0 + index for index in range(5))
    for index, strike in enumerate(strikes):
        snapshot_id = _seed_snapshot(
            daemon_context,
            signal_id=f"2026-05-27:SPY:bull:cap-{index}",
            strategies=[("long_call", [_leg(strike=strike)])],
        )
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )
        _record_candidate_dispositions(daemon_context, snapshot_id)
    daemon_context.settings.validation.managed_capture_max_unique_legs = 4
    daemon_context.settings.ibkr.max_market_data_lines = 2
    now = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    feed = _ConcurrentQuoteFeed(
        {
            ("SPY", "20260527", strike, "C"): _quote(
                now,
                bid=1.0,
                ask=1.1,
                strike=strike,
            )
            for strike in strikes
        }
    )

    summary = await run_managed_capture_tick(
        daemon_context,
        now=now,
        md=feed,  # type: ignore[arg-type]
    )

    assert len(feed.calls) == 2
    assert len(set(feed.calls)) == 2
    assert feed.max_active == 2
    assert summary.usable_marks == 2
    assert summary.unusable_marks == 3


@pytest.mark.asyncio
async def test_poll_rotation_selects_complete_signal_bundles(
    daemon_context: DaemonContext,
) -> None:
    spread_snapshot = _seed_snapshot(
        daemon_context,
        signal_id="2026-05-27:SPY:bull:spread",
        strategies=[
            (
                "bull_call_spread",
                [_leg(strike=500.0), _leg(strike=501.0, side="sell")],
            )
        ],
    )
    single_snapshot = _seed_snapshot(
        daemon_context,
        signal_id="2026-05-27:SPY:bull:single",
        strategies=[("long_call", [_leg(strike=502.0)])],
    )
    register_snapshot_opportunities(
        daemon_context.engine,
        daemon_context.settings,
        spread_snapshot,
    )
    _record_candidate_dispositions(daemon_context, spread_snapshot)
    register_snapshot_opportunities(
        daemon_context.engine,
        daemon_context.settings,
        single_snapshot,
    )
    _record_candidate_dispositions(daemon_context, single_snapshot)
    daemon_context.settings.validation.managed_capture_max_unique_legs = 2
    daemon_context.settings.ibkr.max_market_data_lines = 2
    first_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    if (
        int(first_at.timestamp())
        // daemon_context.settings.validation.managed_capture_interval_seconds
    ) % 2 == 0:
        first_at = first_at.replace(second=20)
    first_feed = _QuoteFeed(
        {
            ("SPY", "20260527", strike, "C"): _quote(
                first_at,
                bid=1.0 if strike == 500.0 else 0.5,
                ask=1.1 if strike == 500.0 else 0.6,
                strike=strike,
            )
            for strike in (500.0, 501.0, 502.0)
        }
    )

    first = await run_managed_capture_tick(
        daemon_context,
        now=first_at,
        md=first_feed,  # type: ignore[arg-type]
    )
    assert first_feed.calls == [("SPY", "20260527", 502.0, "C")]
    assert first.usable_marks == 1
    assert first.unusable_marks == 1

    second_at = first_at.replace(second=first_at.second + 15)
    second_feed = _QuoteFeed(
        {
            ("SPY", "20260527", strike, "C"): _quote(
                second_at,
                bid=1.0 if strike == 500.0 else 0.5,
                ask=1.1 if strike == 500.0 else 0.6,
                strike=strike,
            )
            for strike in (500.0, 501.0, 502.0)
        }
    )
    second = await run_managed_capture_tick(
        daemon_context,
        now=second_at,
        md=second_feed,  # type: ignore[arg-type]
    )
    assert second_feed.calls == [
        ("SPY", "20260527", 500.0, "C"),
        ("SPY", "20260527", 501.0, "C"),
    ]
    assert second.usable_marks == 1
    assert second.unusable_marks == 1


@pytest.mark.asyncio
async def test_busy_trading_lock_records_gap_without_requesting_data(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    daemon_context.settings.validation.managed_capture_lock_timeout_seconds = 0.01
    now = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    feed = _feed(now, bid=1.0, ask=1.1)
    await daemon_context.ibkr_lock.acquire()
    try:
        summary = await run_managed_capture_tick(
            daemon_context,
            now=now,
            md=feed,  # type: ignore[arg-type]
        )
    finally:
        daemon_context.ibkr_lock.release()
    assert summary.skipped_for_trading
    assert summary.unusable_marks == 1
    assert feed.calls == []
    with daemon_context.engine.connect() as conn:
        mark = conn.execute(select(managed_opportunity_marks)).one()
    assert mark.issue == "trading_market_data_lock_busy"


@pytest.mark.asyncio
async def test_same_poll_bucket_is_idempotent(daemon_context: DaemonContext) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    first_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    duplicate_at = datetime(2026, 5, 27, 14, 0, 10, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=first_at,
        md=_feed(first_at, bid=1.0, ask=1.1),  # type: ignore[arg-type]
    )
    duplicate = await run_managed_capture_tick(
        daemon_context,
        now=duplicate_at,
        md=_feed(duplicate_at, bid=1.4, ask=1.5),  # type: ignore[arg-type]
    )
    assert duplicate.usable_marks == 0
    row = _only_opportunity(daemon_context)
    assert row.status == "active"
    assert row.valid_marks == 1
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(managed_opportunity_marks)) == 1


@pytest.mark.asyncio
async def test_entry_mark_and_reducer_roll_back_then_retry_as_one_unit(
    daemon_context: DaemonContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    original_update = managed_capture_mod._update_reduced_opportunity

    def crash_after_mark(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated reducer crash")

    monkeypatch.setattr(
        managed_capture_mod,
        "_update_reduced_opportunity",
        crash_after_mark,
    )
    with pytest.raises(RuntimeError, match="simulated reducer crash"):
        await run_managed_capture_tick(
            daemon_context,
            now=entry_at,
            md=_feed(entry_at, bid=1.0, ask=1.1),  # type: ignore[arg-type]
        )

    rolled_back = _only_opportunity(daemon_context)
    assert rolled_back.status == "pending_entry"
    assert rolled_back.valid_marks == 0
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(managed_opportunity_marks)) == 0

    monkeypatch.setattr(
        managed_capture_mod,
        "_update_reduced_opportunity",
        original_update,
    )
    retried = await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_feed(entry_at, bid=1.0, ask=1.1),  # type: ignore[arg-type]
    )

    assert retried.usable_marks == 1
    active = _only_opportunity(daemon_context)
    assert active.status == "active"
    assert active.valid_marks == 1
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(managed_opportunity_marks)) == 1


@pytest.mark.asyncio
async def test_terminal_mark_and_label_roll_back_then_retry_as_one_unit(
    daemon_context: DaemonContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_feed(entry_at, bid=1.0, ask=1.1),  # type: ignore[arg-type]
    )
    target_at = datetime(2026, 5, 27, 14, 0, 20, tzinfo=UTC)
    original_update = managed_capture_mod._update_reduced_opportunity

    def crash_after_mark(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated terminal reducer crash")

    monkeypatch.setattr(
        managed_capture_mod,
        "_update_reduced_opportunity",
        crash_after_mark,
    )
    with pytest.raises(RuntimeError, match="simulated terminal reducer crash"):
        await run_managed_capture_tick(
            daemon_context,
            now=target_at,
            md=_feed(target_at, bid=1.4, ask=1.5),  # type: ignore[arg-type]
        )

    rolled_back = _only_opportunity(daemon_context)
    assert rolled_back.status == "active"
    assert rolled_back.outcome is None
    assert rolled_back.valid_marks == 1
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(managed_opportunity_marks)) == 1

    monkeypatch.setattr(
        managed_capture_mod,
        "_update_reduced_opportunity",
        original_update,
    )
    retried = await run_managed_capture_tick(
        daemon_context,
        now=target_at,
        md=_feed(target_at, bid=1.4, ask=1.5),  # type: ignore[arg-type]
    )

    assert retried.resolved == 1
    resolved = _only_opportunity(daemon_context)
    assert resolved.status == "resolved"
    assert resolved.outcome == "target"
    assert resolved.valid_marks == 2
    assert resolved.training_eligible == 1
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(managed_opportunity_marks)) == 2


@pytest.mark.asyncio
async def test_stale_active_row_cannot_append_a_mark_after_terminal_resolution(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    _record_candidate_dispositions(daemon_context, snapshot_id)
    entry_at = datetime(2026, 5, 27, 14, 0, 5, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=entry_at,
        md=_feed(entry_at, bid=1.0, ask=1.1),  # type: ignore[arg-type]
    )
    stale_active = _only_opportunity(daemon_context)
    target_at = datetime(2026, 5, 27, 14, 0, 20, tzinfo=UTC)
    await run_managed_capture_tick(
        daemon_context,
        now=target_at,
        md=_feed(target_at, bid=1.4, ask=1.5),  # type: ignore[arg-type]
    )
    terminal_before = _only_opportunity(daemon_context)
    later = datetime(2026, 5, 27, 14, 0, 35, tzinfo=UTC)

    outcome, inserted = managed_capture_mod._process_usable_mark(
        daemon_context.engine,
        stale_active,
        now=later,
        bucket=managed_capture_mod._poll_bucket(
            later,
            daemon_context.settings.validation.managed_capture_interval_seconds,
        ),
        combo=(-1.6, -1.5),
        min_ts=later,
        max_ts=later,
        legs_json=[],
        settings=daemon_context.settings,
    )

    assert outcome is None
    assert inserted is False
    terminal_after = _only_opportunity(daemon_context)
    assert terminal_after.status == terminal_before.status == "resolved"
    assert terminal_after.outcome == terminal_before.outcome == "target"
    assert terminal_after.resolved_at == terminal_before.resolved_at
    assert terminal_after.net_pnl == terminal_before.net_pnl
    assert terminal_after.valid_marks == terminal_before.valid_marks == 2
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(managed_opportunity_marks)) == 2


def test_stale_active_row_cannot_append_unusable_mark_after_terminal_resolution(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    opportunity_id = int(_only_opportunity(daemon_context).id)
    terminal_at = datetime(2026, 5, 27, 20, 1, tzinfo=UTC)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(managed_opportunities)
            .where(managed_opportunities.c.id == opportunity_id)
            .where(managed_opportunities.c.status == "pending_entry")
            .values(
                status="unobservable",
                resolution_reason="no_usable_entry_quote_before_session_close",
            )
        )

    inserted = managed_capture_mod._persist_mark(
        daemon_context.engine,
        opportunity_id=opportunity_id,
        bucket=managed_capture_mod._poll_bucket(
            terminal_at,
            daemon_context.settings.validation.managed_capture_interval_seconds,
        ),
        now=terminal_at,
        min_ts=None,
        max_ts=None,
        combo=None,
        liquidation_net=None,
        gross_pnl=None,
        net_pnl=None,
        usable=False,
        issue="stale_quote_failure",
        legs_json=[],
    )

    assert inserted is False
    terminal = _only_opportunity(daemon_context)
    assert terminal.status == "unobservable"
    assert terminal.resolution_reason == "no_usable_entry_quote_before_session_close"
    with daemon_context.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(managed_opportunity_marks)) == 0


def test_close_finalizer_counts_only_the_terminal_transition_it_wins(
    daemon_context: DaemonContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    opportunity_id = int(_only_opportunity(daemon_context).id)
    after_close = datetime(2026, 5, 27, 21, 0, tzinfo=UTC)
    original = managed_capture_mod._terminalize_after_close_row

    def terminalize_first(
        engine: Any,
        row: Any,
        values: Any,
    ) -> bool:
        with engine.begin() as conn:
            conn.execute(
                update(managed_opportunities)
                .where(managed_opportunities.c.id == opportunity_id)
                .where(managed_opportunities.c.status == "pending_entry")
                .values(
                    status="unobservable",
                    resolution_reason="competing_finalizer_won",
                )
            )
        return original(engine, row, values)

    monkeypatch.setattr(
        managed_capture_mod,
        "_terminalize_after_close_row",
        terminalize_first,
    )

    censored, unobservable = managed_capture_mod._finalize_after_close(
        daemon_context.engine,
        after_close,
        policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
    )

    assert (censored, unobservable) == (0, 0)
    terminal = _only_opportunity(daemon_context)
    assert terminal.status == "unobservable"
    assert terminal.resolution_reason == "competing_finalizer_won"


def test_context_reviews_are_one_per_critic_and_db_immutable(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(daemon_context.engine, daemon_context.settings, snapshot_id)
    opportunity_id = int(_only_opportunity(daemon_context).id)
    values = {
        "opportunity_id": opportunity_id,
        "received_at": DETECTED,
        "timing": "pretrade",
        "response_json": {"verdict": "watch"},
        "response_hash": "a" * 64,
        "context_probability": None,
        "event_conflict": 0,
        "anomaly_json": [],
        "evidence_json": {"sources": []},
        "model_version": "critic-v1",
        "prompt_version": "prompt-v1",
    }
    with daemon_context.engine.begin() as conn:
        review_id = int(
            conn.execute(insert(managed_context_reviews).values(**values)).inserted_primary_key[0]
        )
    with pytest.raises(IntegrityError, match="immutable"):
        with daemon_context.engine.begin() as conn:
            conn.execute(
                update(managed_context_reviews)
                .where(managed_context_reviews.c.id == review_id)
                .values(response_json={"verdict": "rewritten"})
            )
    with pytest.raises(IntegrityError):
        with daemon_context.engine.begin() as conn:
            conn.execute(
                insert(managed_context_reviews).values(
                    **{
                        **values,
                        "response_hash": "b" * 64,
                    }
                )
            )
    with pytest.raises(IntegrityError, match="immutable"):
        with daemon_context.engine.begin() as conn:
            conn.execute(
                delete(managed_context_reviews).where(managed_context_reviews.c.id == review_id)
            )


def test_model_artifacts_and_fold_metrics_are_immutable(
    daemon_context: DaemonContext,
) -> None:
    with daemon_context.engine.begin() as conn:
        model_id = int(
            conn.execute(
                insert(managed_models).values(
                    model_version="managed-v1",
                    artifact_hash="c" * 64,
                    feature_schema_version="managed_capture_features_v1",
                    outcome_policy_version="marketable_nbbo_15s_v1",
                    trained_from_session="2026-01-02",
                    trained_through_session="2026-06-30",
                    metrics_json={"brier": 0.2},
                    status="challenger",
                    created_at=DETECTED,
                )
            ).inserted_primary_key[0]
        )
        evaluation_id = int(
            conn.execute(
                insert(managed_model_evaluations).values(
                    model_id=model_id,
                    evaluation_kind="walk_forward",
                    fold_index=0,
                    train_from_session="2026-01-02",
                    train_through_session="2026-04-30",
                    test_from_session="2026-05-01",
                    test_through_session="2026-05-31",
                    metrics_json={"brier": 0.21},
                    created_at=DETECTED,
                )
            ).inserted_primary_key[0]
        )
    # Lifecycle status is intentionally mutable; the artifact is not.
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(managed_models).where(managed_models.c.id == model_id).values(status="rejected")
        )
    with pytest.raises(IntegrityError, match="immutable"):
        with daemon_context.engine.begin() as conn:
            conn.execute(
                update(managed_models)
                .where(managed_models.c.id == model_id)
                .values(metrics_json={"brier": 0.01})
            )
    with pytest.raises(IntegrityError, match="immutable"):
        with daemon_context.engine.begin() as conn:
            conn.execute(
                update(managed_model_evaluations)
                .where(managed_model_evaluations.c.id == evaluation_id)
                .values(metrics_json={"brier": 0.01})
            )
