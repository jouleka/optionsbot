"""Adversarial checks for immutable managed scan-admission identity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.managed_capture import (
    record_snapshot_bot_dispositions,
    register_snapshot_opportunities,
)
from optionsbot.storage.schema import managed_opportunities
from tests.unit.test_daemon.test_managed_capture import (
    DETECTED,
    SESSION,
    _leg,
    _seed_snapshot,
)


def _row(context: DaemonContext):
    with context.engine.connect() as conn:
        return conn.execute(managed_opportunities.select()).one()


def _shadow_confirmed_opening_range_plan() -> dict[str, object]:
    return {
        "schema_version": "managed_signal_plan_v1",
        "status": "shadow_confirmed",
        "source": "trusted_daemon",
        "authority": "shadow_research_only_no_order_or_halt_authority",
        # Deliberately forged. Status is one-way and must keep this shadow.
        "admission_enabled": True,
        "signal_id": "forged-shadow-opening-range",
        "session": SESSION,
        "direction": "bull",
        "generator": "opening_range_fvg",
        "setup_type": "fvg_retest",
        "option_expiry": SESSION.replace("-", ""),
        "thesis_expires_at": "2026-05-27T18:00:00+00:00",
        "stop_pct": 0.15,
        "target_r": 1.5,
        "target_pct": 0.225,
    }


def test_shadow_confirmed_opening_range_cannot_forge_admission(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(
        daemon_context,
        suggestions={
            "long_call": {
                "managed_signal_plan": _shadow_confirmed_opening_range_plan(),
            }
        },
    )

    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )
        == 1
    )
    row = _row(daemon_context)
    assert row.admission_eligible == 0
    assert row.shadow_only == 1
    assert row.bot_action == "hold"


def test_unknown_non_registry_strategy_is_never_captured_as_primary(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(
        daemon_context,
        strategies=[("forged_primary_strategy", [_leg()])],
    )

    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )
        == 0
    )
    with daemon_context.engine.connect() as conn:
        assert conn.execute(managed_opportunities.select()).fetchall() == []


def test_capture_baseline_hold_can_become_candidate_only_on_the_exact_score(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            snapshot_id,
        )
        == 1
    )
    before = _row(daemon_context)
    assert before.baseline_action == "hold"
    assert before.bot_action is None

    assert (
        record_snapshot_bot_dispositions(
            daemon_context.engine,
            snapshot_id,
            {"long_call": ("candidate", "promoted_model_scan_admission_passed")},
            policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
            decided_at=DETECTED + timedelta(seconds=1),
            account_value_usd=100_000.0,
        )
        == 1
    )
    assert _row(daemon_context).bot_action == "candidate"


def test_candidate_disposition_requires_decision_time_equity(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(
        daemon_context.engine,
        daemon_context.settings,
        snapshot_id,
    )

    with pytest.raises(ValueError, match="decision-time account value"):
        record_snapshot_bot_dispositions(
            daemon_context.engine,
            snapshot_id,
            {"long_call": ("candidate", "missing_equity")},
            policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
            decided_at=DETECTED + timedelta(seconds=1),
        )

    assert _row(daemon_context).bot_action is None


def test_repeated_scan_cannot_authorize_the_prior_frozen_score(
    daemon_context: DaemonContext,
) -> None:
    signal_id = "same-signal-different-score"
    first_snapshot = _seed_snapshot(daemon_context, signal_id=signal_id)
    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            first_snapshot,
        )
        == 1
    )
    assert (
        record_snapshot_bot_dispositions(
            daemon_context.engine,
            first_snapshot,
            {"long_call": ("hold", "first_scan_held")},
            policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
            decided_at=DETECTED + timedelta(seconds=1),
        )
        == 1
    )

    second_detected = datetime(2026, 5, 27, 14, 2, tzinfo=UTC)
    second_snapshot = _seed_snapshot(
        daemon_context,
        signal_id=signal_id,
        detected=second_detected,
    )
    assert (
        register_snapshot_opportunities(
            daemon_context.engine,
            daemon_context.settings,
            second_snapshot,
        )
        == 0
    )
    assert (
        record_snapshot_bot_dispositions(
            daemon_context.engine,
            second_snapshot,
            {"long_call": ("candidate", "later_scan_attempted_authorization")},
            policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
            decided_at=second_detected + timedelta(seconds=1),
            account_value_usd=100_000.0,
        )
        == 0
    )
    row = _row(daemon_context)
    assert row.bot_action == "hold"
    assert row.bot_reason == "first_scan_held"


def test_backdated_scan_disposition_is_not_recorded(
    daemon_context: DaemonContext,
) -> None:
    snapshot_id = _seed_snapshot(daemon_context)
    register_snapshot_opportunities(
        daemon_context.engine,
        daemon_context.settings,
        snapshot_id,
    )

    assert (
        record_snapshot_bot_dispositions(
            daemon_context.engine,
            snapshot_id,
            {"long_call": ("candidate", "backdated")},
            policy_version=daemon_context.settings.managed_learning.outcome_policy_version,
            decided_at=DETECTED - timedelta(seconds=1),
            account_value_usd=100_000.0,
        )
        == 0
    )
    assert _row(daemon_context).bot_action is None
