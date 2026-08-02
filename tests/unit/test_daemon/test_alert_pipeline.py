"""Tests for the alert pipeline state machine (IBK-67)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from sqlalchemy import insert, select

from optionsbot.daemon.alert_pipeline import (
    BACKOFF_MINUTES,
    MAX_RETRIES,
    _reconstruct_scored,
    dispatch_alert,
    enqueue_alert,
    sweep_retries,
)
from optionsbot.daemon.context import DaemonContext
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.storage.schema import alerts, snapshots, strategy_scores
from optionsbot.strategies import StrategySuggestion


def _scored(name: str = "iron_condor", score: float = 85.0) -> ScoredStrategy:
    sug = MagicMock()
    sug.legs = ()
    sug.credit_or_debit = 1.25
    sug.max_loss = 3.75
    sug.max_profit = 1.25
    sug.prob_profit = 0.68
    sug.suggested_quantity = 5
    sug.defined_risk = True
    sug.reward_risk = 1.5
    sug.expected_value = 50.0
    sug.risk_tier = "balanced"
    return ScoredStrategy(
        strategy_name=name,
        score=score,
        factors=FactorBreakdown(0.7, 0.6, 0.8, 0.9, 1.0, 0.5),
        suggestion=sug,
        rationale="...",
    )


def _frozen_scored() -> ScoredStrategy:
    return ScoredStrategy(
        strategy_name="iron_condor",
        score=85.0,
        factors=FactorBreakdown(0.7, 0.6, 0.8, 0.9, 1.0, 0.5),
        suggestion=StrategySuggestion(
            strategy_name="iron_condor",
            legs=(),
            credit_or_debit=125.0,
            max_loss=375.0,
            max_profit=125.0,
            prob_profit=0.68,
            suggested_quantity=5,
            defined_risk=True,
            rationale="scan economics",
            reward_risk=1 / 3,
            expected_value=50.0,
        ),
        rationale="...",
    )


def _seed_snapshot(daemon_context: DaemonContext, *, symbol: str = "SPY") -> int:
    with daemon_context.engine.begin() as conn:
        result = conn.execute(
            insert(snapshots).values(
                symbol=symbol,
                ts=datetime.now(UTC),
                spot=400.0,
                regime_dir="bull",
                regime_iv="high",
            )
        )
        return result.inserted_primary_key[0]


def _seed_score(daemon_context: DaemonContext, snapshot_id: int) -> int:
    with daemon_context.engine.begin() as conn:
        return int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="iron_condor",
                    score=85.0,
                    rationale="candidate",
                    legs_json=[],
                    suggestion_json={"defined_risk": True},
                )
            ).inserted_primary_key[0]
        )


# ---- enqueue_alert ---------------------------------------------------------


async def test_enqueue_alert_inserts_pending_row_and_dispatches(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    _seed_score(daemon_context, snap_id)
    with patch(
        "optionsbot.daemon.alert_pipeline.should_alert",
        return_value=True,
    ):
        await enqueue_alert(daemon_context, "SPY", _scored(), snap_id)

    with daemon_context.engine.connect() as conn:
        rows = conn.execute(select(alerts)).fetchall()
    assert len(rows) == 1
    assert rows[0].status == "sent"
    assert rows[0].telegram_msg_id == 12345  # from mock_telegram fixture


async def test_enqueue_alert_links_exact_strategy_score(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    with daemon_context.engine.begin() as conn:
        score_id = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snap_id,
                strategy="iron_condor",
                score=85.0,
                rationale="candidate",
                legs_json=[],
                suggestion_json={"defined_risk": True},
            )
        ).inserted_primary_key[0]

    with patch(
        "optionsbot.daemon.alert_pipeline.should_alert",
        return_value=True,
    ):
        await enqueue_alert(daemon_context, "SPY", _scored(), snap_id)

    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(alerts)).one()
    assert row.strategy_score_id == score_id


def test_reconstruct_alert_uses_linked_score_not_newer_snapshot(
    daemon_context: DaemonContext,
) -> None:
    old_snapshot_id = _seed_snapshot(daemon_context)
    with daemon_context.engine.begin() as conn:
        old_score_id = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=old_snapshot_id,
                strategy="iron_condor",
                score=85.0,
                rationale="original candidate",
                legs_json=[],
                suggestion_json={"defined_risk": True},
            )
        ).inserted_primary_key[0]
        alert_id = conn.execute(
            insert(alerts).values(
                strategy_score_id=old_score_id,
                ts=datetime.now(UTC),
                symbol="SPY",
                strategy="iron_condor",
                score=85.0,
                status="failed",
            )
        ).inserted_primary_key[0]

    newer_snapshot_id = _seed_snapshot(daemon_context)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(strategy_scores).values(
                snapshot_id=newer_snapshot_id,
                strategy="iron_condor",
                score=99.0,
                rationale="newer unrelated candidate",
                legs_json=[],
                suggestion_json={"defined_risk": True},
            )
        )

    reconstructed = _reconstruct_scored(daemon_context, int(alert_id), "iron_condor", 85.0)

    assert reconstructed is not None
    assert reconstructed.rationale == "original candidate"
    assert reconstructed.score == 85.0


async def test_enqueue_alert_skips_when_dedup_says_no(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    with patch(
        "optionsbot.daemon.alert_pipeline.should_alert",
        return_value=False,
    ):
        was_enqueued = await enqueue_alert(daemon_context, "SPY", _scored(), snap_id)

    # Dedup-skipped: caller must see False so scan_runs.alerts_fired
    # doesn't get padded with phantom rows.
    assert was_enqueued is False
    with daemon_context.engine.connect() as conn:
        rows = conn.execute(select(alerts)).fetchall()
    assert rows == []
    daemon_context.telegram.send_message.assert_not_awaited()


async def test_opening_range_signal_is_alerted_only_once_per_strategy(
    daemon_context: DaemonContext,
) -> None:
    plan = {
        "status": "entry_confirmed",
        "signal_id": "2026-07-31:SPY:bull:formed:respected",
    }
    with daemon_context.engine.begin() as conn:
        first_snapshot = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=datetime.now(UTC),
                    spot=400.0,
                    regime_dir="bull",
                    regime_iv="neutral",
                    raw_json={"opening_range_fvg": plan},
                )
            ).inserted_primary_key[0]
        )
        first_score = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=first_snapshot,
                    strategy="iron_condor",
                    score=80.0,
                    rationale="first observation",
                    legs_json=[],
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(alerts).values(
                strategy_score_id=first_score,
                ts=datetime.now(UTC),
                symbol="SPY",
                strategy="iron_condor",
                score=80.0,
                status="sent",
            )
        )
        second_snapshot = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=datetime.now(UTC),
                    spot=401.0,
                    regime_dir="bull",
                    regime_iv="neutral",
                    raw_json={"opening_range_fvg": plan},
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(strategy_scores).values(
                snapshot_id=second_snapshot,
                strategy="iron_condor",
                score=90.0,
                rationale="same setup observed again",
                legs_json=[],
                suggestion_json={},
            )
        )

    with patch("optionsbot.daemon.alert_pipeline.should_alert") as cooldown_gate:
        was_enqueued = await enqueue_alert(
            daemon_context,
            "SPY",
            _scored(score=90.0),
            second_snapshot,
        )

    assert was_enqueued is False
    cooldown_gate.assert_not_called()
    daemon_context.telegram.send_message.assert_not_awaited()


async def test_enqueue_alert_returns_true_when_actually_enqueued(
    daemon_context: DaemonContext,
) -> None:
    """When dedup allows the alert, enqueue_alert must return True."""
    snap_id = _seed_snapshot(daemon_context)
    _seed_score(daemon_context, snap_id)
    with patch(
        "optionsbot.daemon.alert_pipeline.should_alert",
        return_value=True,
    ):
        was_enqueued = await enqueue_alert(daemon_context, "SPY", _scored(), snap_id)
    assert was_enqueued is True


async def test_enqueue_alert_dispatches_fresh_economics_for_frozen_suggestion(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    _seed_score(daemon_context, snap_id)
    scored = _frozen_scored()
    evidence = {
        "economics": {
            "credit_or_debit": 90.0,
            "max_loss": 410.0,
            "max_profit": 90.0,
            "reward_risk": 90 / 410,
            "expected_value": -5.0,
        }
    }

    with (
        patch(
            "optionsbot.daemon.alert_pipeline.should_alert",
            return_value=True,
        ),
        patch(
            "optionsbot.daemon.candidate_evidence.capture_candidate_evidence",
            new=AsyncMock(return_value=evidence),
        ),
        patch(
            "optionsbot.daemon.alert_pipeline.dispatch_alert",
            new=AsyncMock(),
        ) as dispatch,
    ):
        await enqueue_alert(daemon_context, "SPY", scored, snap_id)

    dispatched = dispatch.await_args.args[3]
    assert dispatched is not scored
    assert dispatched.suggestion.credit_or_debit == 90.0
    assert dispatched.suggestion.max_loss == 410.0
    assert dispatched.suggestion.max_profit == 90.0
    assert dispatched.suggestion.reward_risk == 90 / 410
    assert dispatched.suggestion.expected_value == -5.0
    assert scored.suggestion.credit_or_debit == 125.0
    assert scored.suggestion.expected_value == 50.0


async def test_enqueue_alert_on_send_failure_marks_failed_with_backoff(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    _seed_score(daemon_context, snap_id)
    daemon_context.telegram.send_message = AsyncMock(
        side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=MagicMock())
    )
    with patch(
        "optionsbot.daemon.alert_pipeline.should_alert",
        return_value=True,
    ):
        await enqueue_alert(daemon_context, "SPY", _scored(), snap_id)

    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(alerts)).fetchone()
    assert row.status == "failed"
    assert row.retry_count == 1
    assert row.last_error is not None
    assert row.next_retry_ts is not None


# ---- dispatch_alert direct -------------------------------------------------


async def test_dispatch_alert_records_telegram_msg_id_on_success(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    with daemon_context.engine.begin() as conn:
        result = conn.execute(
            insert(alerts).values(
                ts=datetime.now(UTC),
                symbol="SPY",
                strategy="iron_condor",
                score=85.0,
                status="pending",
            )
        )
        alert_id = result.inserted_primary_key[0]
    daemon_context.telegram.send_message = AsyncMock(return_value=777)

    await dispatch_alert(daemon_context, alert_id, snap_id, _scored())

    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(alerts).where(alerts.c.id == alert_id)).fetchone()
    assert row.status == "sent"
    assert row.telegram_msg_id == 777
    assert row.sent_ts is not None


# ---- backoff schedule ------------------------------------------------------


def test_backoff_minutes_is_strictly_increasing() -> None:
    """1m -> 5m -> 15m -> 60m -> 240m or similar; reasonable scheme."""
    for a, b in zip(BACKOFF_MINUTES, BACKOFF_MINUTES[1:], strict=False):
        assert a < b
    assert len(BACKOFF_MINUTES) == MAX_RETRIES


# ---- sweep_retries ---------------------------------------------------------


async def test_sweep_retries_processes_failed_rows_past_next_retry_ts(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    score_id = _seed_score(daemon_context, snap_id)
    past = datetime.now(UTC) - timedelta(minutes=5)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(alerts).values(
                strategy_score_id=score_id,
                ts=datetime.now(UTC),
                symbol="SPY",
                strategy="iron_condor",
                score=85.0,
                status="failed",
                retry_count=1,
                next_retry_ts=past,
                last_error="prev fail",
            )
        )

    daemon_context.telegram.send_message = AsyncMock(return_value=99)
    with patch(
        "optionsbot.daemon.alert_pipeline._reconstruct_scored",
        return_value=_scored(),
    ):
        count = await sweep_retries(daemon_context)
    assert count == 1
    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(alerts)).fetchone()
    assert row.status == "sent"
    assert row.telegram_msg_id == 99
    # _mark_sent must scrub the previous failure trail; a sent row carrying
    # stale last_error or next_retry_ts is a state-machine bug.
    assert row.last_error is None
    assert row.next_retry_ts is None


async def test_sweep_retries_uses_linked_scores_original_snapshot(
    daemon_context: DaemonContext,
) -> None:
    old_snapshot_id = _seed_snapshot(daemon_context)
    past = datetime.now(UTC) - timedelta(minutes=5)
    with daemon_context.engine.begin() as conn:
        old_score_id = conn.execute(
            insert(strategy_scores).values(
                snapshot_id=old_snapshot_id,
                strategy="iron_condor",
                score=85.0,
                rationale="original candidate",
                legs_json=[],
                suggestion_json={"defined_risk": True},
            )
        ).inserted_primary_key[0]
        alert_id = conn.execute(
            insert(alerts).values(
                strategy_score_id=old_score_id,
                ts=datetime.now(UTC),
                symbol="SPY",
                strategy="iron_condor",
                score=85.0,
                status="failed",
                retry_count=1,
                next_retry_ts=past,
            )
        ).inserted_primary_key[0]

    newer_snapshot_id = _seed_snapshot(daemon_context)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(strategy_scores).values(
                snapshot_id=newer_snapshot_id,
                strategy="iron_condor",
                score=99.0,
                rationale="newer unrelated candidate",
                legs_json=[],
                suggestion_json={"defined_risk": True},
            )
        )

    dispatch = AsyncMock()
    with patch("optionsbot.daemon.alert_pipeline.dispatch_alert", new=dispatch):
        count = await sweep_retries(daemon_context)

    assert count == 1
    dispatch.assert_awaited_once()
    call = dispatch.await_args
    assert call is not None
    assert call.args[1] == alert_id
    assert call.args[2] == old_snapshot_id


async def test_sweep_retries_skips_row_when_reconstruct_returns_none(
    daemon_context: DaemonContext,
) -> None:
    """If the strategy_scores history is missing (e.g., snapshots were never
    persisted for this symbol), sweep_retries logs and skips the row rather
    than crashing. The row's status stays unchanged so a later tick can
    retry once the data lands."""
    past = datetime.now(UTC) - timedelta(minutes=5)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(alerts).values(
                ts=datetime.now(UTC),
                symbol="NEVERSCANNED",
                strategy="iron_condor",
                score=85.0,
                status="failed",
                retry_count=1,
                next_retry_ts=past,
                last_error="prev fail",
            )
        )

    # No snapshot rows exist for NEVERSCANNED, so _latest_snapshot_id_for_symbol
    # returns None inside _reconstruct_scored.
    count = await sweep_retries(daemon_context)
    assert count == 0
    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(alerts)).fetchone()
    # Row untouched -- status, retry_count, next_retry_ts all preserved.
    assert row.status == "failed"
    assert row.retry_count == 1
    daemon_context.telegram.send_message.assert_not_awaited()


async def test_sweep_retries_skips_rows_with_future_next_retry_ts(
    daemon_context: DaemonContext,
) -> None:
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(alerts).values(
                ts=datetime.now(UTC),
                symbol="SPY",
                strategy="iron_condor",
                score=85.0,
                status="failed",
                retry_count=1,
                next_retry_ts=datetime.now(UTC) + timedelta(minutes=5),
            )
        )
    count = await sweep_retries(daemon_context)
    assert count == 0


async def test_sweep_retries_drops_after_max_retries(
    daemon_context: DaemonContext,
) -> None:
    snap_id = _seed_snapshot(daemon_context)
    score_id = _seed_score(daemon_context, snap_id)
    past = datetime.now(UTC) - timedelta(minutes=5)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(alerts).values(
                strategy_score_id=score_id,
                ts=datetime.now(UTC),
                symbol="SPY",
                strategy="iron_condor",
                score=85.0,
                status="failed",
                retry_count=MAX_RETRIES,
                next_retry_ts=past,
                last_error="repeated fail",
            )
        )
    daemon_context.telegram.send_message = AsyncMock(
        side_effect=httpx.HTTPStatusError("nope", request=MagicMock(), response=MagicMock())
    )
    with patch(
        "optionsbot.daemon.alert_pipeline._reconstruct_scored",
        return_value=_scored(),
    ):
        await sweep_retries(daemon_context)
    with daemon_context.engine.connect() as conn:
        row = conn.execute(select(alerts)).fetchone()
    assert row.status == "dropped"


async def test_retry_preserves_undefined_risk_warning_via_suggestion_json(
    daemon_context: DaemonContext,
) -> None:
    """The Critical bug Opus 4.7 flagged: a retry must render the SAME
    alert as the first attempt -- in particular, an undefined-risk
    strategy (short_straddle, short_strangle) must still get the
    `⚠ UNDEFINED RISK` header on retry. This works only when scan_symbol
    persisted suggestion_json (defined_risk + financials) so the retry
    reconstructor can rebuild a faithful suggestion.
    """
    # Seed snapshot + a strategy_scores row for short_straddle with
    # suggestion_json indicating undefined risk.
    now = datetime.now(UTC)
    with daemon_context.engine.begin() as conn:
        snap_result = conn.execute(
            insert(snapshots).values(
                symbol="SPY",
                ts=now,
                spot=400.0,
                regime_dir="neutral",
                regime_iv="high",
            )
        )
        snap_id = snap_result.inserted_primary_key[0]
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snap_id,
                    strategy="short_straddle",
                    score=85.0,
                    rationale="High IV + neutral",
                    legs_json=[
                        {
                            "symbol": "SPY",
                            "side": "sell",
                            "sec_type": "OPT",
                            "strike": 400.0,
                            "right": "C",
                            "expiry": "20260711",
                            "quantity": 1,
                        }
                    ],
                    suggestion_json={
                        "defined_risk": False,
                        "credit_or_debit": 8.50,
                        "max_loss": None,
                        "max_profit": 8.50,
                        "prob_profit": 0.45,
                        "suggested_quantity": 1,
                    },
                )
            ).inserted_primary_key[0]
        )
        # Seed a failed alert so sweep_retries finds it.
        past = now - timedelta(minutes=5)
        conn.execute(
            insert(alerts).values(
                strategy_score_id=score_id,
                ts=now,
                symbol="SPY",
                strategy="short_straddle",
                score=85.0,
                status="failed",
                retry_count=1,
                next_retry_ts=past,
                last_error="prev fail",
            )
        )

    sent_text: list[str] = []

    async def _capture(text: str) -> int:
        sent_text.append(text)
        return 99

    daemon_context.telegram.send_message = AsyncMock(side_effect=_capture)

    # Use the REAL _reconstruct_scored (no patch) so the suggestion_json
    # round-trip is exercised end-to-end.
    count = await sweep_retries(daemon_context)

    assert count == 1
    assert len(sent_text) == 1
    body = sent_text[0]
    # The retry must include the warning the first attempt would have shown.
    assert "UNDEFINED RISK" in body
    # The financial figures from suggestion_json must also surface.
    assert "8.50" in body
    assert "45%" in body  # prob_profit 0.45 -> "45%"
