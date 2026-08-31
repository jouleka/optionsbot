"""Hermes shadow context-critic MCP and audit-ledger tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, insert, select

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.control_intents import (
    consume_control_intents,
    consume_control_intents_async,
)
from optionsbot.execution.state import load_state
from optionsbot.hermes_context import CONTEXT_CONTRACT_VERSION
from optionsbot.hermes_context_metrics import context_shadow_report
from optionsbot.mcp_server.intent_queue import (
    control_intents,
    create_intent_engine,
    enqueue_intent,
)
from optionsbot.mcp_server.restricted_context import RestrictedServerContext
from optionsbot.mcp_server.tools import context_critic
from optionsbot.storage.schema import (
    managed_context_reviews,
    managed_opportunities,
    orders,
    snapshots,
    strategy_scores,
)
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


def _seed_opportunity(
    engine: Engine,
    *,
    key: str = "op-1",
    signal_id: str = "2026-08-28:SPY:bull:fvg-retest:1",
    detected_at: datetime | None = None,
    baseline_action: str = "candidate",
    bot_action: str | None = "candidate",
    status: str = "pending_entry",
    outcome: str | None = None,
    net_pnl: float | None = None,
    resolved_at: datetime | None = None,
    training_eligible: int = 0,
    entry_ts: datetime | None = None,
    entry_cutoff_at: datetime | None = None,
) -> int:
    now = detected_at or datetime.now(UTC)
    with engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=now,
                    spot=650.0,
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        legs = [
            {
                "symbol": "SPY",
                "side": "buy",
                "sec_type": "OPT",
                "expiry": now.date().strftime("%Y%m%d"),
                "strike": 650.0,
                "right": "C",
                "quantity": 1,
            }
        ]
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="long_call",
                    score=75.0,
                    rationale="context fixture",
                    legs_json=legs,
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        pk = conn.execute(
            insert(managed_opportunities).values(
                opportunity_key=key,
                signal_id=signal_id,
                session=now.date().isoformat(),
                symbol="SPY",
                direction="bull",
                setup_type="fvg_retest",
                strategy="long_call",
                strategy_score_id=score_id,
                structure_hash=f"hash-{key}",
                legs_json=legs,
                features_json={"rvol": 1.2},
                policy_version="managed-path-v1",
                decision_batch_id=f"context-batch:{snapshot_id}",
                decision_score=75.0,
                decision_defined_risk=1,
                decision_max_loss=100.0,
                created_at=now,
                detected_at=now,
                baseline_action=baseline_action,
                baseline_reason="capture economics baseline",
                admission_eligible=1,
                shadow_only=0,
                bot_action=bot_action,
                bot_reason=("scan admission decision" if bot_action else None),
                bot_decided_at=(now if bot_action else None),
                decision_account_value_available=(1 if bot_action else None),
                decision_account_value_usd=(50_000.0 if bot_action else None),
                session_close_at=now + timedelta(hours=5),
                entry_cutoff_at=entry_cutoff_at or now + timedelta(hours=4),
                timeout_at=now + timedelta(hours=4, minutes=15),
                entry_ts=entry_ts,
                entry_net=(-1.0 if status == "resolved" else None),
                basis_dollars=(100.0 if status == "resolved" else None),
                stop_pct=0.15,
                target_pct=0.225,
                commission_estimate=1.30,
                status=status,
                outcome=outcome,
                resolved_at=resolved_at,
                gross_pnl=(
                    net_pnl + 1.30 if status == "resolved" and net_pnl is not None else None
                ),
                net_pnl=net_pnl,
                training_eligible=training_eligible,
            )
        ).inserted_primary_key
    assert pk is not None
    return int(pk[0])


def _context(
    engine: Engine,
    intent_path: Path,
) -> RestrictedServerContext:
    return RestrictedServerContext(
        engine=engine,
        intent_engine=create_intent_engine(intent_path),
        max_pick_age_minutes=20,
    )


def _submit(
    tools: dict[str, Any],
    context: RestrictedServerContext,
    opportunity_id: int,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        tools["submit_context_review"](
            CONTEXT_CONTRACT_VERSION,
            opportunity_id,
            "2026-08-28:SPY:bull:fvg-retest:1",
            0.62,
            False,
            ["market_regime_conflict"],
            ["finnhub:quote:SPY:1724851800"],
            "hermes-shadow-context-1.0.0",
            "optionsbot-context-v1",
            FakeCtx(context),
        ),
    )


def test_context_tools_queue_and_import_only_shadow_evidence(
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    opportunity_id = _seed_opportunity(mcp_engine)
    intent_path = tmp_path / "context-intents.db"
    context = _context(mcp_engine, intent_path)
    tools = get_tools(context_critic.register)

    pending = tools["pending_context_opportunities"](
        10,
        20,
        "hermes-shadow-context-1.0.0",
        "optionsbot-context-v1",
        FakeCtx(context),
    )
    assert pending["count"] == 1
    assert pending["opportunities"][0]["opportunity_id"] == opportunity_id
    assert pending["authority"] == "shadow_only_no_order_or_halt_authority"

    packet = tools["context_opportunity_packet"](
        opportunity_id,
        "2026-08-28:SPY:bull:fvg-retest:1",
        FakeCtx(context),
    )
    assert packet["ok"] is True
    assert packet["opportunity"]["scan_admission_action"] == "candidate"
    assert packet["opportunity"]["capture_baseline_action"] == "candidate"
    assert packet["response_contract"]["context_probability_may_be_null"] is True

    result = _submit(tools, context, opportunity_id)
    assert result["status"] == "queued_for_immutable_shadow_audit"
    assert result["authority"] == "shadow_only_no_order_or_halt_authority"
    with context.intent_engine.connect() as conn:
        intent = conn.execute(select(control_intents)).one()
    assert intent.kind == "context_review"
    assert intent.payload_json["submission"]["opportunity_id"] == opportunity_id
    with mcp_engine.connect() as conn:
        assert conn.execute(select(managed_context_reviews.c.id)).fetchall() == []
        assert conn.execute(select(orders.c.id)).fetchall() == []
    assert load_state(mcp_engine).killed is False

    daemon_context = cast(DaemonContext, SimpleNamespace(engine=mcp_engine))
    assert consume_control_intents(daemon_context, intent_path) == 1
    with mcp_engine.connect() as conn:
        review = conn.execute(select(managed_context_reviews)).one()
        assert conn.execute(select(orders.c.id)).fetchall() == []
    assert review.opportunity_id == opportunity_id
    assert review.timing == "pretrade"
    assert review.response_json["signal_id"] == "2026-08-28:SPY:bull:fvg-retest:1"
    assert review.response_json["timing_classification"] == "pretrade"
    assert review.event_conflict == 0
    assert load_state(mcp_engine).killed is False

    duplicate = _submit(tools, context, opportunity_id)
    assert duplicate["already_recorded"] is True
    assert duplicate["review_id"] == review.id


def test_pending_context_queue_deduplicates_signals_and_prioritizes_pretrade(
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    trusted_now = datetime(2026, 8, 31, 15, 0, tzinfo=UTC)
    post_outcome = _seed_opportunity(
        mcp_engine,
        key="post-outcome",
        signal_id="signal-post-outcome",
        detected_at=trusted_now - timedelta(minutes=15),
        entry_ts=trusted_now - timedelta(minutes=10),
        status="resolved",
        outcome="target",
        net_pnl=20.0,
        resolved_at=trusted_now,
    )
    post_entry = _seed_opportunity(
        mcp_engine,
        key="post-entry",
        signal_id="signal-post-entry",
        detected_at=trusted_now - timedelta(minutes=14),
        entry_ts=trusted_now,
    )
    post_cutoff = _seed_opportunity(
        mcp_engine,
        key="post-cutoff",
        signal_id="signal-post-cutoff",
        detected_at=trusted_now - timedelta(minutes=13),
        entry_cutoff_at=trusted_now,
    )
    _seed_opportunity(
        mcp_engine,
        key="shared-post-entry",
        signal_id="signal-shared",
        detected_at=trusted_now - timedelta(minutes=6),
        entry_ts=trusted_now - timedelta(minutes=1),
    )
    shared_pretrade = _seed_opportunity(
        mcp_engine,
        key="shared-pretrade-first",
        signal_id="signal-shared",
        detected_at=trusted_now - timedelta(minutes=5),
    )
    _seed_opportunity(
        mcp_engine,
        key="shared-pretrade-second",
        signal_id="signal-shared",
        detected_at=trusted_now - timedelta(minutes=4),
    )
    other_pretrade = _seed_opportunity(
        mcp_engine,
        key="other-pretrade",
        signal_id="signal-other-pretrade",
        detected_at=trusted_now - timedelta(minutes=3),
    )
    future = _seed_opportunity(
        mcp_engine,
        key="future",
        signal_id="signal-future",
        detected_at=trusted_now + timedelta(minutes=1),
    )
    context = _context(mcp_engine, tmp_path / "ranked-context-intents.db")
    pending = get_tools(context_critic.register)["pending_context_opportunities"]

    with patch(
        "optionsbot.mcp_server.tools.context_critic._utc_now",
        return_value=trusted_now,
    ):
        limited = pending(2, 30, "critic-v1", "prompt-v1", FakeCtx(context))
        complete = pending(25, 30, "critic-v1", "prompt-v1", FakeCtx(context))

    assert [item["opportunity_id"] for item in limited["opportunities"]] == [
        shared_pretrade,
        other_pretrade,
    ]
    assert [item["timing_now"] for item in limited["opportunities"]] == [
        "pretrade",
        "pretrade",
    ]
    assert [item["opportunity_id"] for item in complete["opportunities"]] == [
        shared_pretrade,
        other_pretrade,
        post_cutoff,
        post_entry,
        post_outcome,
    ]
    assert [item["timing_now"] for item in complete["opportunities"]] == [
        "pretrade",
        "pretrade",
        "post_cutoff",
        "post_entry",
        "post_outcome",
    ]
    returned_ids = {item["opportunity_id"] for item in complete["opportunities"]}
    assert future not in returned_ids
    assert complete["count"] == 5


def test_pending_context_queue_excludes_reviewed_or_pending_signal_siblings(
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    trusted_now = datetime(2026, 8, 31, 15, 0, tzinfo=UTC)
    first = _seed_opportunity(
        mcp_engine,
        key="shared-first",
        signal_id="shared-reviewed-signal",
        detected_at=trusted_now - timedelta(minutes=2),
    )
    second = _seed_opportunity(
        mcp_engine,
        key="shared-second",
        signal_id="shared-reviewed-signal",
        detected_at=trusted_now - timedelta(minutes=1),
    )
    _seed_review(
        mcp_engine,
        first,
        suffix="shared-reviewed",
        timing="pretrade",
        event_conflict=False,
        received_at=trusted_now,
    )
    context = _context(mcp_engine, tmp_path / "group-exclusion-intents.db")
    pending = get_tools(context_critic.register)["pending_context_opportunities"]

    with patch(
        "optionsbot.mcp_server.tools.context_critic._utc_now",
        return_value=trusted_now,
    ):
        reviewed = pending(10, 20, "critic-v1", "prompt-v1", FakeCtx(context))
        unreviewed_version = pending(
            10,
            20,
            "critic-v2",
            "prompt-v1",
            FakeCtx(context),
        )

    assert reviewed["opportunities"] == []
    assert [item["opportunity_id"] for item in unreviewed_version["opportunities"]] == [first]

    enqueue_intent(
        context.intent_engine,
        "context_review",
        {
            "received_at": trusted_now.isoformat(),
            "submission": {
                "opportunity_id": second,
                "signal_id": "shared-reviewed-signal",
                "model_version": "critic-v2",
                "prompt_version": "prompt-v1",
            },
        },
        now=trusted_now,
    )
    with patch(
        "optionsbot.mcp_server.tools.context_critic._utc_now",
        return_value=trusted_now,
    ):
        pending_sibling = pending(
            10,
            20,
            "critic-v2",
            "prompt-v1",
            FakeCtx(context),
        )
        different_version = pending(
            10,
            20,
            "critic-v3",
            "prompt-v1",
            FakeCtx(context),
        )

    assert pending_sibling["opportunities"] == []
    assert [item["opportunity_id"] for item in different_version["opportunities"]] == [first]


def test_context_timing_uses_managed_entry_without_a_broker_order(
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    detected_at = datetime.now(UTC) - timedelta(minutes=2)
    managed_entry_at = detected_at + timedelta(minutes=1)
    opportunity_id = _seed_opportunity(
        mcp_engine,
        detected_at=detected_at,
        entry_ts=managed_entry_at,
    )
    intent_path = tmp_path / "managed-entry-context.db"
    context = _context(mcp_engine, intent_path)
    tools = get_tools(context_critic.register)

    pending = tools["pending_context_opportunities"](
        10,
        20,
        "hermes-shadow-context-1.0.0",
        "optionsbot-context-v1",
        FakeCtx(context),
    )
    assert pending["opportunities"][0]["managed_entry_at"] is not None
    assert pending["opportunities"][0]["first_broker_order_at"] is None
    assert pending["opportunities"][0]["timing_now"] == "post_entry"

    result = _submit(tools, context, opportunity_id)
    assert result["timing_at_receipt"] == "post_entry"
    daemon_context = cast(DaemonContext, SimpleNamespace(engine=mcp_engine))
    assert consume_control_intents(daemon_context, intent_path) == 1

    with mcp_engine.connect() as conn:
        review = conn.execute(select(managed_context_reviews)).one()
        assert conn.execute(select(orders.c.id)).fetchall() == []
    assert review.timing == "post_entry"
    assert review.response_json["timing_classification"] == "post_entry"


def test_context_submission_rejects_signal_identity_mismatch(
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    opportunity_id = _seed_opportunity(mcp_engine)
    context = _context(mcp_engine, tmp_path / "mismatch-intents.db")
    tools = get_tools(context_critic.register)

    result = tools["submit_context_review"](
        CONTEXT_CONTRACT_VERSION,
        opportunity_id,
        "different-signal",
        None,
        False,
        [],
        [],
        "hermes-shadow-context-1.0.0",
        "optionsbot-context-v1",
        FakeCtx(context),
    )

    assert result == {"ok": False, "error": "managed_opportunity_signal_mismatch"}
    with context.intent_engine.connect() as conn:
        assert conn.execute(select(control_intents.c.id)).fetchall() == []


def test_daemon_rejects_backdated_context_receipt(
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    opportunity_id = _seed_opportunity(mcp_engine)
    intent_path = tmp_path / "backdated-context.db"
    intent_engine = create_intent_engine(intent_path)
    queued_at = datetime.now(UTC)
    enqueue_intent(
        intent_engine,
        "context_review",
        {
            "received_at": (queued_at - timedelta(minutes=10)).isoformat(),
            "submission": {
                "contract_version": CONTEXT_CONTRACT_VERSION,
                "opportunity_id": opportunity_id,
                "signal_id": "2026-08-28:SPY:bull:fvg-retest:1",
                "context_probability": None,
                "event_conflict": False,
                "anomaly_codes": [],
                "evidence_ids": [],
                "model_version": "hermes-shadow-context-1.0.0",
                "prompt_version": "optionsbot-context-v1",
            },
        },
        now=queued_at,
    )

    daemon_context = cast(DaemonContext, SimpleNamespace(engine=mcp_engine))
    assert consume_control_intents(daemon_context, intent_path) == 0
    with intent_engine.connect() as conn:
        intent = conn.execute(select(control_intents)).one()
    assert intent.status == "rejected"
    assert "must match the local intent receipt time" in intent.result_text
    with mcp_engine.connect() as conn:
        assert conn.execute(select(managed_context_reviews.c.id)).fetchall() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async_consumer", [False, True], ids=["sync", "async"])
async def test_daemon_uses_trusted_consumption_time_for_jointly_forged_queue_timestamps(
    use_async_consumer: bool,
    mcp_engine: Engine,
    tmp_path: Path,
) -> None:
    before = datetime.now(UTC)
    detected_at = before - timedelta(minutes=30)
    entry_at = detected_at + timedelta(minutes=5)
    resolved_at = before - timedelta(minutes=5)
    opportunity_id = _seed_opportunity(
        mcp_engine,
        detected_at=detected_at,
        status="resolved",
        outcome="target",
        net_pnl=20.0,
        resolved_at=resolved_at,
        entry_ts=entry_at,
    )
    intent_path = tmp_path / f"jointly-forged-{'async' if use_async_consumer else 'sync'}.db"
    intent_engine = create_intent_engine(intent_path)
    forged_pretrade_at = detected_at + timedelta(minutes=1)
    with intent_engine.begin() as conn:
        conn.execute(
            insert(control_intents).values(
                intent_uid=f"jointly-forged-{'async' if use_async_consumer else 'sync'}",
                kind="context_review",
                created_at=forged_pretrade_at,
                payload_json={
                    "received_at": forged_pretrade_at.isoformat(),
                    "submission": {
                        "contract_version": CONTEXT_CONTRACT_VERSION,
                        "opportunity_id": opportunity_id,
                        "signal_id": "2026-08-28:SPY:bull:fvg-retest:1",
                        "context_probability": 0.99,
                        "event_conflict": False,
                        "anomaly_codes": ["market_regime_conflict"],
                        "evidence_ids": ["source:post-outcome-observation"],
                        "model_version": "hermes-shadow-context-forged-clock",
                        "prompt_version": "optionsbot-context-v1",
                    },
                },
                status="pending",
            )
        )

    daemon_context = cast(DaemonContext, SimpleNamespace(engine=mcp_engine))
    consumed = (
        await consume_control_intents_async(daemon_context, intent_path)
        if use_async_consumer
        else consume_control_intents(daemon_context, intent_path)
    )
    after = datetime.now(UTC)

    assert consumed == 1
    with intent_engine.connect() as conn:
        intent = conn.execute(select(control_intents)).one()
    assert intent.status == "processed"
    with mcp_engine.connect() as conn:
        review = conn.execute(select(managed_context_reviews)).one()
    trusted_received_at = review.received_at.replace(tzinfo=UTC)
    assert before <= trusted_received_at <= after
    assert trusted_received_at > resolved_at
    assert trusted_received_at != forged_pretrade_at
    assert review.timing == "post_outcome"
    assert review.response_json["timing_classification"] == "post_outcome"
    envelope_received_at = datetime.fromisoformat(
        review.response_json["received_at"].replace("Z", "+00:00")
    )
    assert envelope_received_at == trusted_received_at


def _seed_review(
    engine: Engine,
    opportunity_id: int,
    *,
    suffix: str,
    timing: str,
    event_conflict: bool,
    received_at: datetime | None = None,
) -> None:
    now = received_at or datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(managed_context_reviews).values(
                opportunity_id=opportunity_id,
                received_at=now,
                timing=timing,
                response_json={"review": suffix},
                response_hash=f"response-{suffix}",
                context_probability=0.5,
                event_conflict=int(event_conflict),
                anomaly_json=(["scheduled_macro_event"] if event_conflict else []),
                evidence_json=[f"evidence:{suffix}"],
                model_version="critic-v1",
                prompt_version="prompt-v1",
            )
        )


def test_context_metrics_credit_only_pretrade_event_conflict_disagreements(
    mcp_engine: Engine,
) -> None:
    now = datetime.now(UTC)
    entry_ts = now + timedelta(minutes=1)
    resolved_at = now + timedelta(minutes=3)
    review_at = now + timedelta(seconds=30)
    loss = _seed_opportunity(
        mcp_engine,
        key="loss",
        signal_id="signal-loss",
        status="resolved",
        outcome="stop",
        net_pnl=-15.0,
        resolved_at=resolved_at,
        training_eligible=1,
        entry_ts=entry_ts,
    )
    winner = _seed_opportunity(
        mcp_engine,
        key="winner",
        signal_id="signal-winner",
        status="resolved",
        outcome="target",
        net_pnl=25.0,
        resolved_at=resolved_at,
        training_eligible=1,
        entry_ts=entry_ts,
    )
    agreement = _seed_opportunity(
        mcp_engine,
        key="agreement",
        signal_id="signal-agreement",
        status="resolved",
        outcome="stop",
        net_pnl=-10.0,
        resolved_at=resolved_at,
        training_eligible=1,
        entry_ts=entry_ts,
    )
    hindsight = _seed_opportunity(
        mcp_engine,
        key="hindsight",
        signal_id="signal-hindsight",
        status="resolved",
        outcome="stop",
        net_pnl=-100.0,
        resolved_at=resolved_at,
        training_eligible=1,
        entry_ts=entry_ts,
    )
    late = _seed_opportunity(
        mcp_engine,
        key="late",
        signal_id="signal-late",
        status="resolved",
        outcome="stop",
        net_pnl=-200.0,
        resolved_at=resolved_at,
        training_eligible=1,
        entry_ts=entry_ts,
    )
    missing_entry = _seed_opportunity(
        mcp_engine,
        key="missing-entry",
        signal_id="signal-missing-entry",
        status="censored",
        outcome="censored",
        net_pnl=-300.0,
        resolved_at=resolved_at,
        training_eligible=0,
    )
    _seed_review(
        mcp_engine,
        loss,
        suffix="loss",
        timing="pretrade",
        event_conflict=True,
        received_at=review_at,
    )
    _seed_review(
        mcp_engine,
        winner,
        suffix="winner",
        timing="pretrade",
        event_conflict=True,
        received_at=review_at,
    )
    _seed_review(
        mcp_engine,
        agreement,
        suffix="agreement",
        timing="pretrade",
        event_conflict=False,
        received_at=review_at,
    )
    _seed_review(
        mcp_engine,
        hindsight,
        suffix="hindsight",
        timing="post_outcome",
        event_conflict=True,
        received_at=now + timedelta(minutes=4),
    )
    _seed_review(
        mcp_engine,
        late,
        suffix="late",
        timing="pretrade",
        event_conflict=True,
        received_at=now + timedelta(minutes=2),
    )
    _seed_review(
        mcp_engine,
        missing_entry,
        suffix="missing-entry",
        timing="pretrade",
        event_conflict=True,
        received_at=review_at,
    )

    report = context_shadow_report(mcp_engine)
    critic = report["by_critic"]["critic-v1|prompt-v1"]
    assert critic["observations"] == 6
    assert critic["pretrade_observations"] == 5
    assert critic["causal_pre_managed_entry_observations"] == 3
    assert critic["excluded_pretrade_at_or_after_managed_entry"] == 1
    assert critic["excluded_pretrade_missing_managed_entry_ts"] == 1
    assert critic["judgeable_managed_outcomes"] == 3
    assert critic["event_conflict_disagreements"] == 2
    assert critic["agreements_zero_incremental_credit"] == 1
    assert critic["avoided_losses"] == 15.0
    assert critic["missed_profits"] == 25.0
    assert critic["incremental_net_value"] == -10.0
