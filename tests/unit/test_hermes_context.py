"""Contract tests for the non-authoritative Hermes context critic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from optionsbot.hermes_context import (
    CONTEXT_CONTRACT_VERSION,
    ContextTiming,
    HermesContextSubmissionV1,
    bind_context_response,
    classify_context_timing,
    context_response_hash,
    context_response_payload,
)


def _submission(**overrides: object) -> HermesContextSubmissionV1:
    payload: dict[str, object] = {
        "contract_version": CONTEXT_CONTRACT_VERSION,
        "opportunity_id": 42,
        "signal_id": "2026-08-28:SPY:bull:fvg-retest:1",
        "context_probability": 0.61,
        "event_conflict": False,
        "anomaly_codes": ["market_regime_conflict"],
        "evidence_ids": ["finnhub:quote:SPY:1724851800"],
        "model_version": "hermes-shadow-context-1.0.0",
        "prompt_version": "optionsbot-context-v1",
    }
    payload.update(overrides)
    return HermesContextSubmissionV1.model_validate(payload)


def test_context_contract_is_strict_and_forbids_authority_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _submission(authorize_order=True)
    with pytest.raises(ValidationError):
        _submission(event_conflict=1)
    with pytest.raises(ValidationError):
        _submission(context_probability="0.61")


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.01, 1.01])
def test_context_probability_must_be_null_or_a_finite_probability(
    probability: float,
) -> None:
    with pytest.raises(ValidationError, match="context_probability"):
        _submission(context_probability=probability)

    assert _submission(context_probability=None).context_probability is None


def test_event_conflict_requires_matching_machine_code_and_evidence() -> None:
    with pytest.raises(ValidationError, match="concrete event anomaly code"):
        _submission(event_conflict=True, anomaly_codes=["source_disagreement"])
    with pytest.raises(ValidationError, match="event_conflict=true"):
        _submission(event_conflict=False, anomaly_codes=["scheduled_macro_event"])
    with pytest.raises(ValidationError, match="at least one evidence_id"):
        _submission(
            context_probability=None,
            event_conflict=True,
            anomaly_codes=["scheduled_macro_event"],
            evidence_ids=[],
        )

    result = _submission(
        context_probability=None,
        event_conflict=True,
        anomaly_codes=["scheduled_macro_event"],
        evidence_ids=["fred:event:FOMC:2026-09-16"],
    )
    assert result.event_conflict is True


def test_anomalies_and_evidence_are_deduplicated_strictly() -> None:
    with pytest.raises(ValidationError, match="anomaly_codes must be unique"):
        _submission(anomaly_codes=["source_disagreement", "source_disagreement"])
    with pytest.raises(ValidationError, match="evidence_ids must be unique"):
        _submission(evidence_ids=["Finnhub:news:1", "finnhub:news:1"])


def test_binding_rejects_model_supplied_identity_changes() -> None:
    submission = _submission()
    now = datetime(2026, 8, 28, 14, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match="opportunity_id"):
        bind_context_response(
            submission,
            trusted_opportunity_id=43,
            trusted_signal_id=submission.signal_id,
            received_at=now,
            timing=ContextTiming.PRETRADE,
        )
    with pytest.raises(ValueError, match="signal_id"):
        bind_context_response(
            submission,
            trusted_opportunity_id=submission.opportunity_id,
            trusted_signal_id="different-signal",
            received_at=now,
            timing=ContextTiming.PRETRADE,
        )


def test_timing_classification_is_derived_with_causal_precedence() -> None:
    received = datetime(2026, 8, 28, 15, 0, tzinfo=UTC)
    future = received + timedelta(minutes=1)
    past = received - timedelta(minutes=1)
    assert (
        classify_context_timing(
            received_at=received,
            cutoff_at=future,
            first_entry_at=future,
            outcome_available_at=future,
        )
        is ContextTiming.PRETRADE
    )
    assert (
        classify_context_timing(
            received_at=received,
            cutoff_at=past,
            first_entry_at=future,
            outcome_available_at=future,
        )
        is ContextTiming.POST_CUTOFF
    )
    assert (
        classify_context_timing(
            received_at=received,
            cutoff_at=past,
            first_entry_at=past,
            outcome_available_at=future,
        )
        is ContextTiming.POST_ENTRY
    )
    assert (
        classify_context_timing(
            received_at=received,
            cutoff_at=past,
            first_entry_at=past,
            outcome_available_at=past,
        )
        is ContextTiming.POST_OUTCOME
    )


def test_bound_response_hash_is_stable_and_contains_trusted_timing() -> None:
    submission = _submission()
    response = bind_context_response(
        submission,
        trusted_opportunity_id=submission.opportunity_id,
        trusted_signal_id=submission.signal_id,
        received_at=datetime(2026, 8, 28, 15, 0, tzinfo=UTC),
        timing=ContextTiming.POST_ENTRY,
    )

    assert response.timing_classification.causal_bucket == "posttrade"
    assert context_response_payload(response)["timing_classification"] == "post_entry"
    assert context_response_hash(response) == context_response_hash(response)
    assert len(context_response_hash(response)) == 64


def test_naive_timing_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        classify_context_timing(
            received_at=datetime(2026, 8, 28, 15, 0),
            cutoff_at=None,
            first_entry_at=None,
            outcome_available_at=None,
        )
