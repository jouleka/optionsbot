"""Strict, non-authoritative contract for Hermes context observations.

Hermes is an external-context critic, not an execution policy.  This module is
deliberately pure: it validates the model response, binds it to trusted daemon
identity, classifies when the observation was received, and produces a stable
hash for immutable audit storage.  Nothing here can authorize or modify an
order.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

CONTEXT_CONTRACT_VERSION = "hermes-context-critic/v1"

Identity = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$",
    ),
]
Version = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$",
    ),
]
EvidenceId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@#+?&=\-]*$",
    ),
]


class ContextAnomalyCode(StrEnum):
    """Closed vocabulary for machine-countable Hermes observations."""

    SCHEDULED_MACRO_EVENT = "scheduled_macro_event"
    ISSUER_EVENT_CONFLICT = "issuer_event_conflict"
    MATERIAL_NEWS_CONFLICT = "material_news_conflict"
    MARKET_REGIME_CONFLICT = "market_regime_conflict"
    SOURCE_DISAGREEMENT = "source_disagreement"
    CONTEXT_DATA_MISSING = "context_data_missing"
    CONTEXT_DATA_STALE = "context_data_stale"
    DAEMON_HEALTH_ANOMALY = "daemon_health_anomaly"
    BROKER_RECONCILE_ANOMALY = "broker_reconcile_anomaly"
    QUOTE_INTEGRITY_ANOMALY = "quote_integrity_anomaly"
    POSITION_STATE_ANOMALY = "position_state_anomaly"


EVENT_CONFLICT_CODES = frozenset(
    {
        ContextAnomalyCode.SCHEDULED_MACRO_EVENT,
        ContextAnomalyCode.ISSUER_EVENT_CONFLICT,
        ContextAnomalyCode.MATERIAL_NEWS_CONFLICT,
    }
)


class ContextTiming(StrEnum):
    """Causal timing assigned from trusted timestamps, never model prose."""

    PRETRADE = "pretrade"
    POST_CUTOFF = "post_cutoff"
    POST_ENTRY = "post_entry"
    POST_OUTCOME = "post_outcome"

    @property
    def causal_bucket(self) -> Literal["pretrade", "posttrade"]:
        """Return the coarse storage bucket used by the managed-data schema."""
        return "pretrade" if self is ContextTiming.PRETRADE else "posttrade"


class HermesContextSubmissionV1(BaseModel):
    """Untrusted structured payload produced by Hermes.

    ``context_probability`` means the probability that the independently
    observed context supports the signal's stated direction.  It is not the
    probability of profit, target-first, or a positive managed return.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract_version: Literal["hermes-context-critic/v1"]
    opportunity_id: Annotated[StrictInt, Field(gt=0)]
    signal_id: Identity
    context_probability: float | None
    event_conflict: bool
    anomaly_codes: tuple[ContextAnomalyCode, ...] = Field(max_length=16)
    evidence_ids: tuple[EvidenceId, ...] = Field(max_length=32)
    model_version: Version
    prompt_version: Version

    @field_validator("anomaly_codes", mode="before")
    @classmethod
    def parse_anomaly_codes(cls, value: object) -> tuple[ContextAnomalyCode, ...]:
        """Accept a JSON array while retaining strict scalar validation."""
        if not isinstance(value, (list, tuple)):
            raise ValueError("anomaly_codes must be an array")
        parsed: list[ContextAnomalyCode] = []
        for item in value:
            if type(item) is not str:
                raise ValueError("anomaly_codes must contain strings")
            try:
                parsed.append(ContextAnomalyCode(item))
            except ValueError as exc:
                raise ValueError(f"unknown anomaly code: {item}") from exc
        return tuple(parsed)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def parse_evidence_ids(cls, value: object) -> tuple[object, ...]:
        """Accept a JSON array without coercing its members to strings."""
        if not isinstance(value, (list, tuple)):
            raise ValueError("evidence_ids must be an array")
        if any(type(item) is not str for item in value):
            raise ValueError("evidence_ids must contain strings")
        return tuple(value)

    @model_validator(mode="after")
    def validate_semantics(self) -> HermesContextSubmissionV1:
        probability = self.context_probability
        if probability is not None and (
            not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("context_probability must be null or finite within [0, 1]")
        if len(set(self.anomaly_codes)) != len(self.anomaly_codes):
            raise ValueError("anomaly_codes must be unique")
        if len({item.casefold() for item in self.evidence_ids}) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")
        event_codes = set(self.anomaly_codes) & EVENT_CONFLICT_CODES
        if self.event_conflict and not event_codes:
            raise ValueError("event_conflict requires a concrete event anomaly code")
        if not self.event_conflict and event_codes:
            raise ValueError("event anomaly codes require event_conflict=true")
        if (probability is not None or self.event_conflict or self.anomaly_codes) and not (
            self.evidence_ids
        ):
            raise ValueError("a non-empty observation requires at least one evidence_id")
        return self


class HermesContextResponseV1(HermesContextSubmissionV1):
    """Trusted audit envelope returned by the MCP boundary and persisted."""

    received_at: datetime
    timing_classification: ContextTiming

    @model_validator(mode="after")
    def validate_received_at(self) -> HermesContextResponseV1:
        if self.received_at.tzinfo is None or self.received_at.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware")
        return self


def bind_context_response(
    submission: HermesContextSubmissionV1,
    *,
    trusted_opportunity_id: int,
    trusted_signal_id: str,
    received_at: datetime,
    timing: ContextTiming,
) -> HermesContextResponseV1:
    """Bind an untrusted response to exact daemon-owned identity and timing."""
    if submission.opportunity_id != trusted_opportunity_id:
        raise ValueError("opportunity_id does not match trusted daemon identity")
    if submission.signal_id != trusted_signal_id:
        raise ValueError("signal_id does not match trusted daemon identity")
    aware_received_at = _aware_utc(received_at, "received_at")
    return HermesContextResponseV1(
        **submission.model_dump(mode="json"),
        received_at=aware_received_at,
        timing_classification=timing,
    )


def classify_context_timing(
    *,
    received_at: datetime,
    cutoff_at: datetime | None,
    first_entry_at: datetime | None,
    outcome_available_at: datetime | None,
) -> ContextTiming:
    """Classify causal eligibility with terminal observations taking precedence."""
    received = _aware_utc(received_at, "received_at")
    outcome = _optional_aware_utc(outcome_available_at, "outcome_available_at")
    entry = _optional_aware_utc(first_entry_at, "first_entry_at")
    cutoff = _optional_aware_utc(cutoff_at, "cutoff_at")
    if outcome is not None and outcome <= received:
        return ContextTiming.POST_OUTCOME
    if entry is not None and entry <= received:
        return ContextTiming.POST_ENTRY
    if cutoff is not None and cutoff <= received:
        return ContextTiming.POST_CUTOFF
    return ContextTiming.PRETRADE


def earliest_context_entry(
    managed_entry_at: datetime | None,
    broker_entry_at: datetime | None,
) -> datetime | None:
    """Return the first event that starts either shadow or executed exposure."""
    candidates = [
        aware
        for value, name in (
            (managed_entry_at, "managed_entry_at"),
            (broker_entry_at, "broker_entry_at"),
        )
        if value is not None
        for aware in (_aware_utc(value, name),)
    ]
    return min(candidates) if candidates else None


def context_response_payload(response: HermesContextResponseV1) -> dict[str, object]:
    """Return the canonical JSON-compatible audit payload."""
    return response.model_dump(mode="json")


def context_response_hash(response: HermesContextResponseV1) -> str:
    """Hash the exact canonical response for immutable deduplication."""
    canonical = json.dumps(
        context_response_payload(response),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _optional_aware_utc(value: datetime | None, name: str) -> datetime | None:
    return None if value is None else _aware_utc(value, name)
