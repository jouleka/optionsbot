"""Version identities for prospective managed-path research data.

The outcome policy is part of every opportunity key and model artifact.  A
capture setting that changes which quotes are usable or when a boundary can be
observed therefore changes the data-generating policy; it must not silently
reuse another policy's identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

MANAGED_FEATURE_SCHEMA_VERSION = "managed_capture_features_v1"
DEFAULT_MANAGED_OUTCOME_POLICY_VERSION = "marketable_nbbo_15s_v1"


@dataclass(frozen=True, slots=True)
class ManagedOutcomePolicySpec:
    """Configurable settings that change managed-label observability."""

    capture_interval_seconds: int
    capture_offset_seconds: int
    quote_max_age_seconds: int
    quote_span_seconds: int
    max_mark_gap_seconds: int


DEFAULT_MANAGED_OUTCOME_POLICY_SPEC = ManagedOutcomePolicySpec(
    capture_interval_seconds=15,
    capture_offset_seconds=5,
    quote_max_age_seconds=45,
    quote_span_seconds=10,
    max_mark_gap_seconds=45,
)


def derive_managed_outcome_policy_version(spec: ManagedOutcomePolicySpec) -> str:
    """Return the stable identity of one managed-label data policy.

    The established default keeps its existing human-readable identity. Any
    change to a label-affecting knob receives a deterministic fingerprint, so
    old and new opportunities, marks, and artifacts cannot be pooled silently.
    """
    if spec == DEFAULT_MANAGED_OUTCOME_POLICY_SPEC:
        return DEFAULT_MANAGED_OUTCOME_POLICY_VERSION
    canonical = json.dumps(
        {
            "algorithm": "marketable_combo_nbbo_first_observed_v1",
            **asdict(spec),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    return f"marketable_nbbo_{spec.capture_interval_seconds}s_v1-{digest}"


def validate_managed_contract(
    *,
    feature_schema_version: str,
    outcome_policy_version: str,
    outcome_policy_spec: ManagedOutcomePolicySpec,
) -> None:
    """Fail closed when configured identities do not describe produced data."""
    if feature_schema_version != MANAGED_FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "managed_learning.feature_schema_version is unsupported by the "
            f"installed encoder; expected {MANAGED_FEATURE_SCHEMA_VERSION!r}"
        )
    expected_policy = derive_managed_outcome_policy_version(outcome_policy_spec)
    if outcome_policy_version != expected_policy:
        raise ValueError(
            "managed_learning.outcome_policy_version does not identify the "
            "configured managed-capture semantics; "
            f"expected {expected_policy!r}"
        )
