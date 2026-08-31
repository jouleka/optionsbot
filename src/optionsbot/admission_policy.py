"""Pure scan-admission policy shared by runtime and managed evaluation.

The model may change an opportunity's expected value, but it must not change
the deterministic score, defined-risk, affordability, ranking, or capacity
rules around that value.  Keeping those rules here prevents promotion evidence
from evaluating a portfolio different from the one the daemon can surface.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optionsbot.config import Settings

def configured_admission_policy(settings: Settings) -> AdmissionSelectionPolicy:
    """Return the selector contract represented by the active daemon config."""

    return AdmissionSelectionPolicy(
        score_floor=float(settings.scan.score_threshold),
        single_trade_cap_pct=float(settings.execution.max_single_trade_risk_pct),
        max_candidates_per_batch=settings.scan.alert_top_n,
        max_admitted_per_session=settings.execution.opening_range_max_entries_per_day,
    )


@dataclass(frozen=True, slots=True)
class AdmissionSelectionPolicy:
    """Immutable policy needed to replay one production scan session."""

    score_floor: float
    single_trade_cap_pct: float
    max_candidates_per_batch: int
    max_admitted_per_session: int

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score_floor)) or not 0.0 <= self.score_floor <= 100.0:
            raise ValueError("admission score floor must be finite and between 0 and 100")
        if (
            not math.isfinite(float(self.single_trade_cap_pct))
            or not 0.0 < self.single_trade_cap_pct <= 1.0
        ):
            raise ValueError("admission single-trade cap must be finite and in (0, 1]")
        if self.max_candidates_per_batch < 1 or self.max_admitted_per_session < 1:
            raise ValueError("admission batch and session limits must be positive")

    def payload(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_payload(cls, raw: object) -> AdmissionSelectionPolicy:
        if not isinstance(raw, dict) or set(raw) != {
            "score_floor",
            "single_trade_cap_pct",
            "max_candidates_per_batch",
            "max_admitted_per_session",
        }:
            raise ValueError("admission selection policy is malformed")
        score_floor = raw["score_floor"]
        cap_pct = raw["single_trade_cap_pct"]
        per_batch = raw["max_candidates_per_batch"]
        per_session = raw["max_admitted_per_session"]
        if (
            isinstance(score_floor, bool)
            or not isinstance(score_floor, int | float)
            or isinstance(cap_pct, bool)
            or not isinstance(cap_pct, int | float)
            or isinstance(per_batch, bool)
            or not isinstance(per_batch, int)
            or isinstance(per_session, bool)
            or not isinstance(per_session, int)
        ):
            raise ValueError("admission selection policy is malformed")
        return cls(
            score_floor=float(score_floor),
            single_trade_cap_pct=float(cap_pct),
            max_candidates_per_batch=per_batch,
            max_admitted_per_session=per_session,
        )


@dataclass(frozen=True, slots=True)
class AdmissionCandidate[T]:
    """Raw decision-time gates plus the model value used for ranking."""

    payload: T
    signal_id: str
    score: float
    expected_value: float | None
    defined_risk: bool
    max_loss: float | None
    account_value_available: bool
    account_value_usd: float | None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def admission_blockers[T](
    candidate: AdmissionCandidate[T],
    policy: AdmissionSelectionPolicy,
) -> list[str]:
    """Return stable reasons the production scan cannot surface a candidate."""

    blockers: list[str] = []
    score = _finite(candidate.score)
    if score is None or score < policy.score_floor:
        score_text = f"{score:.2f}" if score is not None else "unavailable"
        blockers.append(
            f"score_below_floor(score={score_text},floor={policy.score_floor:.2f})"
        )
    expected_value = _finite(candidate.expected_value)
    if expected_value is None or expected_value <= 0.0:
        ev_text = f"{expected_value:.2f}" if expected_value is not None else "unavailable"
        blockers.append(f"non_positive_edge(expected_value={ev_text})")
    max_loss = _finite(candidate.max_loss)
    if not candidate.defined_risk or max_loss is None or max_loss <= 0.0:
        blockers.append("undefined_or_missing_max_loss")
        return blockers
    equity = _finite(candidate.account_value_usd)
    if not candidate.account_value_available or equity is None:
        blockers.append("live_equity_unavailable")
        return blockers
    risk_cap = equity * policy.single_trade_cap_pct
    if max_loss > risk_cap:
        blockers.append(
            "single_contract_risk_over_cap("
            f"max_loss={max_loss:.2f},cap={risk_cap:.2f},"
            f"cap_pct={policy.single_trade_cap_pct:.4f})"
        )
    return blockers


def rank_admission_candidates[T](
    candidates: list[AdmissionCandidate[T]],
    policy: AdmissionSelectionPolicy,
) -> list[AdmissionCandidate[T]]:
    """Filter and stably rank one scan batch by positive EV per dollar at risk."""

    eligible = [candidate for candidate in candidates if not admission_blockers(candidate, policy)]
    # All eligible rows have finite positive EV and max loss. Python's sort is
    # stable, so ties retain the daemon's deterministic scan order and managed
    # replay's opportunity-id order.
    def risk_adjusted_edge(candidate: AdmissionCandidate[T]) -> float:
        expected_value = _finite(candidate.expected_value)
        max_loss = _finite(candidate.max_loss)
        if expected_value is None or max_loss is None or max_loss <= 0.0:
            return float("-inf")
        return expected_value / max_loss

    eligible.sort(key=risk_adjusted_edge, reverse=True)
    selected: list[AdmissionCandidate[T]] = []
    consumed_signals: set[str] = set()
    for candidate in eligible:
        if candidate.signal_id in consumed_signals:
            continue
        consumed_signals.add(candidate.signal_id)
        selected.append(candidate)
    return selected
