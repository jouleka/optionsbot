"""Leakage-resistant managed-outcome challenger model.

The production question is not whether an option finishes profitable at
expiry.  It is which management boundary is observed first: target, stop, or
timeout.  This module fits that three-way event directly and evaluates the
result chronologically by complete trading session.

Artifacts are plain, checksummed JSON data rather than pickle files.  That
makes promotion reviewable and avoids executing code while loading a model.
The implementation deliberately uses a small regularized softmax model: the
capture set will initially be small, and a transparent baseline is preferable
to a high-capacity learner that can memorize symbol/session noise.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from typing import Any, Literal

from optionsbot.admission_policy import (
    AdmissionCandidate,
    AdmissionSelectionPolicy,
    rank_admission_candidates,
)
from optionsbot.managed_contract import (
    DEFAULT_MANAGED_OUTCOME_POLICY_VERSION,
    MANAGED_FEATURE_SCHEMA_VERSION,
)

Outcome = Literal["target", "stop", "timeout"]
_OUTCOMES: tuple[Outcome, ...] = ("target", "stop", "timeout")
_EPSILON = 1e-12
DEFAULT_FEATURE_SCHEMA_VERSION = MANAGED_FEATURE_SCHEMA_VERSION
DEFAULT_OUTCOME_POLICY_VERSION = DEFAULT_MANAGED_OUTCOME_POLICY_VERSION


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sample_group(row: ManagedSample) -> str:
    return row.signal_id or row.opportunity_key


def _sample_weights(rows: Sequence[ManagedSample]) -> list[float]:
    """Give every independent thesis total weight one.

    The capture layer may retain several correlated structures for research.
    Duplicating one of those rows must not move medians, normalization, class
    priors, calibration, or residual uncertainty.
    """
    counts = Counter(_sample_group(row) for row in rows)
    return [1.0 / counts[_sample_group(row)] for row in rows]


def _weighted_median(values: Sequence[tuple[float, float]]) -> float:
    if not values:
        raise ValueError("weighted median requires observations")
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    midpoint = total / 2.0
    cumulative = 0.0
    for index, (value, weight) in enumerate(ordered):
        cumulative += weight
        if cumulative > midpoint:
            return value
        if math.isclose(cumulative, midpoint) and index + 1 < len(ordered):
            return (value + ordered[index + 1][0]) / 2.0
    return ordered[-1][0]


@dataclass(frozen=True, slots=True)
class ManagedSample:
    """One prospectively captured structure-level observation.

    Alternative executable structures may share ``signal_id``. Fitting and
    evaluation then give that signal total weight one and admit at most one of
    its structures. ``realized_net_pnl`` and all payoff values are dollars per
    one structure unit after the recorded commission convention.
    """

    opportunity_id: int
    opportunity_key: str
    session: str
    features: Mapping[str, float | int | None]
    outcome: Outcome
    basis_dollars: float
    target_gain: float
    stop_loss: float
    timeout_gross_return: float
    costs: float
    realized_net_pnl: float
    # All alternative structures generated from one thesis share this ID.
    # It lets training weight a signal once and lets evaluation choose at most
    # one structure from it instead of pretending correlated variants are
    # independent trades. Older callers may omit it and fall back to the
    # opportunity key.
    signal_id: str | None = None
    # Immutable scan-selector inputs. Defaults keep synthetic/unit samples
    # ergonomic; repository-loaded production samples always bind every field.
    decision_batch_id: str | None = None
    detected_at: datetime | None = None
    decision_score: float = 100.0
    decision_defined_risk: bool = True
    decision_max_loss: float | None = 1.0
    decision_account_value_available: bool = True
    decision_account_value_usd: float | None = 1_000_000_000.0

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.session)
        except ValueError as exc:
            raise ValueError("session must be an ISO calendar date") from exc
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"unsupported managed outcome: {self.outcome}")
        for name in (
            "basis_dollars",
            "target_gain",
            "stop_loss",
            "timeout_gross_return",
            "costs",
            "realized_net_pnl",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            self.basis_dollars <= 0.0
            or self.target_gain <= 0.0
            or self.stop_loss <= 0.0
            or self.costs < 0.0
        ):
            raise ValueError("basis/gain/loss must be positive and costs non-negative")
        if self.decision_batch_id is not None and not self.decision_batch_id:
            raise ValueError("decision batch identity cannot be empty")
        if self.detected_at is not None:
            if self.detected_at.tzinfo is None:
                raise ValueError("decision detection timestamp must be timezone-aware")
            if self.detected_at.astimezone(UTC).date().isoformat() != self.session:
                raise ValueError("decision detection timestamp must belong to sample session")
        if (
            not math.isfinite(float(self.decision_score))
            or not 0.0 <= self.decision_score <= 100.0
        ):
            raise ValueError("decision score must be finite and between 0 and 100")
        if self.decision_max_loss is not None and (
            not math.isfinite(float(self.decision_max_loss)) or self.decision_max_loss <= 0.0
        ):
            raise ValueError("decision max loss must be positive when present")
        if self.decision_account_value_available:
            if self.decision_account_value_usd is None or not math.isfinite(
                float(self.decision_account_value_usd)
            ):
                raise ValueError("available decision account value must be finite")
        elif self.decision_account_value_usd is not None:
            raise ValueError("unavailable decision account value must be null")


@dataclass(frozen=True, slots=True)
class FeatureEncoder:
    names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Sequence[ManagedSample]) -> FeatureEncoder:
        candidates = sorted(
            {
                name
                for row in rows
                for name, value in row.features.items()
                if _finite(value) is not None
            }
        )
        row_weights = _sample_weights(rows)
        effective_samples = sum(row_weights)
        minimum_presence = max(3.0, math.ceil(effective_samples * 0.20))
        names: list[str] = []
        prepared: list[tuple[float, float, float]] = []
        for name in candidates:
            present = [
                (value, weight)
                for row, weight in zip(rows, row_weights, strict=True)
                if (value := _finite(row.features.get(name))) is not None
            ]
            if sum(weight for _, weight in present) < minimum_presence:
                continue
            median = _weighted_median(present)
            filled = [
                value if (value := _finite(row.features.get(name))) is not None else median
                for row in rows
            ]
            mean = (
                sum(value * weight for value, weight in zip(filled, row_weights, strict=True))
                / effective_samples
            )
            variance = (
                sum(
                    weight * (value - mean) ** 2
                    for value, weight in zip(filled, row_weights, strict=True)
                )
                / effective_samples
            )
            # Constant identifiers/configuration add parameters but no signal.
            if variance <= 1e-12:
                continue
            names.append(name)
            prepared.append((median, mean, math.sqrt(variance)))
        if not names:
            raise ValueError("managed model requires at least one finite feature")
        return cls(
            tuple(names),
            tuple(item[0] for item in prepared),
            tuple(item[1] for item in prepared),
            tuple(item[2] for item in prepared),
        )

    def transform(self, features: Mapping[str, float | int | None]) -> tuple[float, ...]:
        encoded: list[float] = []
        for name, median, mean, scale in zip(
            self.names, self.medians, self.means, self.scales, strict=True
        ):
            value = _finite(features.get(name))
            missing = value is None
            actual = median if value is None else value
            encoded.append((actual - mean) / scale)
            encoded.append(1.0 if missing else 0.0)
        return tuple(encoded)


@dataclass(frozen=True, slots=True)
class ManagedModelArtifact:
    model_version: str
    feature_schema_version: str
    outcome_policy_version: str
    encoder: FeatureEncoder
    weights: tuple[tuple[float, ...], ...]
    temperature: float
    timeout_expected_return: float
    ev_residual_q05: float | None
    calibration_effective_sessions: int
    trained_from_session: str
    trained_through_session: str
    sample_count: int
    session_count: int
    artifact_hash: str = ""

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("artifact_hash", None)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))

    def with_hash(self) -> ManagedModelArtifact:
        digest = hashlib.sha256(self.canonical_json().encode()).hexdigest()
        return replace(self, artifact_hash=digest)

    def verify(self) -> None:
        expected = self.with_hash().artifact_hash
        if not self.artifact_hash or not hmac.compare_digest(self.artifact_hash, expected):
            raise ValueError("managed model artifact checksum mismatch")
        if len(self.weights) != len(_OUTCOMES):
            raise ValueError("managed model must have target/stop/timeout weights")
        expected_width = 1 + 2 * len(self.encoder.names)
        if any(len(row) != expected_width for row in self.weights):
            raise ValueError("managed model coefficient width does not match encoder")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("managed model temperature must be positive")
        if not math.isfinite(self.timeout_expected_return):
            raise ValueError("managed model timeout expectation must be finite")

    def to_json(self) -> str:
        checked = self if self.artifact_hash else self.with_hash()
        return json.dumps(asdict(checked), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> ManagedModelArtifact:
        payload = json.loads(raw)
        encoder_raw = payload.pop("encoder")
        artifact = cls(
            encoder=FeatureEncoder(
                names=tuple(encoder_raw["names"]),
                medians=tuple(float(value) for value in encoder_raw["medians"]),
                means=tuple(float(value) for value in encoder_raw["means"]),
                scales=tuple(float(value) for value in encoder_raw["scales"]),
            ),
            weights=tuple(tuple(float(value) for value in row) for row in payload.pop("weights")),
            **payload,
        )
        artifact.verify()
        return artifact


@dataclass(frozen=True, slots=True)
class ManagedPrediction:
    target_probability: float
    stop_probability: float
    timeout_probability: float
    target_probability_lcb: float
    expected_value: float
    expected_value_lcb: float
    model_version: str
    artifact_hash: str


@dataclass(frozen=True, slots=True)
class WalkForwardRow:
    opportunity_id: int
    session: str
    outcome: Outcome
    probabilities: tuple[float, float, float]
    expected_value: float
    expected_value_lcb: float
    realized_net_pnl: float
    signal_id: str | None = None
    decision_batch_id: str | None = None
    detected_at: datetime | None = None
    decision_score: float = 100.0
    decision_defined_risk: bool = True
    decision_max_loss: float | None = 1.0
    decision_account_value_available: bool = True
    decision_account_value_usd: float | None = 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class WalkForwardMetrics:
    samples: int
    independent_signals: int
    sessions: int
    folds: int
    multiclass_brier: float
    baseline_brier: float
    log_loss: float
    baseline_log_loss: float
    calibration_error: float
    admitted: int
    admitted_sessions: int
    admitted_net_pnl: float
    admitted_mean_pnl: float
    admitted_mean_pnl_lcb: float
    profit_factor: float | None
    max_drawdown: float


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    min_sessions: int = 30
    min_samples: int = 100
    min_independent_signals: int = 100
    min_oof_samples: int = 50
    min_folds: int = 10
    min_admitted: int = 20
    min_admitted_sessions: int = 10
    min_profit_factor: float = 1.05
    score_floor: float = 0.0
    single_trade_cap_pct: float = 1.0
    max_candidates_per_batch: int = 3
    max_admitted_per_session: int = 3
    bootstrap_iterations: int = 2_000

    def admission_selection_policy(self) -> AdmissionSelectionPolicy:
        return AdmissionSelectionPolicy(
            score_floor=self.score_floor,
            single_trade_cap_pct=self.single_trade_cap_pct,
            max_candidates_per_batch=self.max_candidates_per_batch,
            max_admitted_per_session=self.max_admitted_per_session,
        )


@dataclass(frozen=True, slots=True)
class PromotionReport:
    eligible: bool
    reasons: tuple[str, ...]
    metrics: WalkForwardMetrics
    rows: tuple[WalkForwardRow, ...]


@dataclass(frozen=True, slots=True)
class ContextIncrementalReport:
    """Causal utility of a context challenger relative to the bot baseline."""

    eligible: bool
    disagreements: int
    disagreement_sessions: int
    incremental_net_pnl: float
    mean_incremental_pnl: float
    mean_incremental_pnl_lcb: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProspectiveReport:
    """One immutable future block scored by an already-frozen artifact."""

    eligible: bool
    samples: int
    independent_signals: int
    sessions: int
    admitted: int
    admitted_sessions: int
    admitted_net_pnl: float
    admitted_mean_pnl: float
    admitted_mean_pnl_lcb: float
    profit_factor: float | None
    max_drawdown: float
    reasons: tuple[str, ...]
    rows: tuple[WalkForwardRow, ...]


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    scaled = [value / temperature for value in logits]
    high = max(scaled)
    exp_values = [math.exp(value - high) for value in scaled]
    total = sum(exp_values)
    return tuple(value / total for value in exp_values)


def _logits(weights: Sequence[Sequence[float]], encoded: Sequence[float]) -> tuple[float, ...]:
    vector = (1.0, *encoded)
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, vector, strict=True))
        for row in weights
    )


def _fit_weights(
    rows: Sequence[ManagedSample],
    encoder: FeatureEncoder,
    *,
    l2: float = 0.1,
    learning_rate: float = 0.08,
    iterations: int = 500,
) -> tuple[tuple[float, ...], ...]:
    if len(rows) < 3:
        raise ValueError("managed model requires at least three samples")
    sample_weights = _sample_weights(rows)
    total = sum(sample_weights)
    class_counts = [
        sum(
            sample_weight
            for row, sample_weight in zip(rows, sample_weights, strict=True)
            if row.outcome == outcome
        )
        for outcome in _OUTCOMES
    ]
    if any(count == 0 for count in class_counts):
        raise ValueError("managed model requires target, stop, and timeout observations")
    width = 1 + 2 * len(encoder.names)
    weights = [[0.0] * width for _ in _OUTCOMES]
    for class_index, count in enumerate(class_counts):
        weights[class_index][0] = math.log(count / total)
    encoded_rows = [encoder.transform(row.features) for row in rows]
    targets = [_OUTCOMES.index(row.outcome) for row in rows]
    for step in range(iterations):
        gradients = [[0.0] * width for _ in _OUTCOMES]
        for encoded, target, sample_weight in zip(
            encoded_rows, targets, sample_weights, strict=True
        ):
            vector = (1.0, *encoded)
            probabilities = _softmax(_logits(weights, encoded))
            for class_index, probability in enumerate(probabilities):
                error = probability - (1.0 if class_index == target else 0.0)
                for feature_index, value in enumerate(vector):
                    gradients[class_index][feature_index] += sample_weight * error * value
        rate = learning_rate / math.sqrt(1.0 + step / 200.0)
        for class_index in range(len(_OUTCOMES)):
            for feature_index in range(width):
                regularizer = l2 * weights[class_index][feature_index] if feature_index > 0 else 0.0
                gradient = gradients[class_index][feature_index] / total + regularizer
                weights[class_index][feature_index] -= rate * gradient
    return tuple(tuple(row) for row in weights)


def _fit_temperature(
    weights: Sequence[Sequence[float]],
    encoder: FeatureEncoder,
    rows: Sequence[ManagedSample],
) -> float:
    if not rows:
        return 1.0
    row_weights = _sample_weights(rows)
    weight_total = sum(row_weights)
    best_temperature = 1.0
    best_loss = float("inf")
    for step in range(10, 81):
        temperature = step / 20.0
        loss = 0.0
        for row, row_weight in zip(rows, row_weights, strict=True):
            probabilities = _softmax(_logits(weights, encoder.transform(row.features)), temperature)
            loss -= row_weight * math.log(
                max(probabilities[_OUTCOMES.index(row.outcome)], _EPSILON)
            )
        loss /= weight_total
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _weighted_lower_quantile(values: Sequence[tuple[float, float]], probability: float) -> float:
    """Return a conservative weighted empirical lower quantile.

    Unlike an interpolated row quantile, this is invariant to adding another
    correlated structure to an existing signal when each signal's total
    weight remains one.
    """
    if not values:
        return 0.0
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered if weight > 0.0)
    if total <= 0.0:
        raise ValueError("residual weights must contain positive mass")
    threshold = max(0.0, min(1.0, probability)) * total
    cumulative = 0.0
    for value, weight in ordered:
        if weight <= 0.0:
            continue
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _wilson_lower(success_probability: float, effective_n: int, z: float = 1.96) -> float:
    if effective_n <= 0:
        return 0.0
    p = max(0.0, min(1.0, success_probability))
    n = float(effective_n)
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (center - margin) / denominator)


def _raw_prediction(
    artifact: ManagedModelArtifact,
    features: Mapping[str, float | int | None],
) -> tuple[float, float, float]:
    artifact.verify()
    result = _softmax(
        _logits(artifact.weights, artifact.encoder.transform(features)),
        artifact.temperature,
    )
    return (result[0], result[1], result[2])


def predict_managed_outcome(
    artifact: ManagedModelArtifact,
    features: Mapping[str, float | int | None],
    *,
    basis_dollars: float,
    target_gain: float,
    stop_loss: float,
    costs: float,
) -> ManagedPrediction:
    """Predict path probabilities and conservative after-cost value.

    The residual lower bound is learned only from chronological out-of-fold
    predictions.  It therefore represents observed model/payoff error rather
    than an invented probability haircut.
    """
    values = (basis_dollars, target_gain, stop_loss, costs)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("managed economics must be finite")
    if basis_dollars <= 0.0 or target_gain <= 0.0 or stop_loss <= 0.0 or costs < 0.0:
        raise ValueError("managed basis/gain/loss must be positive and costs non-negative")
    target, stop, timeout = _raw_prediction(artifact, features)
    expected = (
        target * target_gain
        - stop * stop_loss
        + timeout * artifact.timeout_expected_return * basis_dollars
        - costs
    )
    return ManagedPrediction(
        target_probability=target,
        stop_probability=stop,
        timeout_probability=timeout,
        target_probability_lcb=_wilson_lower(target, artifact.calibration_effective_sessions),
        expected_value=expected,
        expected_value_lcb=(
            expected + artifact.ev_residual_q05 * basis_dollars
            if artifact.ev_residual_q05 is not None
            else float("-inf")
        ),
        model_version=artifact.model_version,
        artifact_hash=artifact.artifact_hash,
    )


def _split_calibration_sessions(
    rows: Sequence[ManagedSample],
) -> tuple[list[ManagedSample], list[ManagedSample]]:
    sessions = sorted({row.session for row in rows})
    if len(sessions) < 5:
        return list(rows), []
    calibration_count = max(1, len(sessions) // 5)
    calibration_sessions = set(sessions[-calibration_count:])
    return (
        [row for row in rows if row.session not in calibration_sessions],
        [row for row in rows if row.session in calibration_sessions],
    )


def fit_managed_model(
    rows: Sequence[ManagedSample],
    *,
    model_version: str,
    feature_schema_version: str,
    outcome_policy_version: str,
    ev_residuals: Sequence[float] = (),
    ev_residual_groups: Sequence[str] = (),
) -> ManagedModelArtifact:
    """Fit one final challenger artifact from chronologically-valid samples."""
    if not rows:
        raise ValueError("cannot fit managed model without samples")
    ordered = sorted(rows, key=lambda row: (row.session, row.opportunity_id))
    train_rows, calibration_rows = _split_calibration_sessions(ordered)
    if any(not any(row.outcome == outcome for row in train_rows) for outcome in _OUTCOMES):
        train_rows, calibration_rows = list(ordered), []
    calibration_encoder = FeatureEncoder.fit(train_rows)
    calibration_weights = _fit_weights(train_rows, calibration_encoder)
    temperature = _fit_temperature(calibration_weights, calibration_encoder, calibration_rows)
    # Keep the calibration slice genuinely unseen by the published
    # coefficients. Re-fitting on it after choosing temperature would make the
    # advertised calibration in-sample.
    encoder = calibration_encoder
    weights = calibration_weights
    sessions = sorted({row.session for row in ordered})
    timeout_rows = [row for row in ordered if row.outcome == "timeout"]
    timeout_groups = Counter(row.signal_id or row.opportunity_key for row in timeout_rows)
    timeout_weighted = [
        (
            row.timeout_gross_return,
            1.0 / timeout_groups[row.signal_id or row.opportunity_key],
        )
        for row in timeout_rows
    ]
    if ev_residual_groups and len(ev_residual_groups) != len(ev_residuals):
        raise ValueError("residual values and groups must have equal length")
    residual_groups = (
        tuple(ev_residual_groups)
        if ev_residual_groups
        else tuple(f"residual:{index}" for index in range(len(ev_residuals)))
    )
    residual_counts = Counter(residual_groups)
    weighted_residuals = [
        (float(value), 1.0 / residual_counts[group])
        for value, group in zip(ev_residuals, residual_groups, strict=True)
    ]
    artifact = ManagedModelArtifact(
        model_version=model_version,
        feature_schema_version=feature_schema_version,
        outcome_policy_version=outcome_policy_version,
        encoder=encoder,
        weights=weights,
        temperature=temperature,
        timeout_expected_return=(
            sum(value * weight for value, weight in timeout_weighted)
            / sum(weight for _, weight in timeout_weighted)
            if timeout_weighted
            else 0.0
        ),
        ev_residual_q05=(
            _weighted_lower_quantile(weighted_residuals, 0.05) if weighted_residuals else None
        ),
        calibration_effective_sessions=len({row.session for row in calibration_rows}),
        trained_from_session=sessions[0],
        trained_through_session=sessions[-1],
        sample_count=len(ordered),
        session_count=len(sessions),
    )
    return artifact.with_hash()


def _point_expected_value(
    probabilities: Sequence[float],
    sample: ManagedSample,
    timeout_expected_return: float,
) -> float:
    return (
        probabilities[0] * sample.target_gain
        - probabilities[1] * sample.stop_loss
        + probabilities[2] * timeout_expected_return * sample.basis_dollars
        - sample.costs
    )


def score_frozen_artifact(
    artifact: ManagedModelArtifact,
    samples: Sequence[ManagedSample],
) -> tuple[WalkForwardRow, ...]:
    """Score a strictly future cohort without fitting or recalibrating."""
    rows: list[WalkForwardRow] = []
    for sample in sorted(samples, key=lambda item: (item.session, item.opportunity_id)):
        if sample.session <= artifact.trained_through_session:
            raise ValueError("prospective sample overlaps artifact training window")
        prediction = predict_managed_outcome(
            artifact,
            sample.features,
            basis_dollars=sample.basis_dollars,
            target_gain=sample.target_gain,
            stop_loss=sample.stop_loss,
            costs=sample.costs,
        )
        rows.append(
            WalkForwardRow(
                opportunity_id=sample.opportunity_id,
                session=sample.session,
                outcome=sample.outcome,
                probabilities=(
                    prediction.target_probability,
                    prediction.stop_probability,
                    prediction.timeout_probability,
                ),
                expected_value=prediction.expected_value,
                expected_value_lcb=prediction.expected_value_lcb,
                realized_net_pnl=sample.realized_net_pnl,
                signal_id=sample.signal_id,
                decision_batch_id=sample.decision_batch_id,
                detected_at=sample.detected_at,
                decision_score=sample.decision_score,
                decision_defined_risk=sample.decision_defined_risk,
                decision_max_loss=sample.decision_max_loss,
                decision_account_value_available=(
                    sample.decision_account_value_available
                ),
                decision_account_value_usd=sample.decision_account_value_usd,
            )
        )
    return tuple(rows)


def _selected_admissions(
    rows: Sequence[WalkForwardRow],
    *,
    policy: AdmissionSelectionPolicy,
) -> list[WalkForwardRow]:
    """Causally replay production batches until each session's cap is spent.

    A later scan may never replace a trade admitted by an earlier scan. Within
    each batch the shared runtime selector applies the same score, positive
    edge, defined-risk, affordability, one-signal, and EV/max-loss ordering.
    """
    by_session: dict[str, list[WalkForwardRow]] = {}
    for row in rows:
        by_session.setdefault(row.session, []).append(row)
    admitted: list[WalkForwardRow] = []
    for session in sorted(by_session):
        batches: dict[str, list[WalkForwardRow]] = {}
        for row in sorted(by_session[session], key=lambda item: item.opportunity_id):
            batch_id = row.decision_batch_id or f"legacy-session:{session}"
            batches.setdefault(batch_id, []).append(row)
        ordered_batches = sorted(
            batches.items(),
            key=lambda item: (
                min(
                    (
                        row.detected_at.astimezone(UTC)
                        if row.detected_at is not None
                        else datetime.fromisoformat(f"{row.session}T00:00:00+00:00")
                    )
                    for row in item[1]
                ),
                min(row.opportunity_id for row in item[1]),
                item[0],
            ),
        )
        consumed_signals: set[str] = set()
        session_count = 0
        for _batch_id, batch_rows in ordered_batches:
            candidates = [
                AdmissionCandidate(
                    payload=row,
                    signal_id=row.signal_id or f"opportunity:{row.opportunity_id}",
                    score=row.decision_score,
                    # Production installs the conservative managed LCB as the
                    # suggestion EV before invoking the shared selector.
                    expected_value=row.expected_value_lcb,
                    defined_risk=row.decision_defined_risk,
                    max_loss=row.decision_max_loss,
                    account_value_available=row.decision_account_value_available,
                    account_value_usd=row.decision_account_value_usd,
                )
                for row in sorted(batch_rows, key=lambda item: item.opportunity_id)
            ]
            batch_count = 0
            for candidate in rank_admission_candidates(candidates, policy):
                if session_count >= policy.max_admitted_per_session:
                    break
                if batch_count >= policy.max_candidates_per_batch:
                    break
                if candidate.signal_id in consumed_signals:
                    continue
                consumed_signals.add(candidate.signal_id)
                admitted.append(candidate.payload)
                batch_count += 1
                session_count += 1
            if session_count >= policy.max_admitted_per_session:
                break
    return admitted


def _session_bootstrap_lcb(
    admitted: Sequence[WalkForwardRow], iterations: int, seed: int = 0x0D7E
) -> float:
    by_session: dict[str, list[float]] = {}
    for row in admitted:
        by_session.setdefault(row.session, []).append(row.realized_net_pnl)
    sessions = sorted(by_session)
    if not sessions:
        return float("-inf")
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(max(1, iterations)):
        sampled = [rng.choice(sessions) for _ in sessions]
        pnl = [value for session in sampled for value in by_session[session]]
        bootstrapped.append(sum(pnl) / len(pnl))
    return _quantile(bootstrapped, 0.025)


def evaluate_prospective_rows(
    rows: Sequence[WalkForwardRow],
    *,
    min_sessions: int,
    min_independent_signals: int,
    min_admitted: int,
    min_admitted_sessions: int,
    min_profit_factor: float,
    max_admitted_per_session: int,
    bootstrap_iterations: int,
    score_floor: float = 0.0,
    single_trade_cap_pct: float = 1.0,
    max_candidates_per_batch: int | None = None,
) -> ProspectiveReport:
    """Apply the operational promotion gate to one fixed future cohort."""
    if min_sessions < 1 or min_independent_signals < 1:
        raise ValueError("prospective evidence minimums must be positive")
    selection_policy = AdmissionSelectionPolicy(
        score_floor=score_floor,
        single_trade_cap_pct=single_trade_cap_pct,
        max_candidates_per_batch=(
            max_admitted_per_session
            if max_candidates_per_batch is None
            else max_candidates_per_batch
        ),
        max_admitted_per_session=max_admitted_per_session,
    )
    admitted = _selected_admissions(rows, policy=selection_policy)
    pnl = [row.realized_net_pnl for row in admitted]
    gross_wins = sum(value for value in pnl if value > 0.0)
    gross_losses = -sum(value for value in pnl if value < 0.0)
    profit_factor = (
        gross_wins / gross_losses
        if gross_losses > 0.0
        else float("inf")
        if gross_wins > 0.0
        else None
    )
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in admitted:
        equity += row.realized_net_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    sessions = len({row.session for row in rows})
    independent_signals = len(
        {row.signal_id or f"opportunity:{row.opportunity_id}" for row in rows}
    )
    admitted_sessions = len({row.session for row in admitted})
    lcb = _session_bootstrap_lcb(admitted, bootstrap_iterations)
    reasons: list[str] = []
    if sessions < min_sessions:
        reasons.append(f"prospective_sessions_below_minimum({sessions}<{min_sessions})")
    if independent_signals < min_independent_signals:
        reasons.append(
            f"prospective_signals_below_minimum({independent_signals}<{min_independent_signals})"
        )
    if len(admitted) < min_admitted:
        reasons.append(f"prospective_admitted_below_minimum({len(admitted)}<{min_admitted})")
    if admitted_sessions < min_admitted_sessions:
        reasons.append(
            "prospective_admitted_sessions_below_minimum("
            f"{admitted_sessions}<{min_admitted_sessions})"
        )
    if not lcb > 0.0:
        reasons.append("prospective_after_cost_mean_pnl_lcb_not_positive")
    if profit_factor is None or profit_factor < min_profit_factor:
        reasons.append("prospective_profit_factor_below_minimum")
    return ProspectiveReport(
        eligible=not reasons,
        samples=len(rows),
        independent_signals=independent_signals,
        sessions=sessions,
        admitted=len(admitted),
        admitted_sessions=admitted_sessions,
        admitted_net_pnl=sum(pnl),
        admitted_mean_pnl=sum(pnl) / len(pnl) if pnl else 0.0,
        admitted_mean_pnl_lcb=lcb,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        reasons=tuple(reasons),
        rows=tuple(rows),
    )


def _incremental_session_bootstrap_lcb(
    values: Mapping[str, Sequence[float]], iterations: int, seed: int = 0xC07E
) -> float:
    sessions = sorted(values)
    if not sessions:
        return float("-inf")
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(max(1, iterations)):
        sampled = [rng.choice(sessions) for _ in sessions]
        increments = [value for session in sampled for value in values[session]]
        bootstrapped.append(sum(increments) / len(increments))
    return _quantile(bootstrapped, 0.025)


def compare_context_incremental_value(
    baseline_rows: Sequence[WalkForwardRow],
    context_rows: Sequence[WalkForwardRow],
    *,
    min_disagreements: int = 20,
    min_sessions: int = 10,
    bootstrap_iterations: int = 2_000,
    max_admitted_per_session: int = 3,
    score_floor: float = 0.0,
    single_trade_cap_pct: float = 1.0,
    max_candidates_per_batch: int | None = None,
) -> ContextIncrementalReport:
    """Score Hermes only where it changes the executable admission policy.

    Agreement gets exactly zero credit.  A context-only admission contributes
    the realized P&L; a context veto of a baseline admission contributes the
    negative realized P&L.  Admission is collapsed to one structure per signal
    and the same causal batch/session capacity used by production before
    disagreements are counted. Rows must share immutable opportunity and
    chronological identity.
    """
    baseline = {row.opportunity_id: row for row in baseline_rows}
    context = {row.opportunity_id: row for row in context_rows}
    if baseline.keys() != context.keys():
        raise ValueError("baseline and context evaluations cover different opportunities")
    for opportunity_id, base in baseline.items():
        challenger = context[opportunity_id]
        if (
            base.session != challenger.session
            or base.outcome != challenger.outcome
            or base.signal_id != challenger.signal_id
            or base.decision_batch_id != challenger.decision_batch_id
            or base.detected_at != challenger.detected_at
            or base.decision_score != challenger.decision_score
            or base.decision_defined_risk != challenger.decision_defined_risk
            or base.decision_max_loss != challenger.decision_max_loss
            or base.decision_account_value_available
            != challenger.decision_account_value_available
            or base.decision_account_value_usd != challenger.decision_account_value_usd
        ):
            raise ValueError("baseline/context outcome identity mismatch")
    selection_policy = AdmissionSelectionPolicy(
        score_floor=score_floor,
        single_trade_cap_pct=single_trade_cap_pct,
        max_candidates_per_batch=(
            max_admitted_per_session
            if max_candidates_per_batch is None
            else max_candidates_per_batch
        ),
        max_admitted_per_session=max_admitted_per_session,
    )
    base_selected = _selected_admissions(baseline_rows, policy=selection_policy)
    context_selected = _selected_admissions(context_rows, policy=selection_policy)
    base_by_key = {
        (row.session, row.signal_id or f"opportunity:{row.opportunity_id}"): row
        for row in base_selected
    }
    context_by_key = {
        (row.session, row.signal_id or f"opportunity:{row.opportunity_id}"): row
        for row in context_selected
    }
    increments_by_session: dict[str, list[float]] = {}
    for key in sorted(base_by_key.keys() | context_by_key.keys()):
        selected_base = base_by_key.get(key)
        selected_challenger = context_by_key.get(key)
        if selected_base is not None and selected_challenger is not None:
            if selected_base.opportunity_id == selected_challenger.opportunity_id:
                continue
            increment = selected_challenger.realized_net_pnl - selected_base.realized_net_pnl
        elif selected_challenger is not None:
            increment = selected_challenger.realized_net_pnl
        elif selected_base is not None:
            increment = -selected_base.realized_net_pnl
        else:  # pragma: no cover - set union makes this unreachable
            continue
        increments_by_session.setdefault(key[0], []).append(increment)
    increments = [value for values in increments_by_session.values() for value in values]
    lcb = _incremental_session_bootstrap_lcb(increments_by_session, bootstrap_iterations)
    reasons: list[str] = []
    if len(increments) < min_disagreements:
        reasons.append(
            f"context_disagreements_below_minimum({len(increments)}<{min_disagreements})"
        )
    if len(increments_by_session) < min_sessions:
        reasons.append(
            "context_disagreement_sessions_below_minimum("
            f"{len(increments_by_session)}<{min_sessions})"
        )
    if not lcb > 0.0:
        reasons.append("context_incremental_value_lcb_not_positive")
    return ContextIncrementalReport(
        eligible=not reasons,
        disagreements=len(increments),
        disagreement_sessions=len(increments_by_session),
        incremental_net_pnl=sum(increments),
        mean_incremental_pnl=(sum(increments) / len(increments) if increments else 0.0),
        mean_incremental_pnl_lcb=lcb,
        reasons=tuple(reasons),
    )


def _row_weights(rows: Sequence[WalkForwardRow]) -> dict[int, float]:
    counts = Counter(
        (row.session, row.signal_id or f"opportunity:{row.opportunity_id}") for row in rows
    )
    return {
        row.opportunity_id: 1.0
        / counts[(row.session, row.signal_id or f"opportunity:{row.opportunity_id}")]
        for row in rows
    }


def _calibration_error(rows: Sequence[WalkForwardRow], buckets: int = 10) -> float:
    if not rows:
        return float("inf")
    weights = _row_weights(rows)
    total_weight = sum(weights.values())
    total_error = 0.0
    for class_index, outcome in enumerate(_OUTCOMES):
        groups: list[list[WalkForwardRow]] = [[] for _ in range(buckets)]
        for row in rows:
            probability = row.probabilities[class_index]
            index = min(buckets - 1, int(probability * buckets))
            groups[index].append(row)
        for group in groups:
            if not group:
                continue
            group_weight = sum(weights[row.opportunity_id] for row in group)
            predicted = (
                sum(row.probabilities[class_index] * weights[row.opportunity_id] for row in group)
                / group_weight
            )
            observed = (
                sum((row.outcome == outcome) * weights[row.opportunity_id] for row in group)
                / group_weight
            )
            total_error += group_weight / total_weight * abs(predicted - observed)
    return total_error / len(_OUTCOMES)


def _evaluate_rows(
    rows: Sequence[WalkForwardRow],
    baseline_probabilities: Mapping[str, tuple[float, float, float]],
    folds: int,
    policy: PromotionPolicy,
) -> WalkForwardMetrics:
    if not rows:
        return WalkForwardMetrics(
            samples=0,
            independent_signals=0,
            sessions=0,
            folds=folds,
            multiclass_brier=float("inf"),
            baseline_brier=float("inf"),
            log_loss=float("inf"),
            baseline_log_loss=float("inf"),
            calibration_error=float("inf"),
            admitted=0,
            admitted_sessions=0,
            admitted_net_pnl=0.0,
            admitted_mean_pnl=0.0,
            admitted_mean_pnl_lcb=float("-inf"),
            profit_factor=None,
            max_drawdown=0.0,
        )
    weights = _row_weights(rows)
    total_weight = sum(weights.values())
    brier = 0.0
    baseline_brier = 0.0
    log_loss = 0.0
    baseline_log_loss = 0.0
    for row in rows:
        row_weight = weights[row.opportunity_id]
        target_index = _OUTCOMES.index(row.outcome)
        baseline = baseline_probabilities[row.session]
        brier += row_weight * sum(
            (probability - (1.0 if index == target_index else 0.0)) ** 2
            for index, probability in enumerate(row.probabilities)
        )
        baseline_brier += row_weight * sum(
            (probability - (1.0 if index == target_index else 0.0)) ** 2
            for index, probability in enumerate(baseline)
        )
        log_loss -= row_weight * math.log(max(row.probabilities[target_index], _EPSILON))
        baseline_log_loss -= row_weight * math.log(max(baseline[target_index], _EPSILON))
    admitted = _selected_admissions(
        rows,
        policy=policy.admission_selection_policy(),
    )
    pnl = [row.realized_net_pnl for row in admitted]
    gross_wins = sum(value for value in pnl if value > 0.0)
    gross_losses = -sum(value for value in pnl if value < 0.0)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in admitted:
        equity += row.realized_net_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return WalkForwardMetrics(
        samples=len(rows),
        independent_signals=len(
            {row.signal_id or f"opportunity:{row.opportunity_id}" for row in rows}
        ),
        sessions=len({row.session for row in rows}),
        folds=folds,
        multiclass_brier=brier / total_weight,
        baseline_brier=baseline_brier / total_weight,
        log_loss=log_loss / total_weight,
        baseline_log_loss=baseline_log_loss / total_weight,
        calibration_error=_calibration_error(rows),
        admitted=len(admitted),
        admitted_sessions=len({row.session for row in admitted}),
        admitted_net_pnl=sum(pnl),
        admitted_mean_pnl=sum(pnl) / len(pnl) if pnl else 0.0,
        admitted_mean_pnl_lcb=_session_bootstrap_lcb(admitted, policy.bootstrap_iterations),
        profit_factor=(
            gross_wins / gross_losses
            if gross_losses > 0.0
            else float("inf")
            if gross_wins > 0.0
            else None
        ),
        max_drawdown=max_drawdown,
    )


def walk_forward_evaluate(
    rows: Sequence[ManagedSample],
    *,
    policy: PromotionPolicy | None = None,
    min_train_sessions: int = 15,
    embargo_sessions: int = 1,
    feature_schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION,
    outcome_policy_version: str = DEFAULT_OUTCOME_POLICY_VERSION,
) -> PromotionReport:
    """Expanding-window evaluation grouped by full session with an embargo."""
    policy = policy or PromotionPolicy()
    if min_train_sessions < 3 or embargo_sessions < 0:
        raise ValueError("invalid walk-forward session settings")
    canonical: dict[str, ManagedSample] = {}
    duplicate_keys: set[str] = set()
    for row in rows:
        if row.opportunity_key in canonical:
            duplicate_keys.add(row.opportunity_key)
        else:
            canonical[row.opportunity_key] = row
    if duplicate_keys:
        raise ValueError(
            "walk-forward input contains duplicate opportunity keys: "
            + ", ".join(sorted(duplicate_keys)[:3])
        )
    ordered = sorted(canonical.values(), key=lambda row: (row.session, row.opportunity_id))
    sessions = sorted({row.session for row in ordered})
    oof: list[WalkForwardRow] = []
    baseline_by_session: dict[str, tuple[float, float, float]] = {}
    folds = 0
    residual_history: list[tuple[str, float, str]] = []
    for test_index in range(min_train_sessions + embargo_sessions, len(sessions)):
        test_session = sessions[test_index]
        train_end = test_index - embargo_sessions
        train_sessions = set(sessions[:train_end])
        train = [row for row in ordered if row.session in train_sessions]
        test = [row for row in ordered if row.session == test_session]
        missing_outcome = any(
            not any(row.outcome == outcome for row in train) for outcome in _OUTCOMES
        )
        if not test or missing_outcome:
            continue
        try:
            eligible_residuals = [
                (value, group)
                for session, value, group in residual_history
                if session in train_sessions
            ]
            artifact = fit_managed_model(
                train,
                model_version=f"walk-forward-{test_session}",
                feature_schema_version=feature_schema_version,
                outcome_policy_version=outcome_policy_version,
                ev_residuals=[value for value, _ in eligible_residuals],
                ev_residual_groups=[group for _, group in eligible_residuals],
            )
        except ValueError:
            continue
        train_groups = Counter(row.signal_id or row.opportunity_key for row in train)
        counts = [
            sum(
                1.0 / train_groups[row.signal_id or row.opportunity_key]
                for row in train
                if row.outcome == outcome
            )
            for outcome in _OUTCOMES
        ]
        count_total = sum(counts)
        baseline_by_session[test_session] = tuple(count / count_total for count in counts)  # type: ignore[assignment]
        fold_rows: list[WalkForwardRow] = []
        for sample in test:
            probabilities = _raw_prediction(artifact, sample.features)
            expected = _point_expected_value(
                probabilities, sample, artifact.timeout_expected_return
            )
            expected_lcb = (
                expected + artifact.ev_residual_q05 * sample.basis_dollars
                if artifact.ev_residual_q05 is not None
                else float("-inf")
            )
            fold_rows.append(
                WalkForwardRow(
                    opportunity_id=sample.opportunity_id,
                    session=sample.session,
                    outcome=sample.outcome,
                    probabilities=probabilities,
                    expected_value=expected,
                    expected_value_lcb=expected_lcb,
                    realized_net_pnl=sample.realized_net_pnl,
                    signal_id=sample.signal_id,
                    decision_batch_id=sample.decision_batch_id,
                    detected_at=sample.detected_at,
                    decision_score=sample.decision_score,
                    decision_defined_risk=sample.decision_defined_risk,
                    decision_max_loss=sample.decision_max_loss,
                    decision_account_value_available=(
                        sample.decision_account_value_available
                    ),
                    decision_account_value_usd=sample.decision_account_value_usd,
                )
            )
            residual = (sample.realized_net_pnl - expected) / sample.basis_dollars
            residual_history.append((sample.session, residual, _sample_group(sample)))
        oof.extend(fold_rows)
        folds += 1
    metrics = _evaluate_rows(oof, baseline_by_session, folds, policy)
    reasons: list[str] = []
    if len(sessions) < policy.min_sessions:
        reasons.append(f"sessions_below_minimum({len(sessions)}<{policy.min_sessions})")
    if len(ordered) < policy.min_samples:
        reasons.append(f"samples_below_minimum({len(ordered)}<{policy.min_samples})")
    independent_signals = {row.signal_id or row.opportunity_key for row in ordered}
    if len(independent_signals) < policy.min_independent_signals:
        reasons.append(
            "independent_signals_below_minimum("
            f"{len(independent_signals)}<{policy.min_independent_signals})"
        )
    if metrics.samples < policy.min_oof_samples:
        reasons.append(f"oof_samples_below_minimum({metrics.samples}<{policy.min_oof_samples})")
    if metrics.folds < policy.min_folds:
        reasons.append(f"folds_below_minimum({metrics.folds}<{policy.min_folds})")
    if not metrics.multiclass_brier < metrics.baseline_brier:
        reasons.append("brier_not_better_than_training_base_rate")
    if not metrics.log_loss < metrics.baseline_log_loss:
        reasons.append("log_loss_not_better_than_training_base_rate")
    if metrics.admitted < policy.min_admitted:
        reasons.append(f"admitted_below_minimum({metrics.admitted}<{policy.min_admitted})")
    if metrics.admitted_sessions < policy.min_admitted_sessions:
        reasons.append(
            "admitted_sessions_below_minimum("
            f"{metrics.admitted_sessions}<{policy.min_admitted_sessions})"
        )
    if not metrics.admitted_mean_pnl_lcb > 0.0:
        reasons.append("after_cost_mean_pnl_lcb_not_positive")
    if metrics.profit_factor is None or metrics.profit_factor < policy.min_profit_factor:
        reasons.append("profit_factor_below_minimum")
    return PromotionReport(
        eligible=not reasons,
        reasons=tuple(reasons),
        metrics=metrics,
        rows=tuple(oof),
    )
