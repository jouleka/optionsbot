"""Canonical, leakage-safe features for managed-path learning and inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from optionsbot.managed_contract import MANAGED_FEATURE_SCHEMA_VERSION

_CATEGORICAL_FIELDS = frozenset(
    {
        "direction",
        "direction_strength",
        "failed_side",
        "generator",
        "iv_regime",
        "parameter_version",
        "setup_type",
        "strategy",
        "regime_dir",
        "regime_iv",
        "structure_kind",
    }
)

# Only measurements known before a trade can enter the model. Terminal PoP/EV,
# prior model outputs, sizing, reviews, and realized fields are omitted so a
# promoted model can never become one of its own inputs on the next session.
_SAFE_SUGGESTION_FIELDS = frozenset(
    {
        "credit_or_debit",
        "max_loss",
        "max_profit",
        "reward_risk",
        "estimated_round_trip_cost",
        "managed_marketable_entry_net",
        "managed_marketable_basis_dollars",
        "managed_commission_estimate",
        "structure_leg_count",
        "structure_width",
        "structure_net_delta",
        "structure_net_gamma",
        "structure_net_theta",
        "structure_net_vega",
        "structure_friction_fraction",
        "structure_desired_premium_target_dollars",
        "structure_target_scenario_pnl_dollars",
        "structure_invalidation_scenario_pnl_dollars",
        "structure_timeout_scenario_pnl_dollars",
        "thesis_entry_spot",
        "thesis_invalidation_spot",
        "thesis_target_spot",
        "thesis_underlying_risk_fraction",
        "thesis_underlying_reward_risk",
        "thesis_timeout_minutes",
        "premium_target_feasible",
        # Row-local managed plans are recursively reduced to finite numeric,
        # boolean, and explicitly categorical causal fields. Identity,
        # timestamps, authority/status text, and outcomes cannot pass through.
        "managed_signal_plan",
    }
)

_SAFE_RAW_FIELDS = frozenset(
    {
        "delayed",
        "n_chain_legs",
        "warming_up",
        "iv_rank_is_proxy",
        "earnings_in_window",
        "relative_strength",
        "beta_to_benchmark",
        "front_dte",
        "expected_move",
    }
)

_SAFE_MARKET_VIEW_FIELDS = frozenset(
    {
        "direction",
        "direction_strength",
        "iv_regime",
        "earnings_in_window",
        "warming_up",
        "iv_rank_is_proxy",
    }
)
_SAFE_OR_PLAN_FIELDS = frozenset(
    {
        "timeframe_minutes",
        "direction",
        "opening_range_high",
        "opening_range_low",
        "fvg_low",
        "fvg_high",
        "entry_underlying_price",
        "stop_pct",
        "target_r",
        "target_pct",
        "setup_type",
    }
)
_SAFE_OR_QUALITY_FIELDS: dict[str, frozenset[str]] = {
    "opening_range": frozenset(
        {
            "width",
            "midpoint",
            "width_pct",
            "atr_14",
            "width_atr_ratio",
            "width_atr_normalized",
        }
    ),
    "breakout": frozenset(
        {
            "displacement",
            "displacement_atr",
            "displacement_or_ratio",
            "displacement_normalized",
            "candle_range",
            "body_fraction",
            "directional_body_fraction",
            "rejection_wick_fraction",
            "directional_close_location",
            "volume",
            "relative_volume",
            "relative_volume_normalized",
        }
    ),
    "gap": frozenset(
        {
            "size",
            "size_or_ratio",
            "size_atr_ratio",
            "size_normalized",
            "formation_lag_bars",
        }
    ),
    "retest": frozenset(
        {
            "depth_fraction",
            "rejection_fraction",
            "body_fraction",
            "directional_body_fraction",
            "directional_close_location",
            "volume",
            "relative_volume",
            "relative_volume_normalized",
            "lag_bars",
        }
    ),
    "vwap": frozenset(
        {
            "value",
            "directional_distance_pct",
            "directional_distance_or_ratio",
            "directional_distance_normalized",
            "direction_aligned",
        }
    ),
    "timing": frozenset(
        {
            "breakout_minutes_from_open",
            "breakout_time_fraction",
            "confirmation_minutes_from_open",
            "confirmation_time_fraction",
            "confirmation_age_minutes",
            "confirmation_age_bars",
            "freshness_normalized",
            "entry_window_remaining_fraction",
        }
    ),
    "regime": frozenset(
        {
            "direction_aligned",
            "direction_opposed",
            "direction_neutral",
            "strong_trend",
            "iv_low",
            "iv_neutral",
            "iv_high",
            "earnings_in_window",
            "iv_warming_up",
            "iv_rank_is_proxy",
        }
    ),
}

_SAFE_MANAGED_PLAN_FIELDS = frozenset(
    {
        "direction",
        "generator",
        "setup_type",
        "stop_pct",
        "target_r",
        "target_pct",
        "thesis_entry_spot",
        "thesis_invalidation_spot",
        "thesis_target_spot",
    }
)
_SAFE_HYPOTHESIS_FIELDS = frozenset(
    {
        "direction",
        "generator",
        "reference_price",
        "invalidation_level",
    }
)
_SAFE_CAUSAL_WINDOW_FIELDS = frozenset({"bar_count"})
_SAFE_MOMENTUM_FIELDS = frozenset(
    {
        "open_price",
        "close_price",
        "high_price",
        "low_price",
        "return_pct",
        "directional_return_pct",
        "range_pct",
        "atr_14",
        "directional_return_atr_ratio",
        "directional_return_atr_normalized",
        "directional_efficiency",
        "directional_close_location",
        "vwap",
        "directional_vwap_distance_pct",
        "vwap_direction_aligned",
        "total_volume",
        "mean_volume",
        "relative_volume",
        "relative_volume_normalized",
    }
)
_SAFE_HYPOTHESIS_FEATURE_FIELDS = frozenset(
    {
        # Opening momentum.
        "opening_window_minutes",
        "thesis_lifetime_minutes",
        "second_half_volume_ratio",
        "second_half_volume_normalized",
        # Failed opening-range breakout.
        "opening_range_high",
        "opening_range_low",
        "opening_range_width",
        "opening_range_atr_ratio",
        "opening_range_minutes",
        "failed_side",
        "breakout_close",
        "breakout_extreme",
        "breakout_close_displacement_or_ratio",
        "breakout_extreme_excursion_or_ratio",
        "breakout_rejection_wick_fraction",
        "breakout_volume",
        "breakout_relative_volume",
        "breakout_relative_volume_normalized",
        "reentry_close",
        "bars_to_reentry",
        "max_reentry_bars",
        "reentry_depth_fraction",
        "reentry_directional_body_fraction",
        "reentry_vwap",
        "reentry_vwap_direction_aligned",
        # Late-session momentum.
        "minutes_from_open",
        "minutes_to_close",
        "window_minutes",
        "parameter_version",
    }
)


def _finite(value: object) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return None
    return value


def _safe_mapping(
    payload: object,
    *,
    allowed: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_name, value in payload.items():
        name = str(raw_name)
        if allowed is not None and name not in allowed:
            continue
        # Every field accepted here is scalar by contract. Nested structures
        # are admitted only by their dedicated closed-schema projectors below;
        # recursively accepting a mapping under a scalar name would let shape
        # drift smuggle arbitrary numeric outcome fields into training.
        if isinstance(value, bool | int | float):
            finite = _finite(value)
            if finite is not None:
                result[name] = finite
        elif value is None:
            result[name] = None
        elif isinstance(value, str) and name in _CATEGORICAL_FIELDS:
            result[name] = value
    return result


def _safe_managed_signal_plan(payload: object) -> dict[str, Any]:
    """Project a row plan onto the versioned, causal hypothesis schema.

    This intentionally does not use unrestricted recursive traversal. A future
    producer adding realized PnL, labels, model output, identifiers, or time
    fields cannot silently turn them into training inputs.
    """
    if not isinstance(payload, Mapping):
        return {}
    result = _safe_mapping(payload, allowed=_SAFE_MANAGED_PLAN_FIELDS)
    hypothesis = payload.get("hypothesis")
    if not isinstance(hypothesis, Mapping):
        return result
    safe_hypothesis = _safe_mapping(
        hypothesis,
        allowed=_SAFE_HYPOTHESIS_FIELDS,
    )
    features = hypothesis.get("features")
    if isinstance(features, Mapping):
        safe_features = _safe_mapping(
            features,
            allowed=_SAFE_HYPOTHESIS_FEATURE_FIELDS,
        )
        causal_window = features.get("causal_window")
        if isinstance(causal_window, Mapping):
            safe_features["causal_window"] = _safe_mapping(
                causal_window,
                allowed=_SAFE_CAUSAL_WINDOW_FIELDS,
            )
        momentum = features.get("momentum")
        if isinstance(momentum, Mapping):
            safe_features["momentum"] = _safe_mapping(
                momentum,
                allowed=_SAFE_MOMENTUM_FIELDS,
            )
        safe_hypothesis["features"] = safe_features
    result["hypothesis"] = safe_hypothesis
    return result


def _safe_opening_range_plan(payload: object) -> dict[str, Any]:
    """Project the OR plan and quality vector onto a closed causal schema."""
    if not isinstance(payload, Mapping):
        return {}
    result = _safe_mapping(payload, allowed=_SAFE_OR_PLAN_FIELDS)
    quality = payload.get("quality")
    if not isinstance(quality, Mapping):
        return result
    safe_quality: dict[str, Any] = {}
    for section, fields in _SAFE_OR_QUALITY_FIELDS.items():
        section_payload = quality.get(section)
        if isinstance(section_payload, Mapping):
            safe_quality[section] = _safe_mapping(section_payload, allowed=fields)
    result["quality"] = safe_quality
    return result


def build_capture_feature_payload(
    *,
    feature_schema_version: str,
    snapshot_id: int | None,
    spot: object,
    iv_rank: object,
    hv20: object,
    iv_hv_ratio: object,
    expected_move: object,
    regime_dir: object,
    regime_iv: object,
    raw_snapshot: object,
    score: object,
    suggestion: object,
    registered_before_entry_cutoff: bool,
) -> dict[str, Any]:
    """Build the exact point-in-time payload shared by capture and live use."""
    if feature_schema_version != MANAGED_FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported managed feature schema; expected {MANAGED_FEATURE_SCHEMA_VERSION!r}"
        )
    safe_suggestion = _safe_mapping(
        suggestion,
        allowed=_SAFE_SUGGESTION_FIELDS,
    )
    if isinstance(suggestion, Mapping) and "managed_signal_plan" in suggestion:
        safe_suggestion["managed_signal_plan"] = _safe_managed_signal_plan(
            suggestion.get("managed_signal_plan")
        )
    safe_raw = _safe_mapping(raw_snapshot, allowed=_SAFE_RAW_FIELDS)
    if isinstance(raw_snapshot, Mapping):
        for view_name in ("inferred_market_view", "configured_market_view"):
            if view_name in raw_snapshot:
                safe_raw[view_name] = _safe_mapping(
                    raw_snapshot.get(view_name),
                    allowed=_SAFE_MARKET_VIEW_FIELDS,
                )
        if "opening_range_fvg" in raw_snapshot:
            safe_raw["opening_range_fvg"] = _safe_opening_range_plan(
                raw_snapshot.get("opening_range_fvg")
            )
    return {
        "feature_schema_version": feature_schema_version,
        # Retained for audit identity, then explicitly excluded by flattening.
        "snapshot_id": snapshot_id,
        "snapshot": {
            "spot": _finite(spot),
            "iv_rank": _finite(iv_rank),
            "hv20": _finite(hv20),
            "iv_hv_ratio": _finite(iv_hv_ratio),
            "expected_move": _finite(expected_move),
            "regime_dir": regime_dir if isinstance(regime_dir, str) else None,
            "regime_iv": regime_iv if isinstance(regime_iv, str) else None,
            "raw": safe_raw,
        },
        "score": _finite(score),
        "suggestion": safe_suggestion,
        "registered_before_entry_cutoff": registered_before_entry_cutoff,
    }


def flatten_capture_features(
    payload: Mapping[str, object],
    *,
    prefix: str = "",
) -> dict[str, float | None]:
    """Flatten numeric/categorical features while excluding identities."""
    result: dict[str, float | None] = {}
    for raw_name, value in payload.items():
        if raw_name == "id" or raw_name.endswith("_id"):
            continue
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)
        if isinstance(value, Mapping):
            result.update(flatten_capture_features(value, prefix=name))
        elif isinstance(value, bool):
            result[name] = 1.0 if value else 0.0
        elif isinstance(value, int | float) and math.isfinite(float(value)):
            result[name] = float(value)
        elif value is None:
            result[name] = None
        elif isinstance(value, str) and raw_name in _CATEGORICAL_FIELDS:
            result[f"{name}={value}"] = 1.0
    return result


def model_features(
    payload: Mapping[str, object],
    *,
    basis_dollars: float,
    stop_pct: float,
    target_pct: float,
    commission_estimate: float,
    direction: str,
    setup_type: str,
    strategy: str,
    context: Mapping[str, float | None] | None = None,
) -> dict[str, float | None]:
    """Add decision economics and identities to one canonical feature row."""
    if not math.isfinite(basis_dollars) or basis_dollars <= 0.0:
        raise ValueError("managed basis must be finite and positive")
    if not math.isfinite(commission_estimate) or commission_estimate < 0.0:
        raise ValueError("managed commission must be finite and non-negative")
    features = flatten_capture_features(payload)
    strategy_category = (
        ":".join(strategy.split(":", 2)[:2]) if strategy.startswith("shadow_grid_v1:") else strategy
    )
    features.update(
        {
            "economics.basis_dollars": basis_dollars,
            "economics.stop_pct": stop_pct,
            "economics.target_pct": target_pct,
            "economics.commission_fraction": commission_estimate / basis_dollars,
            f"direction={direction}": 1.0,
            f"setup_type={setup_type}": 1.0,
            f"strategy={strategy_category}": 1.0,
        }
    )
    if context is not None:
        features.update(context)
    return features
