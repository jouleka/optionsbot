from __future__ import annotations

import pytest

from optionsbot.learning.features import (
    build_capture_feature_payload,
    flatten_capture_features,
    model_features,
)
from optionsbot.managed_contract import MANAGED_FEATURE_SCHEMA_VERSION


def test_feature_payload_rejects_an_unimplemented_schema_identity() -> None:
    with pytest.raises(ValueError, match="unsupported managed feature schema"):
        build_capture_feature_payload(
            feature_schema_version="managed_capture_features_v2",
            snapshot_id=1,
            spot=100.0,
            iv_rank=0.4,
            hv20=0.2,
            iv_hv_ratio=1.1,
            expected_move=2.0,
            regime_dir="bull",
            regime_iv="neutral",
            raw_snapshot={},
            score=50.0,
            suggestion={},
            registered_before_entry_cutoff=True,
        )


def test_feature_payload_excludes_predictions_reviews_and_history() -> None:
    payload = build_capture_feature_payload(
        feature_schema_version=MANAGED_FEATURE_SCHEMA_VERSION,
        snapshot_id=42,
        spot=100.0,
        iv_rank=0.4,
        hv20=0.2,
        iv_hv_ratio=1.1,
        expected_move=2.0,
        regime_dir="bull",
        regime_iv="neutral",
        raw_snapshot={
            "relative_strength": 0.03,
            "recent_price_history": [{"close": 99.0}],
            "opening_range_fvg": {
                "direction": "bull",
                "signal_id": "not-a-feature",
                "quality": {
                    "breakout": {
                        "displacement_atr": 1.2,
                        "future_return": 9.0,
                    },
                    "outcome": {"realized_pnl": 999.0},
                },
                "realized_pnl": 999.0,
            },
        },
        score=77.0,
        suggestion={
            "credit_or_debit": -100.0,
            "max_loss": 100.0,
            "expected_value": 999.0,
            "managed_target_hit_probability": 0.99,
            "review_evidence": {"ready": True},
        },
        registered_before_entry_cutoff=True,
    )
    flat = flatten_capture_features(payload)
    assert "snapshot_id" not in flat
    assert flat["snapshot.raw.relative_strength"] == 0.03
    assert flat["snapshot.raw.opening_range_fvg.quality.breakout.displacement_atr"] == 1.2
    assert not any("future_return" in name for name in flat)
    assert not any("realized_pnl" in name for name in flat)
    assert "snapshot.raw.recent_price_history" not in flat
    assert flat["suggestion.credit_or_debit"] == -100.0
    assert "suggestion.expected_value" not in flat
    assert "suggestion.managed_target_hit_probability" not in flat
    assert "suggestion.review_evidence.ready" not in flat


def test_allowlisted_scalar_fields_reject_nested_mapping_leakage() -> None:
    payload = build_capture_feature_payload(
        feature_schema_version=MANAGED_FEATURE_SCHEMA_VERSION,
        snapshot_id=42,
        spot=100.0,
        iv_rank=0.4,
        hv20=0.2,
        iv_hv_ratio=1.1,
        expected_move=2.0,
        regime_dir="bull",
        regime_iv="neutral",
        raw_snapshot={
            "delayed": {"realized_pnl": 777.0},
            "relative_strength": 0.03,
        },
        score=77.0,
        suggestion={
            "credit_or_debit": {"realized_pnl": 999.0},
            "max_loss": 100.0,
        },
        registered_before_entry_cutoff=True,
    )

    flat = flatten_capture_features(payload)

    assert flat["snapshot.raw.relative_strength"] == 0.03
    assert flat["suggestion.max_loss"] == 100.0
    assert not any("delayed" in name for name in flat)
    assert not any("credit_or_debit" in name for name in flat)
    assert not any("realized_pnl" in name for name in flat)


def test_model_features_add_only_current_economics_and_low_cardinality_identity() -> None:
    features = model_features(
        {"quality": {"rvol": 1.5}},
        basis_dollars=125.0,
        stop_pct=0.15,
        target_pct=0.30,
        commission_estimate=2.8,
        direction="bear",
        setup_type="failed_breakout",
        strategy="long_put",
    )
    assert features["quality.rvol"] == 1.5
    assert features["economics.commission_fraction"] == 2.8 / 125.0
    assert features["direction=bear"] == 1.0
    assert features["setup_type=failed_breakout"] == 1.0
    assert features["strategy=long_put"] == 1.0


def test_shadow_hypothesis_features_are_causal_bounded_and_leakage_safe() -> None:
    payload = build_capture_feature_payload(
        feature_schema_version=MANAGED_FEATURE_SCHEMA_VERSION,
        snapshot_id=9,
        spot=501.0,
        iv_rank=0.45,
        hv20=0.22,
        iv_hv_ratio=1.2,
        expected_move=4.0,
        regime_dir="bear",
        regime_iv="neutral",
        raw_snapshot={},
        score=0.0,
        suggestion={
            "managed_signal_plan": {
                "signal_id": "must-not-enter-features",
                "signal_at": "2026-08-28T14:05:00+00:00",
                "authority": "shadow_research_only_no_order_or_halt_authority",
                "direction": "bear",
                "generator": "failed_breakout_reversal",
                "setup_type": "failed_breakout_reversal",
                "stop_pct": 0.15,
                "target_r": 1.5,
                "target_pct": 0.225,
                "thesis_entry_spot": 501.0,
                "thesis_invalidation_spot": 503.0,
                "thesis_target_spot": 498.0,
                # Deliberate future-schema leakage attempts.
                "realized_pnl": 999.0,
                "model_probability": 0.99,
                "hypothesis": {
                    "hypothesis_id": "also-must-not-enter",
                    "observed_at": "2026-08-28T14:05:00+00:00",
                    "direction": "bear",
                    "generator": "failed_breakout_reversal",
                    "reference_price": 501.0,
                    "invalidation_level": 503.0,
                    "win": 1,
                    "features": {
                        "causal_window": {
                            "bar_count": 31,
                            "end_at": "2026-08-28T14:05:00+00:00",
                            "future_return": 0.8,
                        },
                        "opening_range_width": 2.0,
                        "breakout_relative_volume": 1.8,
                        "breakout_rejection_wick_fraction": 0.42,
                        "failed_side": "high",
                        "parameter_version": "intraday_shadow_windows_v1",
                        "realized_pnl": 999.0,
                        "outcome": {"target_hit": 1},
                    },
                },
            }
        },
        registered_before_entry_cutoff=True,
    )
    flat = flatten_capture_features(payload)

    prefix = "suggestion.managed_signal_plan"
    assert flat[f"{prefix}.generator=failed_breakout_reversal"] == 1.0
    assert flat[f"{prefix}.hypothesis.features.opening_range_width"] == 2.0
    assert flat[f"{prefix}.hypothesis.features.breakout_relative_volume"] == 1.8
    assert flat[f"{prefix}.hypothesis.features.causal_window.bar_count"] == 31.0
    assert flat[f"{prefix}.hypothesis.features.failed_side=high"] == 1.0
    assert flat[f"{prefix}.hypothesis.features.parameter_version=intraday_shadow_windows_v1"] == 1.0
    forbidden_fragments = (
        "signal_id",
        "hypothesis_id",
        "signal_at",
        "observed_at",
        "authority",
        "realized_pnl",
        "model_probability",
        "future_return",
        "outcome",
        ".win",
    )
    leaked = [name for name in flat if any(fragment in name for fragment in forbidden_fragments)]
    assert leaked == []
