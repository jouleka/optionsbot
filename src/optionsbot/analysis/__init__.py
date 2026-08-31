"""Analysis layer: pure functions over IBKR-layer outputs."""

from optionsbot.analysis.events import earnings_within, next_earnings
from optionsbot.analysis.intraday_hypotheses import (
    HYPOTHESIS_GENERATOR_REGISTRY,
    HypothesisResearchConfig,
    ShadowIntradayHypothesis,
    generate_failed_breakout_reversals,
    generate_late_session_momentum,
    generate_opening_momentum_continuation,
    generate_shadow_hypotheses,
)
from optionsbot.analysis.iv_rank import iv_rank
from optionsbot.analysis.opening_range_quality import (
    OpeningRangeQualityFeatures,
    build_opening_range_quality_features,
    quality_payload_with_regime,
)
from optionsbot.analysis.technicals import adx, sma, trend_regime
from optionsbot.analysis.types import (
    EarningsInfo,
    IVRankResult,
    IVRegime,
    MarketView,
    TrendRegime,
)
from optionsbot.analysis.view import infer_view
from optionsbot.analysis.volatility import (
    expected_move,
    historical_volatility,
    iv_hv_ratio,
)

__all__ = [
    "EarningsInfo",
    "IVRankResult",
    "IVRegime",
    "HYPOTHESIS_GENERATOR_REGISTRY",
    "HypothesisResearchConfig",
    "MarketView",
    "OpeningRangeQualityFeatures",
    "TrendRegime",
    "ShadowIntradayHypothesis",
    "adx",
    "build_opening_range_quality_features",
    "earnings_within",
    "expected_move",
    "historical_volatility",
    "infer_view",
    "iv_hv_ratio",
    "iv_rank",
    "next_earnings",
    "generate_failed_breakout_reversals",
    "generate_late_session_momentum",
    "generate_opening_momentum_continuation",
    "generate_shadow_hypotheses",
    "quality_payload_with_regime",
    "sma",
    "trend_regime",
]
