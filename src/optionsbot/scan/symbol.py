"""End-to-end scan for a single symbol: IBKR fetch -> analysis -> scoring -> persist."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import cast

import pandas as pd
from sqlalchemy import Engine, insert

from optionsbot.analysis.beta_weighting import beta
from optionsbot.analysis.events import next_earnings
from optionsbot.analysis.intraday_hypotheses import ShadowIntradayHypothesis
from optionsbot.analysis.news import (
    news_cache_is_stale,
    refresh_news_if_stale,
    replace_news,
)
from optionsbot.analysis.opening_range_fvg import OpeningRangeFVGSignal
from optionsbot.analysis.opening_range_quality import quality_payload_with_regime
from optionsbot.analysis.relative_strength import relative_strength
from optionsbot.analysis.structure_optimizer import (
    ShadowStructureCandidate,
    UnderlyingThesis,
    build_shadow_grid_for_thesis,
    build_shadow_structure_grid,
)
from optionsbot.analysis.types import Direction, EarningsInfo, IVRegime, MarketView
from optionsbot.analysis.view import infer_view
from optionsbot.analysis.volatility import historical_volatility, iv_hv_ratio
from optionsbot.config import Settings
from optionsbot.ibkr import (
    ChainClient,
    HistoryClient,
    IBKRClient,
    MarketDataClient,
    NewsClient,
    PositionsClient,
)
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.market_hours import minutes_to_nyse_close, nyse_session_close_utc
from optionsbot.opening_range_economics import (
    estimated_round_trip_cost,
    managed_break_even_probability,
    managed_expected_value,
    with_managed_expected_value,
)
from optionsbot.scan.types import ScanResult
from optionsbot.scoring import score_all
from optionsbot.storage.iv_history import read_atm_iv_history, record_atm_iv
from optionsbot.storage.schema import snapshots as snapshots_t
from optionsbot.storage.schema import strategy_scores as scores_t
from optionsbot.strategies import Leg, StrategySnapshot, StrategySuggestion, get_strategy
from optionsbot.strategies.strikes import closest_expiry_to_dte

log = logging.getLogger(__name__)

_SHADOW_GRID_PREFIX = "shadow_grid_v1"
_SHADOW_GRID_HOLD_REASON = "shadow_structure_pending_promoted_base_model"
_MAX_SHADOW_HYPOTHESES_PER_SNAPSHOT = 3
_MAX_GRID_STRUCTURES_PER_PLAN = 2
_MAX_SHADOW_ROWS_PER_SNAPSHOT = 8


async def _bounded_to_thread[T](
    fn: Callable[..., T], *args: object, timeout: float, default: T, label: str
) -> T:
    """Run a blocking ``fn`` on a worker thread, bounded by ``timeout`` seconds.

    Returns ``default`` (and logs) on timeout or error. This is how the scan
    calls yfinance (earnings/news): a hung or failing Yahoo response neither
    blocks the asyncio event loop (the call runs off-loop) nor stalls the scan
    (it's time-bounded) -- it just degrades to the fallback value (IBK-149).
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout)
    except TimeoutError:
        log.warning("%s timed out after %.1fs; using fallback", label, timeout)
        return default
    except Exception:  # noqa: BLE001 -- external deps fail variously; degrade
        log.exception("%s failed; using fallback", label)
        return default


def _atm_iv(chain: list[OptionChainLeg], spot: float) -> float | None:
    """Pick the IV of the ATM call leg with nearest expiry. Returns None if missing."""
    if not chain:
        return None
    by_expiry: dict[str, list[OptionChainLeg]] = {}
    for leg in chain:
        by_expiry.setdefault(leg.expiry, []).append(leg)
    nearest_expiry = min(by_expiry.keys())
    near_legs = by_expiry[nearest_expiry]
    near_legs.sort(key=lambda leg: (abs(leg.strike - spot), 0 if leg.right == "C" else 1))
    for leg in near_legs:
        if leg.iv is not None:
            return leg.iv
    return None


def _override_view(
    view: MarketView,
    override: tuple[Direction | None, IVRegime | None] | None,
) -> MarketView:
    if override is None:
        return view
    direction, iv_regime = override
    kwargs: dict[str, object] = {}
    if direction is not None:
        kwargs["direction"] = direction
    if iv_regime is not None:
        kwargs["iv_regime"] = iv_regime
    # kwargs is dict[str, object] for partial-replace flexibility; the values
    # are runtime-narrowed to Direction|IVRegime by the None-guards above.
    return replace(view, **kwargs) if kwargs else view  # type: ignore[arg-type]


def _serialize_legs(legs: tuple[Leg, ...]) -> list[dict[str, object]]:
    return [asdict(leg) for leg in legs]


def _shadow_strategy_identity(
    candidate: ShadowStructureCandidate,
    managed_plan: dict[str, object],
) -> str:
    """Ledger-unique identity; model features strip the immutable hash suffix."""
    generator = managed_plan.get("generator")
    if generator == "opening_range_fvg":
        # Preserve the original OR/FVG identity so an upgrade cannot duplicate
        # an already-captured alternative for the same frozen legs.
        suffix = candidate.candidate_id
    else:
        signal_id = str(managed_plan.get("signal_id", ""))
        signal_hash = hashlib.sha256(signal_id.encode("utf-8")).hexdigest()[:12]
        suffix = f"{signal_hash}:{candidate.candidate_id}"
    return f"{_SHADOW_GRID_PREFIX}:{candidate.strategy}:{suffix}"


def _shadow_structure_score_row(
    candidate: ShadowStructureCandidate,
    managed_plan: dict[str, object],
    settings: Settings,
    *,
    opening_range_plan: dict[str, object] | None = None,
) -> dict[str, object]:
    """Serialize one optimizer alternative without creating an alert candidate."""
    leg_count = len(candidate.legs)
    commissions = (
        leg_count
        * 2.0
        * settings.execution.opening_range_commission_per_contract
    )
    strikes = [float(leg.strike) for leg in candidate.legs if leg.strike is not None]
    width = max(strikes) - min(strikes) if len(strikes) > 1 else None
    reward_risk = (
        candidate.maximum_profit_dollars / candidate.maximum_loss_dollars
        if candidate.maximum_profit_dollars is not None
        and candidate.maximum_loss_dollars > 0.0
        else None
    )
    features = candidate.features
    suggestion: dict[str, object] = {
        "defined_risk": True,
        # StrategySuggestion and the managed ledger use credit-positive signed
        # dollars; the optimizer exposes a positive marketable debit.
        "credit_or_debit": -candidate.entry_debit_dollars,
        "max_loss": candidate.maximum_loss_dollars,
        "max_profit": candidate.maximum_profit_dollars,
        "prob_profit": None,
        "suggested_quantity": 0,
        "reward_risk": reward_risk,
        "expected_value": None,
        "expected_value_model": "shadow_structure_requires_promoted_base_model",
        "managed_target_hit_probability": None,
        "managed_target_hit_probability_lcb": None,
        "managed_probability_model": None,
        "risk_tier": "research_only",
        "managed_signal_plan": managed_plan,
        "shadow_only": True,
        "shadow_reason": _SHADOW_GRID_HOLD_REASON,
        "shadow_schema_version": candidate.schema_version,
        "shadow_candidate_id": candidate.candidate_id,
        "shadow_strategy": candidate.strategy,
        "admission_enabled": False,
        "managed_marketable_entry_net": -candidate.entry_debit_dollars / 100.0,
        "managed_marketable_basis_dollars": candidate.entry_debit_dollars,
        "managed_commission_estimate": commissions,
        "estimated_round_trip_cost": candidate.round_trip_friction_dollars,
        "structure_kind": features.get("structure_kind"),
        "structure_leg_count": leg_count,
        "structure_width": width,
        "structure_net_delta": features.get("net_delta"),
        "structure_net_gamma": features.get("net_gamma"),
        "structure_net_theta": features.get("net_theta"),
        "structure_net_vega": features.get("net_vega"),
        "structure_friction_fraction": features.get("friction_fraction"),
        "structure_desired_premium_target_dollars": (
            candidate.desired_premium_target_dollars
        ),
        "structure_target_scenario_pnl_dollars": (
            candidate.target_scenario_pnl_dollars
        ),
        "structure_invalidation_scenario_pnl_dollars": (
            candidate.invalidation_scenario_pnl_dollars
        ),
        "structure_timeout_scenario_pnl_dollars": (
            candidate.timeout_scenario_pnl_dollars
        ),
        "thesis_entry_spot": features.get("thesis_entry_spot"),
        "thesis_invalidation_spot": features.get("thesis_invalidation_spot"),
        "thesis_target_spot": features.get("thesis_target_spot"),
        "thesis_underlying_risk_fraction": features.get(
            "underlying_risk_fraction"
        ),
        "thesis_underlying_reward_risk": features.get(
            "underlying_reward_risk"
        ),
        "thesis_timeout_minutes": features.get("timeout_minutes"),
        "premium_target_feasible": candidate.premium_target_feasible,
    }
    if opening_range_plan is not None:
        # Legacy OR consumers continue to receive the exact plan shape while
        # managed capture uses the row-local, provenance-rich plan above.
        suggestion["opening_range_fvg"] = opening_range_plan
    return {
        "strategy": _shadow_strategy_identity(candidate, managed_plan),
        # A shadow row has no admission score. The non-null legacy column uses
        # zero while `shadow_only` remains the authoritative hold invariant.
        "score": 0.0,
        "rationale": (
            f"shadow-only optimizer alternative ({candidate.strategy}); "
            "requires a promoted causal base model"
        ),
        "legs_json": _serialize_legs(candidate.legs),
        "suggestion_json": suggestion,
    }


def _utc_iso(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


def _force_exit_at(signal_at: datetime, settings: Settings) -> datetime | None:
    close = nyse_session_close_utc(signal_at)
    if close is None:
        return None
    return close - timedelta(minutes=settings.execution.zero_dte_force_exit_minutes)


def _opening_range_managed_plan(
    signal: OpeningRangeFVGSignal,
    *,
    observed_at: datetime,
    settings: Settings,
) -> dict[str, object] | None:
    """Trusted row-local identity for the existing production OR plan."""
    signal_at = signal.respected_ts + timedelta(minutes=signal.timeframe_minutes)
    force_exit_at = _force_exit_at(signal_at, settings)
    if force_exit_at is None:
        return None
    return {
        "schema_version": "managed_signal_plan_v1",
        "status": "entry_confirmed",
        "source": "trusted_daemon",
        "authority": "existing_opening_range_policy",
        "admission_enabled": True,
        "signal_id": signal.signal_id,
        "session": signal.session,
        "direction": signal.direction,
        "generator": "opening_range_fvg",
        "setup_type": signal.setup_type,
        "option_expiry": signal.session.replace("-", ""),
        "signal_at": _utc_iso(signal_at),
        "observed_at": _utc_iso(observed_at),
        "causal_cutoff_at": _utc_iso(signal_at),
        "thesis_expires_at": _utc_iso(force_exit_at),
        "stop_pct": signal.stop_pct,
        "target_r": signal.target_r,
        "target_pct": signal.target_pct,
        "label_policy": "premium_stop_target_before_force_exit_v1",
    }


def _hypothesis_managed_plan(
    hypothesis: ShadowIntradayHypothesis,
    *,
    observed_at: datetime,
    entry_spot: float,
    settings: Settings,
) -> tuple[dict[str, object], UnderlyingThesis] | None:
    """Freeze a causal shadow thesis without granting admission authority."""
    force_exit_at = _force_exit_at(hypothesis.signal_at, settings)
    if force_exit_at is None:
        return None
    hypothesis_expiry = (
        hypothesis.thesis_expires_at
        if hypothesis.thesis_expires_at.tzinfo is not None
        else hypothesis.thesis_expires_at.replace(tzinfo=UTC)
    ).astimezone(UTC)
    expires_at = min(hypothesis_expiry, force_exit_at.astimezone(UTC))
    observed_utc = (
        observed_at
        if observed_at.tzinfo is not None
        else observed_at.replace(tzinfo=UTC)
    ).astimezone(UTC)
    if expires_at <= observed_utc or not math.isfinite(entry_spot) or entry_spot <= 0.0:
        return None
    invalidation = float(hypothesis.invalidation_level)
    risk_distance = (
        entry_spot - invalidation
        if hypothesis.direction == "bull"
        else invalidation - entry_spot
    )
    if not math.isfinite(risk_distance) or risk_distance <= 0.0:
        return None
    target_r = settings.execution.opening_range_target_r_min
    target_spot = (
        entry_spot + risk_distance * target_r
        if hypothesis.direction == "bull"
        else entry_spot - risk_distance * target_r
    )
    if not math.isfinite(target_spot) or target_spot <= 0.0:
        return None
    timeout_minutes = (expires_at - observed_utc).total_seconds() / 60.0
    stop_pct = settings.execution.opening_range_stop_pct
    plan: dict[str, object] = {
        "schema_version": "managed_signal_plan_v1",
        "status": "shadow_confirmed",
        "source": "trusted_daemon",
        "authority": hypothesis.authority,
        "admission_enabled": False,
        "calibration_status": hypothesis.calibration_status,
        "signal_id": hypothesis.hypothesis_id,
        "session": hypothesis.session,
        "direction": hypothesis.direction,
        "generator": hypothesis.generator,
        "setup_type": hypothesis.generator,
        "option_expiry": hypothesis.option_expiry,
        "signal_at": _utc_iso(hypothesis.signal_at),
        "observed_at": _utc_iso(observed_at),
        "causal_cutoff_at": _utc_iso(hypothesis.causal_cutoff_at),
        "thesis_expires_at": _utc_iso(expires_at),
        "stop_pct": stop_pct,
        "target_r": target_r,
        "target_pct": stop_pct * target_r,
        "thesis_entry_spot": entry_spot,
        "thesis_invalidation_spot": invalidation,
        "thesis_target_spot": target_spot,
        "label_policy": "premium_stop_target_before_force_exit_v1",
        "hypothesis": hypothesis.to_dict(),
    }
    return (
        plan,
        UnderlyingThesis(
            direction=hypothesis.direction,
            entry_spot=entry_spot,
            invalidation_spot=invalidation,
            target_spot=target_spot,
            timeout_minutes=timeout_minutes,
        ),
    )


def _opening_range_round_trip_cost(
    suggestion: StrategySuggestion,
    chain: list[OptionChainLeg],
    settings: Settings,
) -> float | None:
    """Estimate one-unit entry+exit costs from the live scan chain."""
    quotes = {(q.expiry, q.strike, q.right): q for q in chain}
    contracts = 0
    combo_spread = 0.0
    for leg in suggestion.legs:
        if leg.sec_type != "OPT":
            continue
        if leg.expiry is None or leg.strike is None or leg.right is None:
            return None
        quote = quotes.get((leg.expiry, leg.strike, leg.right))
        if (
            quote is None
            or quote.bid is None
            or quote.ask is None
            or not math.isfinite(quote.bid)
            or not math.isfinite(quote.ask)
            or quote.ask < quote.bid
        ):
            return None
        contracts += leg.quantity
        combo_spread += (quote.ask - quote.bid) * leg.quantity
    return estimated_round_trip_cost(
        option_contracts_per_unit=contracts,
        combo_spread_per_share=combo_spread,
        commission_per_contract=(
            settings.execution.opening_range_commission_per_contract
        ),
        slippage_spread_fraction=(
            settings.execution.opening_range_round_trip_slippage_spread_frac
        ),
    )


def _opening_range_marketable_entry(
    suggestion: StrategySuggestion,
    chain: list[OptionChainLeg],
    settings: Settings,
) -> tuple[float, float, float] | None:
    """Signed marketable entry net, debit basis, and round-trip commission."""
    quotes = {(q.expiry, q.strike, q.right): q for q in chain}
    entry_net = 0.0
    contracts = 0
    for leg in suggestion.legs:
        if leg.sec_type != "OPT":
            continue
        if leg.expiry is None or leg.strike is None or leg.right is None:
            return None
        quote = quotes.get((leg.expiry, leg.strike, leg.right))
        if quote is None or quote.bid is None or quote.ask is None:
            return None
        if (
            not math.isfinite(float(quote.bid))
            or not math.isfinite(float(quote.ask))
            or quote.bid < 0.0
            or quote.ask < quote.bid
        ):
            return None
        quantity = int(leg.quantity)
        contracts += quantity
        entry_net += (
            quote.bid * quantity
            if leg.side == "sell"
            else -quote.ask * quantity
        )
    basis = abs(entry_net) * 100.0
    if contracts <= 0 or entry_net >= 0.0 or basis <= 0.0:
        return None
    commissions = (
        contracts
        * 2.0
        * settings.execution.opening_range_commission_per_contract
    )
    return entry_net, basis, commissions


def _marketable_entry_component(
    entry: tuple[float, float, float] | None,
    index: int,
) -> float | None:
    """Mypy-friendly projection from an optional frozen entry tuple."""
    return entry[index] if entry is not None else None


async def scan_symbol(
    symbol: str,
    ibkr: IBKRClient,
    engine: Engine,
    settings: Settings,
    resolver: ContractResolver | None = None,
    view_override: tuple[Direction | None, IVRegime | None] | None = None,
    opening_range_signal: OpeningRangeFVGSignal | None = None,
    managed_hypotheses: tuple[ShadowIntradayHypothesis, ...] = (),
) -> ScanResult:
    """Scan one symbol end-to-end and persist snapshot + strategy_scores.

    Returns a ScanResult containing the new snapshot's PK, the synthesized
    MarketView (after applying view_override if given), and every scored
    strategy.

    IV-rank uses a forward-accumulated daily ATM IV history (IBK-89): each
    scan records today's ATM IV into ``iv_history`` (latest scan of the day
    wins) and feeds the trailing daily series to ``iv_rank``. Until ~30
    trading days accumulate, ``iv_rank`` reports ``warming_up=True`` and the
    scoring engine treats IV-rank as neutral (0.5); the other 5 factors are
    unaffected.
    """
    symbol = symbol.upper().strip()
    if resolver is None:
        resolver = ContractResolver(ibkr)
    await ibkr.ensure_connected()

    history_client = HistoryClient(ibkr, resolver=resolver)
    chain_client = ChainClient(
        ibkr,
        resolver=resolver,
        max_market_data_lines=settings.ibkr.max_market_data_lines,
    )
    market_client = MarketDataClient(ibkr, resolver=resolver)
    positions_client = PositionsClient(ibkr)

    # Fetch the stock snapshot first so the spot can drive get_chain's near-ATM
    # strike windowing; the remaining calls then run concurrently.
    stock = await market_client.get_stock_snapshot(symbol)
    spot = stock.mid or stock.last or 0.0
    bars, chain, positions, account = await asyncio.gather(
        history_client.get_history(symbol, days=252),
        chain_client.get_chain(
            symbol,
            underlying_price=spot,
            strike_band_pct=settings.scan.strike_band_pct,
            max_strikes_per_side=settings.scan.max_strikes_per_side,
            dte_window=(settings.scan.dte_window_min, settings.scan.dte_window_max),
            dte_target=settings.scan.dte_target,
            # Exact-0DTE mode must not spend quote lines or alert slots on a
            # back-month calendar that the execution invariant will reject.
            back_dte_gap=(
                None
                if settings.execution.zero_dte_only
                else settings.scan.back_month_dte_gap
            ),
        ),
        positions_client.get_positions(),
        positions_client.get_account_summary(),
    )

    atm_iv = _atm_iv(chain, spot)
    hv20 = historical_volatility(bars["close"], window=20) if not bars.empty else None
    # historical_volatility returns float('nan') (not None) when the bar history
    # is shorter than window+1. The scoring layer's iv_hv_score guards `hv is
    # None` but NOT `math.isnan(hv)`, and Python's min/max skip NaN -- so a NaN
    # hv20 silently produces iv_hv_score=1.0 instead of the intended neutral 0.5.
    # Normalize to None here so downstream guards work as written.
    if hv20 is not None and math.isnan(hv20):
        hv20 = None
    now = datetime.now(UTC)
    front_expiry = closest_expiry_to_dte(
        tuple(chain), settings.scan.dte_target, today=now.date()
    )
    front_dte = (
        (datetime.strptime(front_expiry, "%Y%m%d").date() - now.date()).days
        if front_expiry is not None
        else None
    )
    minutes_left = minutes_to_nyse_close(now)
    expiry_horizon_days = (
        front_dte
        if front_dte is not None and front_dte > 0
        else max(0.0, minutes_left / (24.0 * 60.0))
        if front_dte == 0 and minutes_left is not None
        else None
    )
    expected_move = (
        spot * atm_iv * math.sqrt(expiry_horizon_days / 365.0)
        if atm_iv is not None
        and atm_iv > 0.0
        and spot > 0.0
        and expiry_horizon_days is not None
        and expiry_horizon_days > 0
        else None
    )
    if atm_iv is not None:
        # Record today's ATM IV (keyed by UTC date; latest scan of the day
        # wins -- US sessions don't cross UTC midnight), then read the trailing
        # daily series so iv_rank ranks against real history. Today is included
        # in the window -> standard IV-rank, no spurious clamping.
        record_atm_iv(engine, symbol, now.date(), atm_iv)
        iv_history = read_atm_iv_history(engine, symbol)
    else:
        # No option data today: don't rank a bogus 0.0 against real history.
        iv_history = pd.Series([], dtype=float)
    # Earnings comes from yfinance (blocking, occasionally slow/down). Fetch it
    # off the event loop with a hard timeout so a Yahoo hiccup can't freeze the
    # loop or stall the scan; infer_view itself is pure (IBK-149).
    earnings = await _bounded_to_thread(
        next_earnings,
        symbol,
        timeout=settings.scan.external_data_timeout_s,
        default=EarningsInfo(next_date=None, source="unknown"),
        label=f"next_earnings({symbol})",
    )
    inferred_view = infer_view(
        bars, current_atm_iv=atm_iv or 0.0, atm_iv_history=iv_history, earnings=earnings
    )
    configured_view = _override_view(inferred_view, view_override)
    view = configured_view
    if settings.scan.opening_range_fvg_enabled and opening_range_signal is not None:
        # The price-action setup supplies the directional thesis. A debit
        # vertical still limits IV exposure when the broad IV regime is high,
        # so use neutral applicability without falsifying the raw stored IV.
        scoring_iv = "neutral" if view.iv_regime == "high" else view.iv_regime
        view = replace(
            view,
            direction=opening_range_signal.direction,
            direction_strength="strong",
            iv_regime=scoring_iv,
        )

    sym_position = next(
        (p for p in positions if p.symbol == symbol and p.sec_type == "STK"), None
    )

    snapshot = StrategySnapshot(
        symbol=symbol,
        spot=spot,
        atm_iv=atm_iv,
        hv20=hv20,
        iv_rank=view.iv_rank_value,
        chain=tuple(chain),
        view=view,
        dte_target=settings.scan.dte_target,
        position=sym_position,
        same_day_time_to_expiry_days=(
            expiry_horizon_days if front_dte == 0 else None
        ),
    )

    account_value = (
        float(account.net_liquidation_usd)
        if account.net_liquidation_usd is not None
        else None
    )
    if account_value is None:
        log.info(
            "No account net-liquidation for %s; position sizing skipped "
            "(suggested_quantity will be 0).",
            symbol,
        )
    # Option scoring needs real per-leg data (IV/greeks). If the chain came back
    # with no usable option data -- e.g. the account lacks an options (OPRA)
    # market-data subscription -- skip scoring rather than emit strategies built
    # off the directional view alone (they'd carry no real strikes/pricing).
    has_option_data = any(leg.iv is not None or leg.delta is not None for leg in chain)
    if has_option_data and not settings.scan.opening_range_fvg_enabled:
        scored = score_all(
            snapshot, account_value=account_value, risk_pct=settings.scan.risk_pct
        )
    elif has_option_data and opening_range_signal is not None:
        strategy_names = (
            ("bull_call_spread", "long_call")
            if opening_range_signal.direction == "bull"
            else ("bear_put_spread", "long_put")
        )
        scored = score_all(
            snapshot,
            account_value=account_value,
            risk_pct=settings.scan.risk_pct,
            strategies=tuple(get_strategy(name) for name in strategy_names),
        )
    else:
        scored = ()
        if not has_option_data:
            log.warning(
                "No option market data for %s (%d chain legs, none with IV/greeks); "
                "skipping strategy scoring -- check the options (OPRA) market-data "
                "subscription.",
                symbol,
                len(chain),
            )

    terminal_expected_values = {
        item.strategy_name: item.suggestion.expected_value for item in scored
    }
    opening_range_plan: dict[str, object] | None = None
    opening_range_managed_plan: dict[str, object] | None = None
    if opening_range_signal is not None:
        opening_range_plan = opening_range_signal.to_dict()
        if opening_range_signal.quality is not None:
            # Keep the dedicated setup features shadow-only.  Persist the raw
            # inferred regime beside them instead of letting the OR direction
            # overwrite the historical context needed for later calibration.
            opening_range_plan["quality"] = quality_payload_with_regime(
                opening_range_signal.quality,
                inferred_view,
            )
        opening_range_managed_plan = _opening_range_managed_plan(
            opening_range_signal,
            observed_at=now,
            settings=settings,
        )
    managed_signal_plans: list[dict[str, object]] = []
    if opening_range_managed_plan is not None:
        managed_signal_plans.append(opening_range_managed_plan)
    shadow_structure_rows: list[dict[str, object]] = []
    shadow_row_budget = min(
        _MAX_SHADOW_ROWS_PER_SNAPSHOT,
        settings.validation.managed_capture_max_active,
    )
    if (
        opening_range_plan is not None
        and opening_range_managed_plan is not None
        and opening_range_signal is not None
        and minutes_left is not None
        and minutes_left > settings.execution.zero_dte_force_exit_minutes
        and shadow_row_budget > 0
    ):
        shadow_timeout_minutes = (
            minutes_left - settings.execution.zero_dte_force_exit_minutes
        )
        try:
            shadow_structure_rows = [
                _shadow_structure_score_row(
                    candidate,
                    opening_range_managed_plan,
                    settings,
                    opening_range_plan=opening_range_plan,
                )
                for candidate in build_shadow_structure_grid(
                    chain,
                    opening_range_signal,
                    timeout_minutes=shadow_timeout_minutes,
                    commission_per_contract=(
                        settings.execution.opening_range_commission_per_contract
                    ),
                    max_candidates=min(
                        _MAX_GRID_STRUCTURES_PER_PLAN,
                        shadow_row_budget,
                    ),
                )
            ]
        except ValueError:
            # A malformed thesis should not poison the ordinary scanner. The
            # base score remains available and no alternative is invented.
            log.exception("shadow structure grid rejected %s thesis", symbol)
    shadow_row_budget -= len(shadow_structure_rows)

    # Independent generators remain a separate, shadow-only research path.
    # Newest causal hypotheses get the bounded quote budget first; their rows
    # never enter ``scored`` and therefore cannot reach alerts or execution.
    selected_hypotheses = sorted(
        (
            item
            for item in managed_hypotheses
            if item.symbol == symbol
            and item.option_expiry == item.session.replace("-", "")
        ),
        key=lambda item: (item.signal_at, item.generator, item.hypothesis_id),
        reverse=True,
    )[:_MAX_SHADOW_HYPOTHESES_PER_SNAPSHOT]
    for hypothesis in selected_hypotheses:
        frozen = _hypothesis_managed_plan(
            hypothesis,
            observed_at=now,
            entry_spot=spot,
            settings=settings,
        )
        if frozen is None:
            continue
        managed_plan, thesis = frozen
        managed_signal_plans.append(managed_plan)
        if shadow_row_budget <= 0:
            continue
        try:
            candidates = build_shadow_grid_for_thesis(
                chain,
                thesis,
                expiry=hypothesis.option_expiry,
                target_pct=(
                    settings.execution.opening_range_stop_pct
                    * settings.execution.opening_range_target_r_min
                ),
                commission_per_contract=(
                    settings.execution.opening_range_commission_per_contract
                ),
                max_candidates=min(
                    _MAX_GRID_STRUCTURES_PER_PLAN,
                    shadow_row_budget,
                ),
            )
        except ValueError:
            log.exception(
                "shadow hypothesis structure grid rejected %s/%s thesis",
                symbol,
                hypothesis.hypothesis_id,
            )
            continue
        rows = [
            _shadow_structure_score_row(candidate, managed_plan, settings)
            for candidate in candidates
        ]
        shadow_structure_rows.extend(rows)
        shadow_row_budget -= len(rows)
    opening_range_round_trip_costs: dict[str, float | None] = {}
    opening_range_marketable_entries: dict[
        str, tuple[float, float, float] | None
    ] = {}
    gross_managed_expected_values: dict[str, float | None] = {}
    managed_break_even_probabilities: dict[str, float | None] = {}
    if opening_range_plan is not None:
        for item in scored:
            opening_range_marketable_entries[item.strategy_name] = (
                _opening_range_marketable_entry(item.suggestion, chain, settings)
            )
            opening_range_round_trip_costs[item.strategy_name] = (
                _opening_range_round_trip_cost(item.suggestion, chain, settings)
            )
            gross_managed_expected_values[item.strategy_name] = managed_expected_value(
                credit_or_debit=item.suggestion.credit_or_debit,
                # No target-before-stop model is currently promoted.  Terminal
                # expiry PoP must never silently stand in for this value.
                target_hit_probability=None,
                plan=opening_range_plan,
                maximum_profit=item.suggestion.max_profit,
            )
            managed_break_even_probabilities[item.strategy_name] = (
                managed_break_even_probability(
                    credit_or_debit=item.suggestion.credit_or_debit,
                    plan=opening_range_plan,
                    estimated_round_trip_cost=opening_range_round_trip_costs.get(
                        item.strategy_name
                    ),
                    maximum_profit=item.suggestion.max_profit,
                )
            )
        scored = tuple(
            replace(
                item,
                suggestion=with_managed_expected_value(
                    item.suggestion,
                    opening_range_plan,
                    target_hit_probability=None,
                    estimated_round_trip_cost=opening_range_round_trip_costs.get(
                        item.strategy_name
                    ),
                ),
            )
            for item in scored
        )

    ratio = iv_hv_ratio(atm_iv, hv20) if (atm_iv is not None and hv20 is not None) else None
    relative_strength_value: float | None = None
    beta_to_benchmark: float | None = None
    try:
        if symbol == settings.scan.benchmark_symbol:
            relative_strength_value = 0.0
            beta_to_benchmark = 1.0
        else:
            benchmark_bars = await history_client.get_history(
                settings.scan.benchmark_symbol, days=252
            )
            relative_strength_value = relative_strength(
                bars, benchmark_bars, settings.scan.relative_strength_window
            )
            beta_to_benchmark = beta(
                bars,
                benchmark_bars,
                window=settings.portfolio.beta_window,
            )
    except Exception:  # noqa: BLE001 -- benchmark data is best-effort
        log.exception("relative strength failed for %s", symbol)

    recent_price_history: list[dict[str, object]] = []
    if "close" in bars:
        for index, value in bars["close"].tail(20).items():
            close = float(value)
            if not math.isfinite(close):
                continue
            iso = index.isoformat() if hasattr(index, "isoformat") else str(index)
            recent_price_history.append({"ts": iso, "close": close})

    raw_extra: dict[str, object] = {
        "delayed": stock.delayed,
        "n_chain_legs": len(chain),
        "warming_up": view.warming_up,
        "iv_rank_is_proxy": view.iv_rank_is_proxy,
        # The analysis-window flag and exact event date are deliberately
        # separate. Execution compares the latter with the candidate's actual
        # expiries instead of treating "earnings within 14 days" as "earnings
        # before this 0DTE contract expires".
        "earnings_in_window": view.earnings_in_window,
        "next_earnings_date": (
            earnings.next_date.isoformat() if earnings.next_date is not None else None
        ),
        "earnings_source": earnings.source,
        "relative_strength": relative_strength_value,
        "beta_to_benchmark": beta_to_benchmark,
        "beta_benchmark": settings.scan.benchmark_symbol,
        "recent_price_history": recent_price_history,
        "front_expiry": front_expiry,
        "front_dte": front_dte,
        "expected_move": expected_move,
        "inferred_market_view": asdict(inferred_view),
        "configured_market_view": asdict(configured_view),
        "effective_scoring_view": asdict(view),
        "opening_range_fvg": (
            opening_range_plan
            if opening_range_plan is not None
            else {
                "status": "not_confirmed",
                "source": "trusted_daemon",
            }
            if settings.scan.opening_range_fvg_enabled
            else None
        ),
        "managed_signal_plans": managed_signal_plans,
        "shadow_intraday_hypotheses": [
            item.to_dict() for item in selected_hypotheses
        ],
    }
    with engine.begin() as conn:
        result = conn.execute(
            insert(snapshots_t).values(
                symbol=symbol,
                ts=now,
                spot=spot,
                iv_rank=view.iv_rank_value,
                hv20=hv20,
                iv_hv_ratio=ratio,
                expected_move=expected_move,
                regime_dir=view.direction,
                regime_iv=view.iv_regime,
                raw_json=raw_extra,
            )
        )
        # inserted_primary_key is a named-tuple; index-subscript is the supported API,
        # but SQLAlchemy's stubs type it as Any so mypy reports a spurious index error.
        snapshot_id = cast(int, result.inserted_primary_key[0])  # type: ignore[index]
        persisted_scores: list[dict[str, object]] = [
            {
                "snapshot_id": snapshot_id,
                "strategy": s.strategy_name,
                "score": s.score,
                "rationale": s.rationale,
                "legs_json": _serialize_legs(s.suggestion.legs),
                # Persist the StrategySuggestion fields the retry path
                # needs to render an identical alert. Without these the
                # retry alert silently drops the UNDEFINED RISK header
                # and all financial figures.
                "suggestion_json": {
                    "defined_risk": s.suggestion.defined_risk,
                    "credit_or_debit": s.suggestion.credit_or_debit,
                    "max_loss": s.suggestion.max_loss,
                    "max_profit": s.suggestion.max_profit,
                    "prob_profit": s.suggestion.prob_profit,
                    "suggested_quantity": s.suggestion.suggested_quantity,
                    "reward_risk": s.suggestion.reward_risk,
                    "expected_value": s.suggestion.expected_value,
                    "expected_value_model": (
                        "managed_outcome_calibration_required_v3"
                        if opening_range_plan is not None
                        else "terminal_expiry_v1"
                    ),
                    "terminal_expected_value": terminal_expected_values.get(
                        s.strategy_name
                    ),
                    "gross_managed_expected_value": (
                        gross_managed_expected_values.get(s.strategy_name)
                    ),
                    "managed_target_hit_probability": None,
                    "managed_target_hit_probability_lcb": None,
                    "managed_probability_model": None,
                    "managed_break_even_probability": (
                        managed_break_even_probabilities.get(s.strategy_name)
                    ),
                    "estimated_round_trip_cost": (
                        opening_range_round_trip_costs.get(s.strategy_name)
                    ),
                    "managed_marketable_entry_net": (
                        _marketable_entry_component(
                            opening_range_marketable_entries.get(s.strategy_name),
                            0,
                        )
                    ),
                    "managed_marketable_basis_dollars": (
                        _marketable_entry_component(
                            opening_range_marketable_entries.get(s.strategy_name),
                            1,
                        )
                    ),
                    "managed_commission_estimate": (
                        _marketable_entry_component(
                            opening_range_marketable_entries.get(s.strategy_name),
                            2,
                        )
                    ),
                    "risk_tier": s.suggestion.risk_tier,
                    "opening_range_fvg": (
                        opening_range_plan
                    ),
                    "managed_signal_plan": opening_range_managed_plan,
                },
            }
            for s in scored
        ]
        persisted_scores.extend(
            {"snapshot_id": snapshot_id, **row}
            for row in shadow_structure_rows
        )
        if persisted_scores:
            conn.execute(
                insert(scores_t),
                persisted_scores,
            )

    # Prefer the Gateway's entitled API news: it is timely, source-attributed,
    # and available independently of the quote feed. Refresh once per scan
    # interval. Yahoo remains a slow-path fallback for accounts with no API
    # news entitlement or a transient news-provider failure.
    news_refresh_due = news_cache_is_stale(
        symbol,
        engine,
        throttle_minutes=settings.scan.interval_minutes,
    )
    news_ready = not news_refresh_due
    if news_refresh_due:
        try:
            headlines = await asyncio.wait_for(
                NewsClient(ibkr, resolver=resolver).headlines(symbol, limit=10),
                timeout=settings.scan.external_data_timeout_s,
            )
            if headlines:
                replace_news(symbol, engine, headlines)
                news_ready = True
        except TimeoutError:
            log.warning(
                "IBKR news refresh for %s timed out after %.1fs",
                symbol,
                settings.scan.external_data_timeout_s,
            )
        except Exception:  # noqa: BLE001 -- catalyst data is best-effort
            log.exception("IBKR news refresh failed for %s", symbol)
    if not news_ready:
        await _bounded_to_thread(
            refresh_news_if_stale,
            symbol,
            engine,
            timeout=settings.scan.external_data_timeout_s,
            default=None,
            label=f"refresh_news({symbol})",
        )

    return ScanResult(
        symbol=symbol,
        snapshot_id=snapshot_id,
        snapshot_ts=now,
        view=view,
        scored=scored,
    )
