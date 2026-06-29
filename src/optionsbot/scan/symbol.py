"""End-to-end scan for a single symbol: IBKR fetch -> analysis -> scoring -> persist."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import cast

import pandas as pd
from sqlalchemy import Engine, insert

from optionsbot.analysis.events import next_earnings
from optionsbot.analysis.news import refresh_news_if_stale
from optionsbot.analysis.relative_strength import relative_strength
from optionsbot.analysis.types import Direction, EarningsInfo, IVRegime, MarketView
from optionsbot.analysis.view import infer_view
from optionsbot.analysis.volatility import historical_volatility, iv_hv_ratio
from optionsbot.config import Settings
from optionsbot.ibkr import (
    ChainClient,
    HistoryClient,
    IBKRClient,
    MarketDataClient,
    PositionsClient,
)
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.scan.types import ScanResult
from optionsbot.scoring import score_all
from optionsbot.storage.iv_history import read_atm_iv_history, record_atm_iv
from optionsbot.storage.schema import snapshots as snapshots_t
from optionsbot.storage.schema import strategy_scores as scores_t
from optionsbot.strategies import Leg, StrategySnapshot

log = logging.getLogger(__name__)


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


async def scan_symbol(
    symbol: str,
    ibkr: IBKRClient,
    engine: Engine,
    settings: Settings,
    resolver: ContractResolver | None = None,
    view_override: tuple[Direction | None, IVRegime | None] | None = None,
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
            back_dte_gap=settings.scan.back_month_dte_gap,
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
    view = infer_view(
        bars, current_atm_iv=atm_iv or 0.0, atm_iv_history=iv_history, earnings=earnings
    )
    view = _override_view(view, view_override)

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
        dte_target=45,
        position=sym_position,
    )

    account_value = (
        float(account.net_liquidation) if account.net_liquidation is not None else None
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
    if has_option_data:
        scored = score_all(
            snapshot, account_value=account_value, risk_pct=settings.scan.risk_pct
        )
    else:
        log.warning(
            "No option market data for %s (%d chain legs, none with IV/greeks); "
            "skipping strategy scoring -- check the options (OPRA) market-data "
            "subscription.",
            symbol,
            len(chain),
        )
        scored = ()

    ratio = iv_hv_ratio(atm_iv, hv20) if (atm_iv is not None and hv20 is not None) else None
    relative_strength_value: float | None = None
    try:
        if symbol == settings.scan.benchmark_symbol:
            relative_strength_value = 0.0
        else:
            benchmark_bars = await history_client.get_history(
                settings.scan.benchmark_symbol, days=252
            )
            relative_strength_value = relative_strength(
                bars, benchmark_bars, settings.scan.relative_strength_window
            )
    except Exception:  # noqa: BLE001 -- benchmark data is best-effort
        log.exception("relative strength failed for %s", symbol)

    raw_extra: dict[str, object] = {
        "delayed": stock.delayed,
        "n_chain_legs": len(chain),
        "warming_up": view.warming_up,
        "earnings_in_window": view.earnings_in_window,
        "relative_strength": relative_strength_value,
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
                expected_move=None,
                regime_dir=view.direction,
                regime_iv=view.iv_regime,
                raw_json=raw_extra,
            )
        )
        # inserted_primary_key is a named-tuple; index-subscript is the supported API,
        # but SQLAlchemy's stubs type it as Any so mypy reports a spurious index error.
        snapshot_id = cast(int, result.inserted_primary_key[0])  # type: ignore[index]
        if scored:
            conn.execute(
                insert(scores_t),
                [
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
                            "risk_tier": s.suggestion.risk_tier,
                        },
                    }
                    for s in scored
                ],
            )

    # Best-effort catalyst refresh (throttled inside, never raises). Run it
    # off-loop + time-bounded too: recent_news is the same blocking yfinance
    # path as earnings, so a slow Yahoo response must not freeze the loop or
    # stall the scan (IBK-149).
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
