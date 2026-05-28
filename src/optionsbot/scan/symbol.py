"""End-to-end scan for a single symbol: IBKR fetch -> analysis -> scoring -> persist."""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import cast

import pandas as pd
from sqlalchemy import Engine, insert

from optionsbot.analysis.types import Direction, IVRegime, MarketView
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
from optionsbot.storage.schema import snapshots as snapshots_t
from optionsbot.storage.schema import strategy_scores as scores_t
from optionsbot.strategies import Leg, StrategySnapshot


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

    The IV-rank input is intentionally degenerate in v1 -- the analysis
    layer needs a multi-day ATM IV history that the IBKR layer doesn't yet
    expose. Until IBK-7 lands the history-collection daemon, we pass a
    single-element series and accept that iv_rank will return None with
    warming_up=True. The scoring engine treats this as neutral (0.5), so
    the scan is still meaningful for the other 5 factors.
    """
    symbol = symbol.upper().strip()
    if resolver is None:
        resolver = ContractResolver(ibkr)
    await ibkr.ensure_connected()

    history_client = HistoryClient(ibkr, resolver=resolver)
    chain_client = ChainClient(ibkr, resolver=resolver)
    market_client = MarketDataClient(ibkr, resolver=resolver)
    positions_client = PositionsClient(ibkr)

    # Fetch the stock snapshot first so the spot can drive get_chain's near-ATM
    # strike windowing; the remaining calls then run concurrently.
    stock = await market_client.get_stock_snapshot(symbol)
    spot = stock.mid or stock.last or 0.0
    bars, chain, positions = await asyncio.gather(
        history_client.get_history(symbol, days=120),
        chain_client.get_chain(
            symbol,
            underlying_price=spot,
            strike_band_pct=settings.scan.strike_band_pct,
            max_strikes_per_side=settings.scan.max_strikes_per_side,
        ),
        positions_client.get_positions(),
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
    iv_history = (
        pd.Series([atm_iv]) if atm_iv is not None else pd.Series([], dtype=float)
    )
    view = infer_view(symbol, bars, current_atm_iv=atm_iv or 0.0, atm_iv_history=iv_history)
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

    account_value = None  # account-summary integration deferred to IBK-7
    scored = score_all(snapshot, account_value=account_value)

    now = datetime.now(UTC)
    ratio = iv_hv_ratio(atm_iv, hv20) if (atm_iv is not None and hv20 is not None) else None
    raw_extra: dict[str, object] = {
        "delayed": stock.delayed,
        "n_chain_legs": len(chain),
        "warming_up": view.warming_up,
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
                        },
                    }
                    for s in scored
                ],
            )

    return ScanResult(
        symbol=symbol,
        snapshot_id=snapshot_id,
        snapshot_ts=now,
        view=view,
        scored=scored,
    )
