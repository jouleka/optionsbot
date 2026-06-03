"""Calibration backtest: model prob_profit vs historically-realized win-rate."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime

from sqlalchemy import Engine, select

from optionsbot.scoring.payoff import is_terminal_modelable, terminal_pnl_dollars
from optionsbot.storage.schema import snapshots as snapshots_t
from optionsbot.storage.schema import strategy_scores as scores_t
from optionsbot.strategies.base import Leg
from optionsbot.validation.types import (
    BacktestReport,
    BacktestRow,
    CalibrationBucket,
    PickRecord,
)

_TRADING_DAYS_PER_YEAR = 252
_CAL_DAYS_PER_YEAR = 365


def horizon_trading_days(dte_days: int) -> int:
    """Map calendar DTE to a trading-day horizon (>=1)."""
    return max(1, round(dte_days * _TRADING_DAYS_PER_YEAR / _CAL_DAYS_PER_YEAR))


def historical_win_rate(
    legs: Iterable[Leg],
    credit_or_debit: float,
    entry_spot: float,
    closes: Sequence[float],
    dte_days: int,
) -> tuple[float, float, int] | None:
    """(raw_win_rate, dedrift_win_rate, n) over all overlapping DTE-forward returns.

    raw applies the realized log-returns as-is; dedrift subtracts the mean
    log-return (zero-drift, same vol/tails) to isolate the drift the zero-drift
    model ignores. Returns None if there are too few samples.
    """
    legs = tuple(legs)
    px = [float(c) for c in closes]
    n_h = horizon_trading_days(dte_days)
    if len(px) <= n_h or entry_spot <= 0.0:
        return None
    log_rets = [
        math.log(px[i + n_h] / px[i])
        for i in range(len(px) - n_h)
        if px[i] > 0.0 and px[i + n_h] > 0.0
    ]
    if not log_rets:
        return None
    mean_lr = sum(log_rets) / len(log_rets)
    raw_wins = 0
    dedrift_wins = 0
    for lr in log_rets:
        if terminal_pnl_dollars(legs, credit_or_debit, entry_spot * math.exp(lr)) > 0.0:
            raw_wins += 1
        if terminal_pnl_dollars(
            legs, credit_or_debit, entry_spot * math.exp(lr - mean_lr)
        ) > 0.0:
            dedrift_wins += 1
    n = len(log_rets)
    return raw_wins / n, dedrift_wins / n, n


def _bucket(rows: Sequence[BacktestRow], lo: float, hi: float) -> CalibrationBucket:
    count = len(rows)
    if count == 0:
        return CalibrationBucket(lo, hi, 0, 0.0, 0.0, 0.0)
    return CalibrationBucket(
        lo=lo,
        hi=hi,
        count=count,
        mean_pred=sum(r.predicted for r in rows) / count,
        mean_raw=sum(r.raw for r in rows) / count,
        mean_dedrift=sum(r.dedrift for r in rows) / count,
    )


def calibrate(rows: Sequence[BacktestRow], n_buckets: int = 10) -> BacktestReport:
    """Bucket rows by predicted prob_profit; aggregate overall + per strategy."""
    width = 1.0 / n_buckets
    buckets: list[CalibrationBucket] = []
    for i in range(n_buckets):
        lo, hi = i * width, (i + 1) * width
        # Last bucket is inclusive of 1.0.
        last = i == n_buckets - 1
        in_bucket = [
            r for r in rows
            if r.predicted >= lo and (r.predicted < hi or (last and r.predicted <= hi))
        ]
        buckets.append(_bucket(in_bucket, lo, hi))
    strategies = sorted({r.strategy for r in rows})
    by_strategy = {
        s: _bucket([r for r in rows if r.strategy == s], 0.0, 1.0) for s in strategies
    }
    overall = _bucket(list(rows), 0.0, 1.0)
    return BacktestReport(
        buckets=tuple(buckets),
        by_strategy=by_strategy,
        overall_count=overall.count,
        overall_mean_pred=overall.mean_pred,
        overall_mean_raw=overall.mean_raw,
        overall_mean_dedrift=overall.mean_dedrift,
    )


async def run_backtest(
    picks: Sequence[PickRecord],
    fetch_closes: Callable[[str], Awaitable[Sequence[float]]],
) -> BacktestReport:
    """Fetch each symbol's closes (once, cached), compute per-pick win-rates, calibrate."""
    closes_cache: dict[str, Sequence[float]] = {}
    rows: list[BacktestRow] = []
    for pick in picks:
        if pick.symbol not in closes_cache:
            closes_cache[pick.symbol] = await fetch_closes(pick.symbol)
        out = historical_win_rate(
            pick.legs, pick.credit_or_debit, pick.entry_spot,
            closes_cache[pick.symbol], pick.dte_days,
        )
        if out is None:
            continue
        raw, dedrift, n = out
        rows.append(BacktestRow(
            symbol=pick.symbol, strategy=pick.strategy, predicted=pick.prob_profit,
            raw=raw, dedrift=dedrift, n=n,
        ))
    return calibrate(rows)


def load_pick_records(engine: Engine) -> list[PickRecord]:
    """Read terminal-modelable picks from strategy_scores joined to snapshots."""
    stmt = (
        select(
            snapshots_t.c.symbol, snapshots_t.c.spot, snapshots_t.c.ts,
            scores_t.c.strategy, scores_t.c.score, scores_t.c.legs_json,
            scores_t.c.suggestion_json,
        )
        .select_from(scores_t.join(snapshots_t, scores_t.c.snapshot_id == snapshots_t.c.id))
    )
    out: list[PickRecord] = []
    with engine.connect() as conn:
        for row in conn.execute(stmt).fetchall():
            sug = row.suggestion_json or {}
            if isinstance(sug, str):
                sug = json.loads(sug)
            pop = sug.get("prob_profit")
            spot = row.spot
            if pop is None or spot is None:
                continue
            legs_data = row.legs_json or []
            if isinstance(legs_data, str):
                legs_data = json.loads(legs_data)
            legs = tuple(Leg(**le) for le in legs_data)
            if not is_terminal_modelable(legs):
                continue
            expiry = legs[0].expiry
            assert expiry is not None
            entry_date = row.ts.date()
            dte_days = (datetime.strptime(expiry, "%Y%m%d").date() - entry_date).days
            if dte_days <= 0:
                continue
            out.append(PickRecord(
                symbol=row.symbol, entry_spot=float(spot), entry_date=entry_date,
                expiry=expiry, dte_days=dte_days, legs=legs,
                credit_or_debit=float(sug.get("credit_or_debit", 0.0)),
                prob_profit=float(pop), score=float(row.score), strategy=row.strategy,
            ))
    return out
