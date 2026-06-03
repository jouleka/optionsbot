"""Forward outcome ledger: score recorded picks at expiry from the real close."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime

from sqlalchemy import Engine, insert, select

from optionsbot.scoring.payoff import is_terminal_modelable, terminal_pnl_dollars
from optionsbot.storage.schema import pick_outcomes as outcomes_t
from optionsbot.storage.schema import snapshots as snapshots_t
from optionsbot.storage.schema import strategy_scores as scores_t
from optionsbot.strategies.base import Leg
from optionsbot.validation.types import OutcomeGroup, OutcomesReport, UnevaluatedPick

_LEG_FIELDS = frozenset(
    {"symbol", "side", "sec_type", "expiry", "strike", "right", "quantity"}
)


def evaluate_pnl(
    legs: Sequence[Leg], credit_or_debit: float, terminal_spot: float
) -> tuple[float, bool]:
    """Realized P&L (dollars) and win flag at expiry given the terminal underlying."""
    pnl = terminal_pnl_dollars(legs, credit_or_debit, terminal_spot)
    return pnl, pnl > 0.0


def load_unevaluated_expired(engine: Engine, today: date) -> list[UnevaluatedPick]:
    """Terminal-modelable picks whose expiry < today and which have no outcome yet."""
    today_str = today.strftime("%Y%m%d")
    stmt = (
        select(
            scores_t.c.id, snapshots_t.c.symbol, snapshots_t.c.spot,
            scores_t.c.strategy, scores_t.c.score, scores_t.c.legs_json,
            scores_t.c.suggestion_json,
        )
        .select_from(
            scores_t.join(snapshots_t, scores_t.c.snapshot_id == snapshots_t.c.id)
            .outerjoin(outcomes_t, outcomes_t.c.strategy_score_id == scores_t.c.id)
        )
        .where(outcomes_t.c.id.is_(None))
    )
    out: list[UnevaluatedPick] = []
    with engine.connect() as conn:
        for row in conn.execute(stmt).fetchall():
            sug = row.suggestion_json or {}
            if isinstance(sug, str):
                sug = json.loads(sug)
            if sug.get("credit_or_debit") is None or row.spot is None:
                continue
            legs_data = row.legs_json or []
            if isinstance(legs_data, str):
                legs_data = json.loads(legs_data)
            legs = tuple(
                Leg(**{k: v for k, v in le.items() if k in _LEG_FIELDS})
                for le in legs_data
            )
            if not is_terminal_modelable(legs):
                continue
            expiry = legs[0].expiry
            assert expiry is not None
            if expiry >= today_str:
                continue
            out.append(UnevaluatedPick(
                strategy_score_id=row.id, symbol=row.symbol,
                strategy=row.strategy, expiry=expiry, entry_spot=float(row.spot),
                legs=legs, credit_or_debit=float(sug["credit_or_debit"]),
                predicted_prob_profit=sug.get("prob_profit"),
                score=float(row.score), max_profit=sug.get("max_profit"),
                max_loss=sug.get("max_loss"),
                risk_tier=sug.get("risk_tier", "balanced"),
            ))
    return out


async def evaluate_pending(
    engine: Engine,
    fetch_close_at: Callable[[str, str], Awaitable[float | None]],
    today: date,
) -> int:
    """Evaluate every unevaluated expired pick from its terminal close; persist. Returns count."""
    picks = load_unevaluated_expired(engine, today)
    n = 0
    for pick in picks:
        terminal = await fetch_close_at(pick.symbol, pick.expiry)
        if terminal is None:
            continue
        pnl, win = evaluate_pnl(pick.legs, pick.credit_or_debit, terminal)
        with engine.begin() as conn:
            conn.execute(insert(outcomes_t).values(
                strategy_score_id=pick.strategy_score_id, symbol=pick.symbol,
                strategy=pick.strategy, expiry=pick.expiry, entry_spot=pick.entry_spot,
                predicted_prob_profit=pick.predicted_prob_profit, score=pick.score,
                credit_or_debit=pick.credit_or_debit, max_profit=pick.max_profit,
                max_loss=pick.max_loss, risk_tier=pick.risk_tier,
                terminal_spot=terminal, realized_pnl=pnl, win=1 if win else 0,
                evaluated_at=datetime.now(UTC),
            ))
        n += 1
    return n


def _group(label: str, rows: Sequence) -> OutcomeGroup:  # type: ignore[type-arg]
    count = len(rows)
    if count == 0:
        return OutcomeGroup(label, 0, 0.0, 0.0, 0.0, 0.0)
    preds = [r.predicted_prob_profit for r in rows if r.predicted_prob_profit is not None]
    total_pnl = sum(r.realized_pnl for r in rows)
    return OutcomeGroup(
        label=label,
        count=count,
        win_rate=sum(r.win for r in rows) / count,
        mean_pred_pop=(sum(preds) / len(preds)) if preds else 0.0,
        total_pnl=total_pnl,
        avg_pnl=total_pnl / count,
    )


def outcomes_report(engine: Engine) -> OutcomesReport:
    """Aggregate pick_outcomes: overall, by strategy, by risk-tier."""
    with engine.connect() as conn:
        rows = conn.execute(select(outcomes_t)).fetchall()
    strategies = sorted({r.strategy for r in rows})
    tiers = sorted({r.risk_tier for r in rows if r.risk_tier is not None})
    return OutcomesReport(
        overall=_group("overall", rows),
        by_strategy={s: _group(s, [r for r in rows if r.strategy == s]) for s in strategies},
        by_risk_tier={t: _group(t, [r for r in rows if r.risk_tier == t]) for t in tiers},
    )
