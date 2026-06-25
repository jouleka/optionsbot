"""Full-auto entry hook (IBK-130): alerted candidates → execute_pick.

Runs the SAME pipeline as Telegram /execute — every gate (freshness, caps,
liquidity, margin, dedup, plus the auto-only earnings and buying-power
gates) applies per pick. Confirm mode never enters here.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from optionsbot.daemon.context import DaemonContext
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.scoring import ScoredStrategy
from optionsbot.storage.schema import strategy_scores

log = logging.getLogger(__name__)


def _score_id_for(context: DaemonContext, snapshot_id: int, strategy: str) -> int | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(strategy_scores.c.id)
            .where(strategy_scores.c.snapshot_id == snapshot_id)
            .where(strategy_scores.c.strategy == strategy)
            .order_by(strategy_scores.c.id.desc())
            .limit(1)
        ).first()
    return int(row.id) if row is not None else None


async def auto_execute_candidates(
    context: DaemonContext,
    candidates: list[tuple[str, ScoredStrategy, int]],
) -> int:
    """Execute alerted candidates in auto mode. Returns submitted count."""
    if context.settings.execution.mode != "auto" or context.order_client is None:
        return 0
    # Imported lazily for the same commands.py reason: the engine pulls
    # daemon.market_hours, and a module-level import would close a cycle.
    from optionsbot.execution import engine as execution_engine

    walk_md = (
        MarketDataClient(context.exec_ibkr, context.resolver)
        if context.exec_ibkr is not None
        else None
    )
    submitted = 0
    log.info("auto-execute pass: %d candidate(s)", len(candidates))
    for symbol, scored, snapshot_id in candidates:
        try:
            score_id = _score_id_for(context, snapshot_id, scored.strategy_name)
            if score_id is None:
                # Silent-skip guard: without this, a pick that can't be resolved
                # to a strategy_scores row vanished with no trace in the journal.
                log.warning(
                    "auto-execute skip %s/%s: no score_id for snapshot %s",
                    symbol, scored.strategy_name, snapshot_id,
                )
                continue
            deps = execution_engine.ExecutionDeps(
                engine=context.engine,
                settings=context.settings,
                order_client=context.order_client,
                md=MarketDataClient(context.ibkr, context.resolver),
                positions=PositionsClient(context.ibkr),
                ibkr_lock=context.ibkr_lock,
                walk_md=walk_md,
                walk_tasks=context.walk_tasks,
            )
            outcome = await execution_engine.execute_pick(deps, score_id)
            if outcome.ok:
                submitted += 1
            # Mirror the Telegram outcome into the journal so "why didn't it
            # trade?" is answerable from `journalctl` alone.
            log.info(
                "auto-execute %s/%s -> ok=%s | %s",
                symbol, scored.strategy_name, outcome.ok,
                outcome.message.replace("\n", " "),
            )
            await _send(
                context,
                f"🤖 auto-execute {symbol} {scored.strategy_name}:\n{outcome.message}",
            )
        except Exception:  # noqa: BLE001 -- one candidate must not starve the rest
            log.exception("auto-execute failed for %s/%s", symbol, scored.strategy_name)
    return submitted


async def _send(context: DaemonContext, text: str) -> None:
    try:
        await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001
        log.exception("auto-execute notification failed")
