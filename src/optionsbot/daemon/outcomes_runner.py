"""Daily outcome-accrual tick: evaluate newly-expired picks from their close (IBK-117)."""

from __future__ import annotations

from datetime import date

from optionsbot.daemon.context import DaemonContext
from optionsbot.ibkr.history import HistoryClient
from optionsbot.validation.outcomes import evaluate_pending, make_close_fetcher


async def run_outcomes_tick(context: DaemonContext) -> int:
    """Evaluate every unevaluated expired pick from its terminal close; return the count.

    Reads historical closes (no market-hours gate needed); serialized on the ibkr_lock so
    it never contends with the scan/manage ticks for the market-data line."""
    history = HistoryClient(context.ibkr, context.resolver)
    fetch = make_close_fetcher(history)
    async with context.ibkr_lock:
        return await evaluate_pending(context.engine, fetch, date.today())
