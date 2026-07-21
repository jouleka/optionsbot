"""Daily outcome-accrual tick: evaluate newly-expired picks from their close (IBK-117)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import nyse_session_close_utc, nyse_session_date
from optionsbot.ibkr.history import HistoryClient
from optionsbot.validation.outcomes import evaluate_pending, make_close_fetcher


async def run_outcomes_tick(
    context: DaemonContext, *, now: datetime | None = None
) -> int:
    """Evaluate every unevaluated expired pick from its terminal close; return the count.

    Reads historical closes (no market-hours gate needed); serialized on the ibkr_lock so
    it never contends with the scan/manage ticks for the market-data line.

    NOTE: the lock is held across the whole evaluate_pending loop (one get_history per
    pending pick). That's a deliberate tradeoff -- runs daily, pending is normally small,
    and the parquet history cache + max_instances=1/coalesce bound it. A large first-run
    backlog could briefly delay other ticks; cap picks-per-tick here if that ever bites."""
    history = HistoryClient(context.ibkr, context.resolver)
    fetch = make_close_fetcher(history)
    checked_at = now or datetime.now(UTC)
    session_date = nyse_session_date(checked_at)
    close = nyse_session_close_utc(checked_at)
    # load_unevaluated_expired intentionally uses expiry < evaluation_date.
    # Advance one day only after today's official close so same-day picks are
    # never judged on an incomplete bar, but become feedback that evening.
    evaluation_date = (
        session_date + timedelta(days=1)
        if close is not None and checked_at.astimezone(UTC) > close
        else session_date
    )
    async with context.ibkr_lock:
        return await evaluate_pending(context.engine, fetch, evaluation_date)
