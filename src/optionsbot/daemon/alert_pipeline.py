"""Alert pipeline: enqueue, dispatch, retry sweep (filled in Task 5 / IBK-67)."""

from __future__ import annotations

from optionsbot.daemon.context import DaemonContext
from optionsbot.scoring import ScoredStrategy


async def enqueue_alert(
    context: DaemonContext,
    symbol: str,
    scored: ScoredStrategy,
    snapshot_id: int,
) -> None:
    """Stub: real implementation lands in IBK-67 (Task 5)."""
    return None


async def sweep_retries(context: DaemonContext) -> int:
    """Stub: real implementation lands in IBK-67 (Task 5). Returns count of retries."""
    return 0
