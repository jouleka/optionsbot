"""Gateway health paging (IBK-137 Increment 2).

Detects the two conditions under which exit protection silently lapses and
PAGES the human via Telegram, repeating until the condition clears:

- **WEDGED** — the Gateway is connected but its option pipeline is dead
  (``isConnected()`` stays True!): the fingerprint is a scan where a MAJORITY
  of symbols die on the IBK-149 per-symbol budget. Seen 3-4x on 2026-06-29;
  only a Gateway restart clears it.
- **DISCONNECTED** — no Gateway connection during RTH while the ledger holds
  open positions: the dead-man condition from the 2026-06-24 VPS design (§6)
  — TP/stop/DTE protection is DOWN until reconnect, so the human must know.

The monitor itself is a pure state machine (injectable clock, no I/O) so the
enter / re-page / clear transitions are unit-testable; ``page_gateway_health``
is the thin async seam the daemon's scan tick calls, and it never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from optionsbot.config import MonitorSettings
from optionsbot.daemon.market_hours import is_market_open

if TYPE_CHECKING:
    from optionsbot.daemon.context import DaemonContext
    from optionsbot.daemon.scan_runner import ScanRunSummary
    from optionsbot.execution.orders import OrderRecord

log = logging.getLogger(__name__)


def _open_entries(context: DaemonContext) -> list[OrderRecord]:
    """Open ledger positions (filled opens with no filled close). Thin patchable
    wrapper; the import is lazy because exit_runner imports DaemonContext and
    DaemonContext holds a GatewayHealthMonitor -- a module-level import here
    would be circular."""
    from optionsbot.daemon.exit_runner import _open_entries as impl

    return impl(context)

# The per-symbol budget-timeout error suffix. scan_runner IMPORTS this to build
# its error strings, so wedge detection can never silently drift from the
# producer's format. (The constant lives here, not in scan_runner, because
# scan_runner imports DaemonContext which imports this module — the reverse
# import would be circular.)
BUDGET_TIMEOUT_SUFFIX = "TimeoutError (scan budget)"

_ACTION = "Restart IB Gateway (ibgateway.exe) and log in."


def count_budget_timeouts(errors: list[str]) -> int:
    """How many scan errors were IBK-149 per-symbol budget timeouts."""
    return sum(1 for e in errors if e.endswith(BUDGET_TIMEOUT_SUFFIX))


class GatewayHealthMonitor:
    """Pure enter/re-page/clear state machine for gateway-health paging."""

    def __init__(self) -> None:
        self._active_reasons: set[str] = set()
        self._last_page_at: datetime | None = None

    def evaluate(
        self,
        *,
        now: datetime,
        market_open: bool,
        connected: bool,
        tickers_scanned: int,
        budget_timeouts: int,
        open_positions: int,
        settings: MonitorSettings,
    ) -> list[str]:
        """One health evaluation; returns 0 or 1 Telegram messages to send."""
        if not settings.enabled:
            return []

        reasons: set[str] = set()
        # WEDGED: enough budget timeouts to matter AND a majority of the scan
        # failing on them (a couple of slow names must never page).
        if (
            market_open
            and budget_timeouts >= settings.wedge_min_budget_timeouts
            and budget_timeouts > tickers_scanned
        ):
            reasons.add("wedged")
        # DISCONNECTED with open positions during RTH: protection is DOWN.
        if market_open and not connected and open_positions >= 1:
            reasons.add("disconnected")

        if reasons:
            first = not self._active_reasons
            # A CHANGED reason set is effectively a new ENTER (e.g. wedged ->
            # disconnected escalates "degraded" to "DOWN") — page immediately
            # rather than waiting out the re-page window (Opus review MEDIUM).
            changed = reasons != self._active_reasons
            due = self._last_page_at is None or (
                now - self._last_page_at
                >= timedelta(minutes=settings.page_repeat_minutes)
            )
            self._active_reasons = reasons
            if first or changed or due:
                self._last_page_at = now
                return [
                    self._page_text(
                        reasons,
                        tickers_scanned=tickers_scanned,
                        budget_timeouts=budget_timeouts,
                        open_positions=open_positions,
                    )
                ]
            return []

        if self._active_reasons:
            self._active_reasons = set()
            self._last_page_at = None
            return ["✅ Gateway recovered: scans/connection healthy again."]
        return []

    @staticmethod
    def _page_text(
        reasons: set[str],
        *,
        tickers_scanned: int,
        budget_timeouts: int,
        open_positions: int,
    ) -> str:
        parts: list[str] = []
        if "wedged" in reasons:
            parts.append(
                f"🚨 GATEWAY WEDGED: {budget_timeouts} symbol(s) timed out on the "
                f"scan budget vs {tickers_scanned} scanned — option data is not "
                "flowing (connection still shows up). Exit TP/stop protection is "
                "degraded (stale quotes are suppressed)."
            )
        if "disconnected" in reasons:
            parts.append(
                "🚨 GATEWAY DISCONNECTED during market hours — exit protection "
                "is DOWN until it reconnects."
            )
        parts.append(f"Open positions: {open_positions}. {_ACTION}")
        return "\n".join(parts)


async def page_gateway_health(
    context: DaemonContext,
    summary: ScanRunSummary,
    now: datetime | None = None,
) -> None:
    """Evaluate gateway health after a scan tick and send any page. Never raises."""
    try:
        from datetime import UTC
        from datetime import datetime as _dt

        now = now if now is not None else _dt.now(UTC)
        messages = context.monitor.evaluate(
            now=now,
            market_open=is_market_open(now),
            connected=context.ibkr.is_connected,
            tickers_scanned=summary.tickers_scanned,
            budget_timeouts=count_budget_timeouts(summary.errors),
            open_positions=len(_open_entries(context)),
            settings=context.settings.monitor,
        )
        for text in messages:
            await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001 -- health paging must never poison the tick
        log.exception("gateway health paging failed")
