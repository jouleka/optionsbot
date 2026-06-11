"""Exit-trigger evaluation for bot-opened positions (IBK-129).

Pure function over signed credit-positive net space: entry_net is what
opening netted per unit (positive = credit collected, negative = debit
paid); current_net is what opening the SAME structure nets right now.
pnl_per_unit = entry_net − current_net works for both directions:

  credit: collected 1.20, reopens at 0.60 → pnl +0.60 = kept 50% → TP
  debit:  paid 2.00 (−2.00), now worth 3.00 (−3.00) → pnl +1.00 = +50% → TP

Thresholds come from ManageSettings (the alert engine's numbers, IBK-113/
114/116) so alerts and auto-exits can never disagree; the soft stop is OFF
by default (execution.exit_stop_enabled) because the defined-risk width is
the real stop — gaps blow through soft stops anyway.
"""

from __future__ import annotations

from optionsbot.config import Settings


def evaluate_exit(
    *,
    entry_net: float,
    current_net: float | None,
    dte: int,
    settings: Settings,
) -> str | None:
    """Return a human-readable close reason, or None to keep holding."""
    execution = settings.execution
    manage = settings.manage

    # The expiry guard is unconditional — assignment/pin risk dwarfs any
    # remaining theta. Checked before the quote-dependent rules so a missing
    # quote can never block the force-close path in the runner.
    if dte <= execution.expiry_guard_dte:
        return f"expiry guard ({dte} DTE)"

    if current_net is None:
        return None
    basis = abs(entry_net)
    if basis <= 0:
        return None
    pnl = entry_net - current_net

    if entry_net > 0:  # credit structure
        if pnl >= manage.take_profit_pct * basis:
            return f"take-profit ({pnl / basis * 100:.0f}% of credit kept)"
        if execution.exit_stop_enabled and pnl <= -(manage.stop_loss_mult * basis):
            return f"soft stop (loss {abs(pnl) / basis:.1f}x credit)"
    else:  # debit structure
        if pnl >= manage.debit_take_profit_pct * basis:
            return f"take-profit (+{pnl / basis * 100:.0f}% on debit)"
        if execution.exit_stop_enabled and pnl <= -(manage.debit_stop_pct * basis):
            return f"soft stop (-{abs(pnl) / basis * 100:.0f}% of debit)"

    if dte <= manage.manage_dte:
        return f"time exit ({dte} DTE)"
    return None
