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
    minutes_to_close: float | None = None,
    max_profit_per_unit: float | None = None,
    peak_pnl_per_unit: float | None = None,
) -> str | None:
    """Return a human-readable close reason, or None to keep holding."""
    execution = settings.execution
    manage = settings.manage

    # In explicit exact-0DTE mode, don't let the generic 3-DTE guard close a
    # newly filled same-day position immediately. Hold through the session,
    # while preserving take-profit/stop checks, then flatten before expiry.
    zero_dte_session = execution.zero_dte_only and dte == 0
    if zero_dte_session and (
        minutes_to_close is None
        or minutes_to_close <= execution.zero_dte_force_exit_minutes
    ):
        remaining = "unknown" if minutes_to_close is None else f"{minutes_to_close:.0f}m"
        return f"0DTE close guard ({remaining} to close)"

    # The expiry guard is unconditional — assignment/pin risk dwarfs any
    # remaining theta. Checked before the quote-dependent rules so a missing
    # quote can never block the force-close path in the runner.
    if not zero_dte_session and dte <= execution.expiry_guard_dte:
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
        if zero_dte_session:
            # +50%-of-debit is an arming threshold, not an immediate sale.
            # A bounded spread can run until it captures a meaningful share
            # of its full payoff; otherwise a durable high-water trail protects
            # the winner and tightens as expiry approaches.
            activation = manage.debit_take_profit_pct * basis
            peak = max(pnl, peak_pnl_per_unit if peak_pnl_per_unit is not None else pnl)
            max_profit_target = (
                execution.zero_dte_debit_max_profit_take_pct
                * max_profit_per_unit
                / 100.0
                if max_profit_per_unit is not None and max_profit_per_unit > 0
                else None
            )
            if (
                max_profit_target is not None
                and pnl >= max(activation, max_profit_target)
            ):
                assert max_profit_per_unit is not None
                captured = pnl * 100.0 / max_profit_per_unit
                return (
                    "take-profit (0DTE adaptive: "
                    f"{captured:.0%} of max profit; +{pnl / basis * 100:.0f}% on debit)"
                )
            if peak >= activation:
                assert minutes_to_close is not None  # close guard handled None above
                session_minutes = 390.0
                force_minutes = float(execution.zero_dte_force_exit_minutes)
                progress = min(
                    1.0,
                    max(
                        0.0,
                        (minutes_to_close - force_minutes)
                        / (session_minutes - force_minutes),
                    ),
                )
                early = execution.zero_dte_debit_trail_early_giveback_pct
                late = execution.zero_dte_debit_trail_late_giveback_pct
                giveback = late + (early - late) * progress
                trail_floor = peak * (1.0 - giveback)
                if pnl <= trail_floor:
                    return (
                        "profit trail (0DTE adaptive: "
                        f"+{pnl / basis * 100:.0f}% on debit, "
                        f"peak +{peak / basis * 100:.0f}%, "
                        f"{giveback * 100:.0f}% giveback limit)"
                    )
        elif pnl >= manage.debit_take_profit_pct * basis:
            return f"take-profit (+{pnl / basis * 100:.0f}% on debit)"
        # Exact-0DTE debit spreads are already sized from their full structural
        # max loss and can whipsaw through a percentage-of-premium stop before
        # the directional thesis resolves. Keep take-profit and the mandatory
        # pre-close flatten, but do not confuse an intraday premium drawdown
        # with a disproven same-session call.
        if (
            execution.exit_stop_enabled
            and not zero_dte_session
            and pnl <= -(manage.debit_stop_pct * basis)
        ):
            return f"soft stop (-{abs(pnl) / basis * 100:.0f}% of debit)"

    if not zero_dte_session and dte <= manage.manage_dte:
        return f"time exit ({dte} DTE)"
    return None
