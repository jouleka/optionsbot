"""Causal shadow-policy accounting for structured Hermes context reviews."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, select

from optionsbot.storage.schema import managed_context_reviews, managed_opportunities


def context_shadow_report(engine: Engine) -> dict[str, Any]:
    """Compare an event-conflict hold policy with scan-admission policy.

    Hermes never changes the actual action.  The only predefined shadow-policy
    disagreement is: when OptionsBot's immutable scan-admission action is
    ``candidate`` and a causally pretrade Hermes response has
    ``event_conflict=true``, compare holding for zero P&L with that same
    opportunity's captured managed-path net P&L.  This measures the shadow
    admission policy, not downstream liquidity, margin, fill, or realized-order
    value.  Agreement receives zero incremental credit.  Context probabilities
    are not treated as profit probabilities and therefore are not calibrated
    here.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                managed_context_reviews.c.id,
                managed_context_reviews.c.opportunity_id,
                managed_context_reviews.c.received_at,
                managed_context_reviews.c.timing,
                managed_context_reviews.c.context_probability,
                managed_context_reviews.c.event_conflict,
                managed_context_reviews.c.model_version,
                managed_context_reviews.c.prompt_version,
                managed_opportunities.c.session,
                managed_opportunities.c.bot_action,
                managed_opportunities.c.bot_decided_at,
                managed_opportunities.c.entry_ts,
                managed_opportunities.c.status,
                managed_opportunities.c.outcome,
                managed_opportunities.c.net_pnl,
                managed_opportunities.c.training_eligible,
                managed_opportunities.c.admission_eligible,
                managed_opportunities.c.shadow_only,
            ).join(
                managed_opportunities,
                managed_opportunities.c.id == managed_context_reviews.c.opportunity_id,
            )
        ).fetchall()

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rows:
        if int(row.admission_eligible) == 1 and int(row.shadow_only) == 0:
            grouped[(str(row.model_version), str(row.prompt_version))].append(row)

    by_critic: dict[str, dict[str, Any]] = {}
    for (model_version, prompt_version), reviews in sorted(grouped.items()):
        pretrade = [row for row in reviews if row.timing == "pretrade"]
        missing_managed_entry = [row for row in pretrade if row.entry_ts is None]
        late_for_managed_path = [
            row
            for row in pretrade
            if row.entry_ts is not None and _utc(row.received_at) >= _utc(row.entry_ts)
        ]
        causal_pre_entry = [
            row
            for row in pretrade
            if row.entry_ts is not None and _utc(row.received_at) < _utc(row.entry_ts)
        ]
        judgeable = [
            row
            for row in causal_pre_entry
            if row.status == "resolved"
            and row.outcome in {"target", "stop", "timeout"}
            and row.net_pnl is not None
            and int(row.training_eligible) == 1
            and row.bot_action in {"candidate", "hold"}
            and row.bot_decided_at is not None
            and _utc(row.bot_decided_at) <= _utc(row.received_at)
        ]
        disagreements = [
            row for row in judgeable if row.bot_action == "candidate" and bool(row.event_conflict)
        ]
        deltas = [-float(row.net_pnl) for row in disagreements]
        avoided_losses = sum(-float(row.net_pnl) for row in disagreements if float(row.net_pnl) < 0)
        missed_profits = sum(float(row.net_pnl) for row in disagreements if float(row.net_pnl) > 0)
        key = f"{model_version}|{prompt_version}"
        by_critic[key] = {
            "model_version": model_version,
            "prompt_version": prompt_version,
            "observations": len(reviews),
            "pretrade_observations": len(pretrade),
            "posttrade_observations": len(reviews) - len(pretrade),
            "causal_pre_managed_entry_observations": len(causal_pre_entry),
            "excluded_pretrade_missing_managed_entry_ts": len(missing_managed_entry),
            "excluded_pretrade_at_or_after_managed_entry": len(late_for_managed_path),
            "judgeable_managed_outcomes": len(judgeable),
            "judgeable_sessions": len({str(row.session) for row in judgeable}),
            "scan_admission_candidate": sum(row.bot_action == "candidate" for row in judgeable),
            "scan_admission_hold": sum(row.bot_action == "hold" for row in judgeable),
            "event_conflict_disagreements": len(disagreements),
            "agreements_zero_incremental_credit": len(judgeable) - len(disagreements),
            "avoided_losses": round(avoided_losses, 2),
            "missed_profits": round(missed_profits, 2),
            "incremental_net_value": round(sum(deltas), 2),
            "mean_incremental_value_per_disagreement": (
                round(sum(deltas) / len(deltas), 2) if deltas else None
            ),
            "probability_observations_not_profit_scored": sum(
                row.context_probability is not None for row in judgeable
            ),
        }

    return {
        "ok": True,
        "observations": len(rows),
        "critics": len(by_critic),
        "by_critic": by_critic,
        "method": {
            "eligible": (
                "timing=pretrade after immutable OptionsBot scan admission, "
                "received_at strictly before the captured managed-path entry_ts, "
                "with a resolved training-eligible target/stop/timeout managed path"
            ),
            "shadow_policy": (
                "hold only when scan-admission action=candidate and "
                "event_conflict=true; Hermes can never turn a hold into a trade"
            ),
            "incremental_value": (
                "shadow hold utility (zero) minus captured managed-path net P&L; "
                "not downstream execution/fill or realized-order value"
            ),
            "agreement_credit": "zero",
            "context_probability": (
                "directional context support only; not probability of profit and "
                "not calibrated against managed outcomes"
            ),
        },
        "authority": "evaluation_only_no_order_or_halt_authority",
    }


def _utc(value: datetime) -> datetime:
    """Restore UTC tzinfo stripped by SQLite's datetime adapter."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
