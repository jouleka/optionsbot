"""Validation helpers for trusted daemon evidence handed to Hermes."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta


def review_evidence_ready(
    evidence: object,
    *,
    score_id: int,
    now: datetime,
    max_age_minutes: int,
) -> bool:
    """Return whether an immutable daemon-authored packet is usable for review."""
    if not isinstance(evidence, dict):
        return False
    if (
        evidence.get("schema_version") != 1
        or evidence.get("source") != "trusted_daemon"
        or evidence.get("score_id") != score_id
        or evidence.get("ready") is not True
        or evidence.get("readiness_issues") != []
    ):
        return False
    captured_raw = evidence.get("captured_at")
    if not isinstance(captured_raw, str):
        return False
    try:
        captured = datetime.fromisoformat(captured_raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=UTC)
    age = now.astimezone(UTC) - captured.astimezone(UTC)
    if age < -timedelta(minutes=1) or age > timedelta(minutes=max_age_minutes):
        return False
    quotes = evidence.get("option_quotes")
    account = evidence.get("account")
    risk = evidence.get("risk")
    if not isinstance(quotes, list) or not quotes:
        return False
    if not isinstance(account, dict) or not isinstance(risk, dict):
        return False
    for field in ("net_liquidation_usd", "buying_power", "available_funds"):
        value = account.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return (
        risk.get("execution_allowed") is True
        and risk.get("paper_only") is True
        and risk.get("entry_loss_guard_allowed") is True
    )


def snapshot_ready_for_auto(raw: object) -> bool:
    """Require live data plus mature IV history or the explicit HV-rank proxy."""
    if not isinstance(raw, dict) or raw.get("delayed") is not False:
        return False
    if raw.get("warming_up") is False:
        return True
    return raw.get("warming_up") is True and raw.get("iv_rank_is_proxy") is True
