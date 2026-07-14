"""Automated regular-trading-hours paper-data acceptance check."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select

from optionsbot.config import load_settings
from optionsbot.daemon.event_webhook import EventWebhookPublisher
from optionsbot.daemon.market_hours import is_market_open, nyse_session_date
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import scan_runs, snapshots

RESULT_PATH = Path.home() / ".local" / "state" / "optionsbot" / "rth_acceptance.json"
REQUIRED_SERVICES = (
    "optionsbot-gateway.service",
    "optionsbot-daemon.service",
    "hermes-gateway-optionsbot.service",
)
LIVE_DATA_ERROR_MARKERS = ("10197", "10089", "requested market data is not subscribed")


def _service_active(name: str) -> bool:
    result = subprocess.run(  # noqa: S603
        ["systemctl", "is-active", "--quiet", name],
        check=False,
        timeout=10,
    )
    return result.returncode == 0


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def evaluate(now: datetime | None = None) -> dict[str, Any]:
    settings = load_settings()
    now = now or datetime.now(UTC)
    checks: dict[str, Any] = {
        "market_open": is_market_open(now),
        "services": {name: _service_active(name) for name in REQUIRED_SERVICES},
    }
    reasons: list[str] = []
    if not checks["market_open"]:
        reasons.append("NYSE is not open")
    inactive = [name for name, active in checks["services"].items() if not active]
    if inactive:
        reasons.append("inactive services: " + ", ".join(inactive))

    cutoff = now - timedelta(minutes=40)
    engine = create_engine_for_path(settings.storage.db_path)
    try:
        with engine.connect() as conn:
            scan = conn.execute(
                select(scan_runs)
                .where(scan_runs.c.finished.is_not(None))
                .where(scan_runs.c.tickers_scanned > 0)
                .order_by(desc(scan_runs.c.finished))
                .limit(1)
            ).first()
            snapshot = conn.execute(
                select(snapshots.c.ts, snapshots.c.symbol, snapshots.c.spot)
                .where(snapshots.c.spot > 0)
                .order_by(desc(snapshots.c.ts))
                .limit(1)
            ).first()
    finally:
        engine.dispose()

    scan_recent = bool(scan and _utc(scan.finished) >= cutoff)
    snapshot_recent = bool(snapshot and _utc(snapshot.ts) >= cutoff)
    scan_errors = list(scan.errors_json or []) if scan else []
    live_errors = [
        str(error)
        for error in scan_errors
        if any(marker in str(error).lower() for marker in LIVE_DATA_ERROR_MARKERS)
    ]
    checks["scan"] = {
        "recent": scan_recent,
        "finished": _utc(scan.finished).isoformat() if scan else None,
        "tickers_scanned": scan.tickers_scanned if scan else 0,
        "error_count": len(scan_errors),
        "live_data_errors": live_errors,
    }
    checks["snapshot"] = {
        "recent": snapshot_recent,
        "ts": _utc(snapshot.ts).isoformat() if snapshot else None,
        "symbol": snapshot.symbol if snapshot else None,
        "spot": snapshot.spot if snapshot else None,
    }
    if not scan_recent:
        reasons.append("no completed non-empty scan in the last 40 minutes")
    if not snapshot_recent:
        reasons.append("no positive-price snapshot in the last 40 minutes")
    if live_errors:
        reasons.append("IBKR reported live-data subscription errors")

    return {
        "passed": not reasons,
        "checked_at": now.isoformat(),
        "session": nyse_session_date(now).isoformat(),
        "checks": checks,
        "reasons": reasons,
    }


def _write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(RESULT_PATH)


async def run() -> int:
    result = evaluate()
    previous: dict[str, Any] = {}
    if RESULT_PATH.exists():
        try:
            previous = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    _write_result(result)
    if not result["checks"]["market_open"]:
        return 0  # weekend/holiday/DST-safe: wait for the next actual session
    if previous.get("session") == result["session"] and previous.get("passed") is True:
        return 0  # the 10:20 retry is silent after a 09:50 pass
    settings = load_settings()
    publisher = EventWebhookPublisher(settings.hermes_webhook)
    if result["passed"]:
        summary = "RTH paper acceptance passed: services, live scan, and snapshot are healthy"
        severity = "info"
    else:
        summary = "RTH paper acceptance failed: " + "; ".join(result["reasons"])
        severity = "critical"
    if publisher.enabled:
        await publisher.deliver(
            "rth-acceptance", summary, severity=severity, details=result["checks"]
        )
    return 0 if result["passed"] else 1


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
