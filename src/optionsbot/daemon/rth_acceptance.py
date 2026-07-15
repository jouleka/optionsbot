"""Automated RTH acceptance plus durable Phase-1 paper-soak evidence."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select

from optionsbot.config import load_settings
from optionsbot.daemon.event_webhook import EventWebhookPublisher
from optionsbot.daemon.market_hours import (
    is_market_open,
    nyse_session_date,
    nyse_session_start_utc,
)
from optionsbot.daemon.operational_state import DEFAULT_PATH as OPERATIONAL_STATE_PATH
from optionsbot.daemon.soak_evidence import append_result
from optionsbot.hermes_overlay import breaker_report, correctness_report
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import fills, orders, scan_runs, snapshots

RESULT_PATH = Path.home() / ".local" / "state" / "optionsbot" / "rth_acceptance.json"
REQUIRED_SERVICES = (
    "optionsbot-gateway.service",
    "optionsbot-daemon.service",
    "hermes-gateway-optionsbot.service",
)
LIVE_DATA_ERROR_PATTERNS = (
    re.compile(r"\b(?:error|warning)\s*[:=]?\s*(?:10197|10089|162)\b", re.IGNORECASE),
    re.compile(r"\berrorcode\s*=\s*(?:10197|10089|162)\b", re.IGNORECASE),
    re.compile(r"\brequested market data is not subscribed\b", re.IGNORECASE),
)


def _is_live_data_error(value: object) -> bool:
    text = str(value)
    return any(pattern.search(text) for pattern in LIVE_DATA_ERROR_PATTERNS)


def _run_output(command: list[str]) -> str:
    result = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout if result.returncode == 0 else ""


def _service_details(name: str) -> dict[str, Any]:
    output = _run_output(
        [
            "systemctl",
            "show",
            name,
            "--property=ActiveState",
            "--property=SubState",
            "--property=NRestarts",
            "--property=ExecMainStartTimestamp",
            "--property=Result",
            "--no-pager",
        ]
    )
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    return {
        "active": values.get("ActiveState") == "active",
        "sub_state": values.get("SubState"),
        "restart_count": int(values.get("NRestarts", "0") or 0),
        "started_at": values.get("ExecMainStartTimestamp") or None,
        "result": values.get("Result") or None,
    }


def _parse_systemd_timestamp(value: str | None) -> datetime | None:
    if not value or value == "n/a":
        return None
    try:
        return datetime.strptime(value, "%a %Y-%m-%d %H:%M:%S %Z").replace(tzinfo=UTC)
    except ValueError:
        return None


def _journal_metrics(since: datetime) -> dict[str, Any]:
    output = _run_output(
        [
            "journalctl",
            "-u",
            "optionsbot-daemon.service",
            "--since",
            since.isoformat(),
            "--output=cat",
            "--no-pager",
            "-n",
            "10000",
        ]
    )
    lowered = output.lower()
    live_error_lines = [
        line[:500]
        for line in output.splitlines()
        if _is_live_data_error(line)
    ]
    return {
        "gateway_connections": lowered.count("connected to ib gateway"),
        "gateway_connect_failures": lowered.count("ib gateway connect failed"),
        "startup_reconciles": lowered.count("startup reconcile:"),
        "disconnect_events": sum(
            lowered.count(marker) for marker in ("warning 1100", "error 1100")
        ),
        "reconnect_events": sum(
            lowered.count(marker)
            for marker in ("warning 1101", "warning 1102", "error 1101", "error 1102")
        ),
        "live_data_errors": live_error_lines[-20:],
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _order_metrics(conn: Any, session_start: datetime) -> dict[str, Any]:
    status_rows = conn.execute(
        select(orders.c.intent, orders.c.status, func.count())
        .where(orders.c.staged_ts >= session_start)
        .group_by(orders.c.intent, orders.c.status)
    ).all()
    by_status: Counter[str] = Counter()
    by_intent: Counter[str] = Counter()
    for intent, status, count in status_rows:
        by_status[str(status)] += int(count)
        by_intent[str(intent)] += int(count)
    fill_count = int(
        conn.execute(
            select(func.count()).select_from(fills).where(fills.c.ts >= session_start)
        ).scalar_one()
    )
    return {
        "orders": sum(by_status.values()),
        "by_status": dict(sorted(by_status.items())),
        "by_intent": dict(sorted(by_intent.items())),
        "execution_fills": fill_count,
    }


def evaluate(now: datetime | None = None) -> dict[str, Any]:
    settings = load_settings()
    now = now or datetime.now(UTC)
    session_start = nyse_session_start_utc(now)
    services = {name: _service_details(name) for name in REQUIRED_SERVICES}
    checks: dict[str, Any] = {
        "market_open": is_market_open(now),
        "services": services,
    }
    reasons: list[str] = []
    if not checks["market_open"]:
        reasons.append("NYSE is not open")
    inactive = [name for name, detail in services.items() if not detail["active"]]
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
            order_metrics = _order_metrics(conn, session_start)
        overlay = breaker_report(engine)
        overlay["correctness"] = correctness_report(engine)
    finally:
        engine.dispose()

    scan_recent = bool(scan and _utc(scan.finished) >= cutoff)
    snapshot_recent = bool(snapshot and _utc(snapshot.ts) >= cutoff)
    raw_errors = scan.errors_json if scan else []
    if raw_errors is None:
        scan_errors: list[Any] = []
    elif isinstance(raw_errors, list):
        scan_errors = list(raw_errors)
    else:
        scan_errors = [raw_errors]
    scan_live_errors = [
        str(error)[:500]
        for error in scan_errors
        if _is_live_data_error(error)
    ]
    journal = _journal_metrics(session_start)
    live_errors = scan_live_errors + list(journal["live_data_errors"])
    checks["scan"] = {
        "recent": scan_recent,
        "finished": _utc(scan.finished).isoformat() if scan else None,
        "tickers_scanned": scan.tickers_scanned if scan else 0,
        "error_count": len(scan_errors),
        "live_data_errors": scan_live_errors,
    }
    checks["snapshot"] = {
        "recent": snapshot_recent,
        "ts": _utc(snapshot.ts).isoformat() if snapshot else None,
        "symbol": snapshot.symbol if snapshot else None,
        "spot": snapshot.spot if snapshot else None,
    }
    checks["orders"] = order_metrics
    checks["hermes_overlay"] = overlay
    checks["journal"] = journal

    operational = _load_json(OPERATIONAL_STATE_PATH)
    reconcile = operational.get("reconcile", {})
    reconcile_at: datetime | None = None
    try:
        reconcile_at = datetime.fromisoformat(str(reconcile.get("at")))
        if reconcile_at.tzinfo is None:
            reconcile_at = reconcile_at.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        reconcile_at = None
    reconcile_recent = bool(reconcile_at and now - reconcile_at <= timedelta(minutes=15))
    reconcile_clean = bool(
        reconcile.get("ok") is True
        and int(reconcile.get("mismatches", 0)) == 0
        and int(reconcile.get("orphan_positions", 0)) == 0
    )
    checks["reconcile"] = {
        **reconcile,
        "recent": reconcile_recent,
        "clean": reconcile_clean,
    }

    gateway_started = _parse_systemd_timestamp(
        services["optionsbot-gateway.service"].get("started_at")
    )
    restart_observed = bool(gateway_started and session_start <= gateway_started <= now)
    restart_survived = bool(
        restart_observed
        and services["optionsbot-daemon.service"]["active"]
        and reconcile_recent
        and reconcile_clean
        and reconcile_at is not None
        and gateway_started is not None
        and reconcile_at >= gateway_started
    )
    checks["gateway_restart"] = {
        "observed_this_session": restart_observed,
        "gateway_started_at": gateway_started.isoformat() if gateway_started else None,
        "survived": restart_survived,
    }

    if not scan_recent:
        reasons.append("no completed non-empty scan in the last 40 minutes")
    if not snapshot_recent:
        reasons.append("no positive-price snapshot in the last 40 minutes")
    if live_errors:
        reasons.append("IBKR reported live-data subscription errors")
    if not reconcile_recent:
        reasons.append("no successful recent reconciliation evidence")
    elif not reconcile_clean:
        reasons.append("latest reconciliation is not clean")
    if restart_observed and not restart_survived:
        reasons.append("Gateway restart was observed but not proven recovered")

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
    previous = _load_json(RESULT_PATH)
    _write_result(result)
    if not result["checks"]["market_open"]:
        return 0  # weekend/holiday/DST-safe: wait for the next actual session
    ledger = append_result(result)
    if previous.get("session") == result["session"] and previous.get("passed") is True:
        return 0  # the 10:20 retry is silent after a 09:50 pass
    settings = load_settings()
    publisher = EventWebhookPublisher(settings.hermes_webhook)
    if result["passed"]:
        summary = "RTH paper acceptance passed; daily soak evidence recorded"
        severity = "info"
    else:
        summary = "RTH paper acceptance failed: " + "; ".join(result["reasons"])
        severity = "critical"
    if publisher.enabled:
        await publisher.deliver(
            "rth-acceptance",
            summary,
            severity=severity,
            details={"checks": result["checks"], "soak": ledger["summary"]},
        )
    return 0 if result["passed"] else 1


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
