"""Small durable health record shared by the daemon and acceptance job."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".local" / "state" / "optionsbot" / "operational_state.json"
log = logging.getLogger(__name__)


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"version": 1}
    return data if isinstance(data, dict) else {"version": 1}


def _write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def record_daemon_started(
    *, now: datetime | None = None, path: Path = DEFAULT_PATH
) -> None:
    try:
        at = now or datetime.now(UTC)
        data = _load(path)
        data["daemon"] = {"started_at": at.isoformat(), "broker_connected": True}
        _write(data, path)
    except Exception as exc:  # noqa: BLE001 -- evidence must never affect trading
        log.warning("operational daemon-state write failed: %s", type(exc).__name__)


def record_reconcile(
    summary: Any,
    *,
    phase: str,
    now: datetime | None = None,
    path: Path = DEFAULT_PATH,
) -> None:
    try:
        at = now or datetime.now(UTC)
        data = _load(path)
        data["reconcile"] = {
            "at": at.isoformat(),
            "phase": phase,
            "ok": True,
            "adopted": int(summary.adopted),
            "foreign": int(summary.foreign),
            "fills_replayed": int(summary.fills_replayed),
            "resolved": int(summary.resolved),
            "mismatches": int(summary.mismatches),
            "orphan_positions": int(summary.orphan_positions),
        }
        _write(data, path)
    except Exception as exc:  # noqa: BLE001 -- evidence must never affect trading
        log.warning("operational reconcile-state write failed: %s", type(exc).__name__)


def record_reconcile_failure(
    *,
    phase: str,
    error_type: str,
    now: datetime | None = None,
    path: Path = DEFAULT_PATH,
) -> None:
    try:
        at = now or datetime.now(UTC)
        data = _load(path)
        data["reconcile"] = {
            "at": at.isoformat(),
            "phase": phase,
            "ok": False,
            "error_type": error_type,
        }
        _write(data, path)
    except Exception as exc:  # noqa: BLE001 -- evidence must never affect trading
        log.warning("operational reconcile-failure write failed: %s", type(exc).__name__)
