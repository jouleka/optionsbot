"""Durable per-session paper-soak evidence ledger."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".local" / "state" / "optionsbot" / "soak_evidence.json"
TARGET_PASS_SESSIONS = 10
MAX_SESSION_RECORDS = 40
MAX_ATTEMPTS_PER_SESSION = 4


def load_ledger(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"version": 1, "target_pass_sessions": TARGET_PASS_SESSIONS, "sessions": {}}
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
        return {"version": 1, "target_pass_sessions": TARGET_PASS_SESSIONS, "sessions": {}}
    return data


def _summary(sessions: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(sessions.items())
    passed = [day for day, item in ordered if item.get("passed") is True]
    restart_survived = [
        day for day, item in ordered if item.get("restart_survived") is True
    ]
    consecutive = 0
    for _, item in reversed(ordered):
        if item.get("passed") is not True:
            break
        consecutive += 1
    return {
        "recorded_sessions": len(ordered),
        "passed_sessions": len(passed),
        "restart_survived_sessions": len(restart_survived),
        "consecutive_passed_sessions": consecutive,
        "target_pass_sessions": TARGET_PASS_SESSIONS,
        "phase1_soak_ready": (
            len(passed) >= TARGET_PASS_SESSIONS
            and len(restart_survived) >= TARGET_PASS_SESSIONS
            and consecutive >= TARGET_PASS_SESSIONS
        ),
    }


def append_result(
    result: dict[str, Any], *, path: Path = DEFAULT_PATH
) -> dict[str, Any]:
    ledger = load_ledger(path)
    sessions: dict[str, Any] = ledger["sessions"]
    session = str(result["session"])
    existing = sessions.get(session, {})
    attempts = list(existing.get("attempts", []))
    attempts.append(result)
    attempts = attempts[-MAX_ATTEMPTS_PER_SESSION:]
    passed = bool(existing.get("passed")) or bool(result.get("passed"))
    restart_survived = bool(existing.get("restart_survived")) or bool(
        result.get("checks", {}).get("gateway_restart", {}).get("survived")
    )
    sessions[session] = {
        "passed": passed,
        "restart_survived": restart_survived,
        "last_checked_at": result["checked_at"],
        "attempts": attempts,
    }
    while len(sessions) > MAX_SESSION_RECORDS:
        del sessions[sorted(sessions)[0]]
    ledger["version"] = 1
    ledger["target_pass_sessions"] = TARGET_PASS_SESSIONS
    ledger["summary"] = _summary(sessions)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    return ledger
