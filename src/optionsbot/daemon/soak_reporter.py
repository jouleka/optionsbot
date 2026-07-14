"""Root-only YouTrack reporter for the unprivileged soak evidence ledger."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from optionsbot.daemon.rth_acceptance import RESULT_PATH
from optionsbot.daemon.soak_evidence import DEFAULT_PATH as SOAK_PATH

ISSUES = ("IBK-137", "IBK-138")
DEFAULT_BASE_URL = "https://tracker.example.invalid"


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        return json.loads(response.read())


def _already_reported(base: str, token: str, issue: str, marker: str) -> bool:
    fields = urllib.parse.quote("comments(text)", safe="(),")
    data = _request(f"{base}/api/issues/{issue}?fields={fields}", token)
    return any(marker in str(comment.get("text", "")) for comment in data.get("comments", []))


def _format_digest(result: dict[str, Any], ledger: dict[str, Any], marker: str) -> str:
    checks = result.get("checks", {})
    summary = ledger.get("summary", {})
    scan = checks.get("scan", {})
    restart = checks.get("gateway_restart", {})
    reconcile = checks.get("reconcile", {})
    orders = checks.get("orders", {})
    status = "PASS" if result.get("passed") else "FAIL"
    reasons = "; ".join(result.get("reasons", [])) or "none"
    return (
        f"{marker}\n"
        f"Automated Phase-1 paper-soak evidence — {result.get('session')}: {status}\n\n"
        f"- Checked: {result.get('checked_at')}\n"
        f"- Scan: {scan.get('tickers_scanned', 0)} tickers; "
        f"recent={scan.get('recent')}; errors={scan.get('error_count', 0)}\n"
        f"- Reconcile: recent={reconcile.get('recent')}; clean={reconcile.get('clean')}; "
        f"mismatches={reconcile.get('mismatches', 0)}; "
        f"orphan_positions={reconcile.get('orphan_positions', 0)}\n"
        f"- Daily Gateway restart: observed={restart.get('observed_this_session')}; "
        f"survived={restart.get('survived')}\n"
        f"- Paper lifecycle: orders={orders.get('orders', 0)}; "
        f"fills={orders.get('execution_fills', 0)}; "
        f"statuses={json.dumps(orders.get('by_status', {}), sort_keys=True)}\n"
        f"- Failure reasons: {reasons}\n\n"
        f"Cumulative: {summary.get('passed_sessions', 0)}/"
        f"{summary.get('target_pass_sessions', 10)} passed sessions; "
        f"{summary.get('restart_survived_sessions', 0)} restart-survival sessions; "
        f"consecutive={summary.get('consecutive_passed_sessions', 0)}; "
        f"Phase-1 soak ready={summary.get('phase1_soak_ready', False)}.\n\n"
        "This reporter records evidence only; it does not change ticket state or trading mode."
    )


def run() -> int:
    token = (
        os.getenv("YOUTRACK_API_TOKEN", "").strip()
        or os.getenv("YOUTRACK_TOKEN", "").strip()
    )
    if not token:
        raise RuntimeError("YouTrack token is not configured")
    base = os.getenv("YOUTRACK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    result = _load(RESULT_PATH)
    ledger = _load(SOAK_PATH)
    session = str(result.get("session", ""))
    if not session or not result.get("checks", {}).get("market_open"):
        return 0
    if session not in ledger.get("sessions", {}):
        return 0
    marker = f"[optionsbot-soak:{session}:v1]"
    digest = _format_digest(result, ledger, marker)
    failures: list[str] = []
    for issue in ISSUES:
        try:
            if _already_reported(base, token, issue, marker):
                continue
            _request(
                f"{base}/api/issues/{issue}/comments?fields=id",
                token,
                method="POST",
                payload={"text": digest},
            )
        except (OSError, ValueError, urllib.error.URLError) as exc:
            failures.append(f"{issue}:{type(exc).__name__}")
    if failures:
        raise RuntimeError("YouTrack soak reporting failed: " + ",".join(failures))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
