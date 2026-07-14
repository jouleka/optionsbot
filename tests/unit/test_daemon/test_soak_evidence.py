from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import Engine, insert

from optionsbot.config import Settings
from optionsbot.daemon import operational_state, rth_acceptance, soak_reporter
from optionsbot.daemon.operational_state import (
    record_daemon_started,
    record_reconcile,
    record_reconcile_failure,
)
from optionsbot.daemon.soak_evidence import append_result
from optionsbot.storage.schema import scan_runs, snapshots

NOW = datetime(2026, 7, 15, 14, 0, tzinfo=UTC)


def _result(session: str, *, passed: bool, restart: bool) -> dict[str, Any]:
    return {
        "passed": passed,
        "checked_at": f"{session}T14:00:00+00:00",
        "session": session,
        "checks": {"gateway_restart": {"survived": restart}},
        "reasons": [] if passed else ["failed"],
    }


def test_operational_state_records_reconcile_without_exception_text(tmp_path: Path) -> None:
    path = tmp_path / "operational.json"
    record_daemon_started(now=NOW, path=path)
    record_reconcile(
        SimpleNamespace(
            adopted=1,
            foreign=0,
            fills_replayed=2,
            resolved=3,
            mismatches=0,
            orphan_positions=0,
        ),
        phase="startup",
        now=NOW,
        path=path,
    )
    data = json.loads(path.read_text())
    assert data["daemon"]["broker_connected"] is True
    assert data["reconcile"]["fills_replayed"] == 2
    assert data["reconcile"]["ok"] is True

    record_reconcile_failure(
        phase="periodic", error_type="ConnectionError", now=NOW, path=path
    )
    data = json.loads(path.read_text())
    assert data["reconcile"] == {
        "at": NOW.isoformat(),
        "error_type": "ConnectionError",
        "ok": False,
        "phase": "periodic",
    }


def test_operational_state_failure_never_interrupts_daemon(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        operational_state,
        "_write",
        lambda data, path: (_ for _ in ()).throw(OSError("read only")),
    )
    operational_state.record_daemon_started(now=NOW)
    operational_state.record_reconcile_failure(
        phase="periodic", error_type="ConnectionError", now=NOW
    )


def test_soak_ledger_accumulates_sessions_and_retry_can_recover(tmp_path: Path) -> None:
    path = tmp_path / "soak.json"
    append_result(_result("2026-07-15", passed=False, restart=False), path=path)
    ledger = append_result(
        _result("2026-07-15", passed=True, restart=True), path=path
    )
    assert ledger["sessions"]["2026-07-15"]["passed"] is True
    assert len(ledger["sessions"]["2026-07-15"]["attempts"]) == 2
    assert ledger["summary"]["passed_sessions"] == 1
    assert ledger["summary"]["restart_survived_sessions"] == 1


def test_soak_gate_requires_ten_consecutive_passes_and_restarts(tmp_path: Path) -> None:
    path = tmp_path / "soak.json"
    ledger: dict[str, Any] = {}
    for day in range(1, 11):
        ledger = append_result(
            _result(f"2026-07-{day:02d}", passed=True, restart=True), path=path
        )
    assert ledger["summary"]["phase1_soak_ready"] is True


def test_rth_evaluate_collects_clean_soak_evidence(
    tmp_db: Engine, tmp_path: Path, monkeypatch: Any
) -> None:
    with tmp_db.begin() as conn:
        conn.execute(
            insert(scan_runs).values(
                started=NOW - timedelta(minutes=10),
                finished=NOW - timedelta(minutes=5),
                tickers_scanned=3,
                alerts_fired=0,
                errors_json=[],
            )
        )
        conn.execute(
            insert(snapshots).values(
                symbol="SPY", ts=NOW - timedelta(minutes=4), spot=600.0
            )
        )
    operational = tmp_path / "operational.json"
    operational.write_text(
        json.dumps(
            {
                "reconcile": {
                    "at": (NOW - timedelta(minutes=2)).isoformat(),
                    "phase": "periodic",
                    "ok": True,
                    "mismatches": 0,
                    "orphan_positions": 0,
                }
            }
        )
    )
    settings = Settings(storage={"db_path": Path(str(tmp_db.url.database))})
    monkeypatch.setattr(rth_acceptance, "load_settings", lambda: settings)
    monkeypatch.setattr(rth_acceptance, "OPERATIONAL_STATE_PATH", operational)
    monkeypatch.setattr(
        rth_acceptance,
        "_service_details",
        lambda name: {
            "active": True,
            "sub_state": "running",
            "restart_count": 0,
            "started_at": (
                "Wed 2026-07-15 06:35:00 UTC"
                if name == "optionsbot-gateway.service"
                else "Tue 2026-07-14 21:30:00 UTC"
            ),
            "result": "success",
        },
    )
    monkeypatch.setattr(
        rth_acceptance,
        "_journal_metrics",
        lambda since: {
            "gateway_connections": 1,
            "gateway_connect_failures": 0,
            "startup_reconciles": 1,
            "disconnect_events": 0,
            "reconnect_events": 0,
            "live_data_errors": [],
        },
    )

    result = rth_acceptance.evaluate(NOW)

    assert result["passed"] is True
    assert result["checks"]["gateway_restart"]["survived"] is True
    assert result["checks"]["reconcile"]["clean"] is True
    assert result["checks"]["orders"]["orders"] == 0


def test_reporter_posts_idempotency_marked_digest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    result_path = tmp_path / "result.json"
    soak_path = tmp_path / "soak.json"
    result = {
        "passed": True,
        "checked_at": NOW.isoformat(),
        "session": "2026-07-15",
        "checks": {
            "market_open": True,
            "scan": {"tickers_scanned": 3, "recent": True, "error_count": 0},
            "reconcile": {"recent": True, "clean": True},
            "gateway_restart": {"observed_this_session": True, "survived": True},
            "orders": {"orders": 0, "execution_fills": 0, "by_status": {}},
        },
        "reasons": [],
    }
    result_path.write_text(json.dumps(result))
    soak_path.write_text(
        json.dumps(
            {
                "sessions": {"2026-07-15": {"passed": True}},
                "summary": {
                    "passed_sessions": 1,
                    "target_pass_sessions": 10,
                    "restart_survived_sessions": 1,
                    "consecutive_passed_sessions": 1,
                    "phase1_soak_ready": False,
                },
            }
        )
    )
    monkeypatch.setattr(soak_reporter, "RESULT_PATH", result_path)
    monkeypatch.setattr(soak_reporter, "SOAK_PATH", soak_path)
    monkeypatch.setenv("YOUTRACK_API_TOKEN", "test-token")
    posts: list[dict[str, Any]] = []

    def fake_request(
        url: str,
        token: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> Any:
        assert token == "test-token"
        if method == "GET":
            return {"comments": []}
        posts.append(payload or {})
        return {"id": "comment"}

    monkeypatch.setattr(soak_reporter, "_request", fake_request)
    assert soak_reporter.run() == 0
    assert len(posts) == 2
    assert all("[optionsbot-soak:2026-07-15:v1]" in post["text"] for post in posts)
