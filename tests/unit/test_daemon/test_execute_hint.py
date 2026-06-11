"""execute_hint_for lookup (IBK-126) — needs the daemon_context fixture."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import insert

from optionsbot.daemon.alert_pipeline import execute_hint_for
from optionsbot.daemon.context import DaemonContext
from optionsbot.storage.schema import snapshots, strategy_scores

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)


def test_execute_hint_for_looks_up_score_row(daemon_context: DaemonContext) -> None:
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(conn.execute(
            insert(snapshots).values(symbol="SPY", ts=NOW, spot=600.0)
        ).inserted_primary_key[0])
        score_id = int(conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snapshot_id, strategy="bull_put_spread", score=78.0,
                rationale="t", legs_json=[], suggestion_json={},
            )
        ).inserted_primary_key[0])

    daemon_context.settings.execution.enabled = True
    assert execute_hint_for(daemon_context, snapshot_id, "bull_put_spread") == (
        f"/execute {score_id}"
    )
    assert execute_hint_for(daemon_context, snapshot_id, "iron_condor") is None

    daemon_context.settings.execution.enabled = False
    assert execute_hint_for(daemon_context, snapshot_id, "bull_put_spread") is None
