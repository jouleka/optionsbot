"""Execution-side invariants for opening-range/FVG entries."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Engine, insert

from optionsbot.config import Settings
from optionsbot.execution.engine import _opening_range_entry_error
from optionsbot.storage.schema import orders, snapshots, strategy_scores

NOW = datetime(2026, 7, 31, 14, 0, tzinfo=UTC)  # 10:00 America/New_York


def _plan() -> dict[str, object]:
    return {
        "status": "entry_confirmed",
        "source": "trusted_daemon",
        "signal_id": "2026-07-31:SPY:bull:fvg:respect",
        "direction": "bull",
        "respected_ts": "2026-07-31T13:56:00+00:00",
        "stop_pct": 0.15,
        "target_r": 2.0,
        "target_pct": 0.30,
    }


def _settings() -> Settings:
    settings = Settings()
    settings.scan.opening_range_fvg_enabled = True
    settings.scan.opening_range_entry_window_minutes = 90
    settings.execution.opening_range_max_entries_per_day = 3
    return settings


def test_valid_opening_range_plan_is_accepted(tmp_db: Engine) -> None:
    plan = _plan()
    assert (
        _opening_range_entry_error(
            tmp_db,
            _settings(),
            suggestion={"opening_range_fvg": plan},
            snapshot_raw={"opening_range_fvg": plan},
            strategy="bull_call_spread",
            now=NOW,
        )
        is None
    )


def test_direction_mismatch_is_rejected(tmp_db: Engine) -> None:
    plan = _plan()
    error = _opening_range_entry_error(
        tmp_db,
        _settings(),
        suggestion={"opening_range_fvg": plan},
        snapshot_raw={"opening_range_fvg": plan},
        strategy="bear_put_spread",
        now=NOW,
    )
    assert error is not None and "direction" in error


def test_fourth_opening_range_entry_is_rejected(tmp_db: Engine) -> None:
    with tmp_db.begin() as conn:
        for index in range(3):
            conn.execute(
                insert(orders).values(
                    intent="open",
                    symbol=f"S{index}",
                    strategy="long_call",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=datetime(2026, 7, 31, 13, 40 + index, tzinfo=UTC),
                    reprice_count=0,
                )
            )
    plan = _plan()
    error = _opening_range_entry_error(
        tmp_db,
        _settings(),
        suggestion={"opening_range_fvg": plan},
        snapshot_raw={"opening_range_fvg": plan},
        strategy="long_call",
        now=NOW,
    )
    assert error is not None and "daily entry cap" in error


def test_same_opening_range_signal_cannot_be_consumed_twice(tmp_db: Engine) -> None:
    plan = _plan()
    with tmp_db.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=NOW,
                    spot=500.0,
                    regime_dir="bull",
                    regime_iv="neutral",
                    raw_json={"opening_range_fvg": plan},
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_call_spread",
                    score=80.0,
                    rationale="first use",
                    legs_json=[],
                    suggestion_json={"opening_range_fvg": plan},
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(orders).values(
                strategy_score_id=score_id,
                intent="open",
                symbol="SPY",
                strategy="bull_call_spread",
                legs_json=[],
                quantity=1,
                status="filled",
                staged_ts=datetime(2026, 7, 31, 13, 58, tzinfo=UTC),
                reprice_count=0,
            )
        )

    error = _opening_range_entry_error(
        tmp_db,
        _settings(),
        suggestion={"opening_range_fvg": plan},
        snapshot_raw={"opening_range_fvg": plan},
        strategy="bull_call_spread",
        now=NOW,
    )
    assert error is not None and "already consumed" in error


def test_entry_after_eleven_new_york_is_rejected(tmp_db: Engine) -> None:
    plan = _plan()
    error = _opening_range_entry_error(
        tmp_db,
        _settings(),
        suggestion={"opening_range_fvg": plan},
        snapshot_raw={"opening_range_fvg": plan},
        strategy="long_call",
        now=datetime(2026, 7, 31, 15, 1, tzinfo=UTC),
    )
    assert error is not None and "configured session window" in error


def test_production_length_window_accepts_fresh_afternoon_setup(tmp_db: Engine) -> None:
    plan = _plan()
    plan["respected_ts"] = "2026-07-31T19:19:00+00:00"
    settings = _settings()
    settings.scan.opening_range_entry_window_minutes = 360
    assert (
        _opening_range_entry_error(
            tmp_db,
            settings,
            suggestion={"opening_range_fvg": plan},
            snapshot_raw={"opening_range_fvg": plan},
            strategy="long_call",
            now=datetime(2026, 7, 31, 19, 20, tzinfo=UTC),
        )
        is None
    )
