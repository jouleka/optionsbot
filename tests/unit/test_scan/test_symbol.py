"""Tests for scan_symbol."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select

from optionsbot.scan import ScanResult, scan_symbol
from optionsbot.storage.schema import snapshots, strategy_scores


async def test_scan_symbol_returns_scan_result(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    assert isinstance(result, ScanResult)
    assert result.symbol == "SPY"
    assert result.snapshot_id > 0
    assert result.view is not None
    assert isinstance(result.scored, tuple)


async def test_scan_symbol_persists_snapshot(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        rows = conn.execute(select(snapshots)).fetchall()
    assert len(rows) == 1
    assert rows[0].symbol == "SPY"
    assert rows[0].spot == 400.0


async def test_scan_symbol_persists_strategy_scores(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        rows = conn.execute(
            select(strategy_scores).where(
                strategy_scores.c.snapshot_id == result.snapshot_id
            )
        ).fetchall()
    assert len(rows) >= 1
    for r in rows:
        assert 0.0 <= r.score <= 100.0
        assert r.legs_json is not None


async def test_scan_symbol_ensures_ibkr_connected(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]
    mock_ibkr_for_scan.ensure_connected.assert_awaited()  # type: ignore[union-attr]


async def test_scan_symbol_applies_view_override(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    result = await scan_symbol(
        "SPY",
        mock_ibkr_for_scan,  # type: ignore[arg-type]
        scan_engine,  # type: ignore[arg-type]
        scan_settings,  # type: ignore[arg-type]
        view_override=("bear", "high"),
    )
    assert result.view.direction == "bear"
    assert result.view.iv_regime == "high"


async def test_scan_symbol_partial_view_override_only_direction(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """Override one field; the other stays as inferred."""
    result_inferred = await scan_symbol(
        "SPY", mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[arg-type]
    )
    inferred_iv = result_inferred.view.iv_regime

    result = await scan_symbol(
        "SPY",
        mock_ibkr_for_scan,  # type: ignore[arg-type]
        scan_engine,  # type: ignore[arg-type]
        scan_settings,  # type: ignore[arg-type]
        view_override=("bear", None),
    )
    assert result.view.direction == "bear"
    assert result.view.iv_regime == inferred_iv


async def test_scan_symbol_normalizes_nan_hv20_to_none(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    """historical_volatility returns NaN when bars are shorter than window+1;
    scan_symbol must normalize that to None so the snapshots row stores NULL
    and the scoring layer's `hv is None` guard catches it as intended.
    """
    # Replace the history fixture with one shorter than window=20.
    import optionsbot.scan.symbol as symbol_mod
    short_dates = [date(2026, 5, 1) + timedelta(days=i) for i in range(5)]
    short_bars = pd.DataFrame(
        {
            "open": [400.0] * 5,
            "high": [401.0] * 5,
            "low": [399.0] * 5,
            "close": [400.0 + i for i in range(5)],
            "volume": [1_000_000] * 5,
        },
        index=pd.Index(short_dates, name="date"),
    )
    short_history = symbol_mod.HistoryClient.return_value  # type: ignore[attr-defined]
    short_history.get_history.return_value = short_bars

    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    with scan_engine.connect() as conn:  # type: ignore[union-attr]
        row = conn.execute(
            select(snapshots).where(snapshots.c.id == result.snapshot_id)
        ).first()
    assert row is not None
    # Persisted column is NULL (None when read back), NOT NaN.
    assert row.hv20 is None
    # iv_hv_ratio should also be None since hv20 collapsed to None.
    assert row.iv_hv_ratio is None
