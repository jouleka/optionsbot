"""Tests for scan_symbol."""

from __future__ import annotations

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
