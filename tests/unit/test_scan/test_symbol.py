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


async def test_scan_symbol_passes_spot_and_strike_window_to_get_chain(
    mock_ibkr_for_scan: object, scan_engine: object, scan_settings: object
) -> None:
    """scan_symbol feeds the underlying spot + configured strike window into
    get_chain so it can bound the fetch to near-ATM strikes."""
    import optionsbot.scan.symbol as symbol_mod

    await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    get_chain = symbol_mod.ChainClient.return_value.get_chain  # type: ignore[attr-defined]
    get_chain.assert_awaited_once()
    kwargs = get_chain.await_args.kwargs
    assert kwargs["underlying_price"] == 400.0  # fake_stock_quote.mid
    assert kwargs["strike_band_pct"] == scan_settings.scan.strike_band_pct  # type: ignore[attr-defined]
    assert kwargs["max_strikes_per_side"] == scan_settings.scan.max_strikes_per_side  # type: ignore[attr-defined]


async def test_scan_symbol_skips_scoring_when_chain_has_no_option_data(
    monkeypatch, mock_ibkr_for_scan, scan_engine, scan_settings  # type: ignore[no-untyped-def]
) -> None:
    """If no chain leg has IV/greeks (e.g. missing OPRA option market data),
    scan_symbol must NOT run the scorers -- strategies built off the directional
    view alone would carry no real strikes/pricing and be misleading."""
    from typing import cast
    from unittest.mock import MagicMock

    import optionsbot.scan.symbol as symbol_mod
    from optionsbot.ibkr.types import OptionChainLeg, OptionRight

    expiry = (date.today() + timedelta(days=45)).strftime("%Y%m%d")
    nodata_chain = [
        OptionChainLeg(
            symbol="SPY", expiry=expiry, strike=k, right=cast("OptionRight", r),
            bid=None, ask=None, iv=None, delta=None, gamma=None,
            theta=None, vega=None, open_interest=None, volume=None,
        )
        for k in (400.0, 405.0, 410.0)
        for r in ("C", "P")
    ]
    symbol_mod.ChainClient.return_value.get_chain.return_value = nodata_chain  # type: ignore[attr-defined]
    spy_score = MagicMock(return_value=())
    monkeypatch.setattr(symbol_mod, "score_all", spy_score)

    result = await scan_symbol("SPY", mock_ibkr_for_scan, scan_engine, scan_settings)  # type: ignore[arg-type]

    spy_score.assert_not_called()
    assert result.scored == ()
