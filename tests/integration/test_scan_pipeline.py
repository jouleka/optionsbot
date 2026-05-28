"""End-to-end fixture-replay tests for the scan pipeline (IBK-70).

Wires a fixture-driven ``ib_async`` mock into a real ``IBKRClient``, runs the
full ``scan_symbol`` pipeline (history + chain + spot + positions -> analysis
-> scoring -> persist), and asserts the persisted snapshot plus formatted
alert match expectations.

The mock operates one layer deeper than the per-symbol unit tests: rather
than patching ``HistoryClient`` / ``ChainClient`` / etc., we patch the
underlying ``ib_async.IB`` low-level methods so that the adapters' own
response-parsing code executes against the fixture data. This catches
regressions in the adapter glue (e.g. a bug in ``ChainClient._fetch_one``
that the higher-level mocks would hide).

For v1 the fixtures are synthetic (no live IB Gateway was available at
capture time) but realistic enough to exercise every integration seam.
When running the suite against a real paper IB Gateway, regenerate
``tests/fixtures/ibkr/spy_sample.json`` from live responses.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from optionsbot.alerts import format_alert_markdown
from optionsbot.ibkr import IBKRClient
from optionsbot.scan import scan_symbol
from optionsbot.scoring import top_k
from optionsbot.storage.schema import snapshots, strategy_scores
from tests.integration.fixture_loader import build_ib_mock, load_fixture


async def test_scan_pipeline_produces_persisted_snapshot_and_top_strategies(
    integration_engine,
    integration_settings,
) -> None:
    """End-to-end: fixture-driven ``scan_symbol`` -> snapshots row + N
    strategy_scores rows + result.scored has at least one viable strategy.

    Asserts:
    * Exactly one ``snapshots`` row is inserted for "SPY".
    * The spot value persisted matches the fixture's last price (~400).
    * At least one ``strategy_scores`` row is inserted for the snapshot.
    * Every persisted ``suggestion_json`` contains the keys the formatter
      requires (``defined_risk``, ``credit_or_debit``, ``max_loss``,
      ``max_profit``, ``prob_profit``, ``suggested_quantity``).
    * ``result.scored`` is non-empty (at least one applicable strategy).
    """
    spy = load_fixture("spy_sample")
    ib_mock = build_ib_mock(spy)
    client = IBKRClient(
        role="cli",
        settings=integration_settings,
        ib=ib_mock,
        backoff_seconds=(),
    )

    result = await scan_symbol(
        "SPY",
        client,
        integration_engine,
        integration_settings,
    )

    # --- Snapshot row ---
    with integration_engine.connect() as conn:
        snap_rows = conn.execute(select(snapshots)).fetchall()
    assert len(snap_rows) == 1
    snap = snap_rows[0]
    assert snap.symbol == "SPY"
    # Spot is derived from mid(bid, ask) = (399.95 + 400.05) / 2 = 400.0
    assert snap.spot == pytest.approx(400.0, rel=0.01)

    # --- strategy_scores rows ---
    with integration_engine.connect() as conn:
        score_rows = conn.execute(
            select(strategy_scores).where(
                strategy_scores.c.snapshot_id == result.snapshot_id
            )
        ).fetchall()
    assert len(score_rows) >= 1

    for row in score_rows:
        assert 0.0 <= row.score <= 100.0
        sj = row.suggestion_json
        assert sj is not None, f"suggestion_json missing for strategy {row.strategy!r}"
        for key in (
            "defined_risk",
            "credit_or_debit",
            "max_loss",
            "max_profit",
            "prob_profit",
            "suggested_quantity",
        ):
            assert key in sj, (
                f"suggestion_json for {row.strategy!r} is missing key {key!r}: {sj}"
            )

    # --- At least one scored strategy in result ---
    assert len(result.scored) >= 1


async def test_scan_pipeline_renders_alert_for_top_strategy(
    integration_engine,
    integration_settings,
) -> None:
    """End-to-end including the formatter: take the top scored strategy from
    ``scan_symbol``, render it as a Telegram MarkdownV2 alert, and assert the
    output contains the symbol, strategy name, and at least one leg's strike.

    Uses ``threshold=0.0`` for ``top_k`` so that a valid suggestion is always
    returned even when all scores fall below the default alert threshold (e.g.,
    in warming-up / single-element IV-history mode).
    """
    spy = load_fixture("spy_sample")
    ib_mock = build_ib_mock(spy)
    client = IBKRClient(
        role="cli",
        settings=integration_settings,
        ib=ib_mock,
        backoff_seconds=(),
    )

    result = await scan_symbol(
        "SPY",
        client,
        integration_engine,
        integration_settings,
    )

    top = top_k(result.scored, k=1, threshold=0.0)
    assert len(top) == 1, (
        "Expected at least one scored strategy from the fixture data; "
        f"result.scored={result.scored!r}"
    )

    best = top[0]
    text = format_alert_markdown(
        symbol="SPY",
        view=result.view,
        scored=best,
        snapshot_ts=datetime.now(UTC),
    )

    # Symbol appears in header (possibly MD-escaped, but SPY has no special chars).
    assert "SPY" in text

    # Strategy name appears either plain or with underscores escaped (\\_).
    assert best.strategy_name in text or best.strategy_name.replace("_", "\\_") in text

    # At least one leg's strike should appear as an integer in the output.
    if best.suggestion.legs:
        first_strike = best.suggestion.legs[0].strike
        if first_strike is not None:
            assert str(int(first_strike)) in text, (
                f"Expected strike {int(first_strike)} in formatted alert:\n{text}"
            )
