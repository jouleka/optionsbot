"""Live paper order smoke test (IBK-125).

@pytest.mark.live -- skipped by default. To run:

    uv run pytest tests/integration/test_live_orders.py -m live -v

Requires IB Gateway on the configured PAPER port. Places a deep-OTM SPY put
credit spread at an unfillable limit (credit ~= 95% of the width — nobody
pays that), asserts the order is acknowledged, modifies the price once
(same orderId), cancels, and asserts no position resulted. Verifies the
BUY-bag negative-limit credit convention against the real Gateway without
any realistic fill risk.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr import ContractResolver, IBKRClient, OrderClient
from optionsbot.ibkr.types import OrderStatusUpdate

_ACK_STATUSES = {"PreSubmitted", "Submitted"}
_DONE_STATUSES = {"Cancelled", "ApiCancelled", "Inactive"}


async def _wait_for(
    predicate: Any, timeout: float = 20.0, interval: float = 0.25
) -> bool:
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        await asyncio.sleep(interval)
        waited += interval
    return False


@pytest.mark.live
async def test_live_place_modify_cancel_deep_otm_spread() -> None:
    settings = Settings()
    assert settings.ibkr.paper, "live order smoke must run against paper"
    client = IBKRClient(role="exec", settings=settings)
    resolver = ContractResolver(client)
    order_client = OrderClient(client, resolver)

    statuses: list[OrderStatusUpdate] = []
    callback_errors: list[tuple[str, Exception]] = []
    order_client.on_callback_error(
        lambda kind, error: callback_errors.append((kind, error))
    )
    order_client.on_status(statuses.append)

    try:
        await client.connect()

        # Discover a real expiry + two deep-OTM put strikes from secdef.
        spy = await resolver.stock("SPY")
        params = await client.ib.reqSecDefOptParamsAsync(
            spy.symbol, "", spy.secType, spy.conId
        )
        smart = next(
            p for p in params
            if p.exchange == "SMART" and p.tradingClass == spy.symbol
        )
        from datetime import UTC, datetime, timedelta

        from optionsbot.ibkr import MarketDataClient

        cutoff = (datetime.now(UTC) + timedelta(days=30)).strftime("%Y%m%d")
        expiry = min(e for e in smart.expirations if e >= cutoff)  # ~30-37 DTE

        quote = await MarketDataClient(client, resolver).get_stock_snapshot("SPY")
        spot = quote.mid or quote.last
        assert spot, "no delayed SPY quote available for strike selection"
        # 25-35% below spot: certainly listed for a ~monthly expiry, far OTM,
        # and the order is unfillable anyway (we demand ~95% of the width as
        # credit for a spread worth pennies).
        band = [
            s for s in sorted(smart.strikes)
            if s == int(s) and 0.65 * spot <= s <= 0.75 * spot
        ]
        qualified_band = await resolver.qualify_options(
            "SPY", [(expiry, s, "P") for s in band]
        )
        existing = sorted(spec[1] for spec in qualified_band)
        assert len(existing) >= 2, (
            f"could not find two deep-OTM put strikes for {expiry}; "
            f"qualified: {existing}"
        )
        long_strike, short_strike = existing[0], existing[1]

        legs = [
            {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": expiry,
             "strike": short_strike, "right": "P", "quantity": 1},
            {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": expiry,
             "strike": long_strike, "right": "P", "quantity": 1},
        ]
        width = short_strike - long_strike
        unfillable_credit = round(width * 0.95, 2)

        # Margin preview first — also proves whatIf plumbing live.
        preview = await order_client.whatif_combo(
            "SPY", legs, quantity=1, limit_price=-unfillable_credit,
        )
        assert preview is not None  # fields may be None on paper, call must work

        placed = await order_client.place_combo_limit(
            "SPY", legs, quantity=1,
            limit_price=-unfillable_credit,  # negative = net credit
            order_ref="obot-999999999",
        )
        assert placed.action == "BUY"
        assert placed.limit_price < 0

        acked = await _wait_for(
            lambda: any(s.status in _ACK_STATUSES for s in statuses)
        )
        assert acked, f"order never acknowledged; saw {[s.status for s in statuses]}"
        assert not any(s.status == "Filled" for s in statuses), (
            "the deliberately-unfillable order filled — sign convention or "
            "limit logic is wrong, investigate before going further"
        )

        await order_client.modify_price(
            placed.ib_order_id, new_limit_price=-(unfillable_credit - 0.05)
        )
        await asyncio.sleep(1.0)

        await order_client.cancel(placed.ib_order_id)
        cancelled = await _wait_for(
            lambda: any(s.status in _DONE_STATUSES for s in statuses)
        )
        assert cancelled, f"no cancel ack; saw {[s.status for s in statuses]}"

        # No position should exist in either leg.
        positions = await client.ib.reqPositionsAsync()
        leg_con_ids = set()
        qualified = await resolver.qualify_options(
            "SPY",
            [(expiry, short_strike, "P"), (expiry, long_strike, "P")],
        )
        leg_con_ids = {c.conId for c in qualified.values()}
        held = {
            p.contract.conId for p in positions
            if p.contract.conId in leg_con_ids and p.position != 0
        }
        assert not held, f"unexpected position in test legs: {held}"
        assert not callback_errors
    finally:
        await client.disconnect()
