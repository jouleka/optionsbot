"""Tests for OrderClient (IBK-125) against a stubbed ib_async.IB."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.orders import OrderClient
from optionsbot.ibkr.types import (
    CommissionUpdate,
    ExecutionFill,
    MarginPreview,
    OrderStatusUpdate,
    PlacedOrder,
)

CONDOR_LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 620.0, "right": "C", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 625.0, "right": "C", "quantity": 1},
]

CONID_BY_SPEC = {
    ("20260717", 580.0, "P"): 1580,
    ("20260717", 575.0, "P"): 1575,
    ("20260717", 620.0, "C"): 1620,
    ("20260717", 625.0, "C"): 1625,
}


@pytest.fixture()
def order_settings() -> Settings:
    s = Settings()  # paper=True, port=4002 — interlock passes by default
    return s


@pytest.fixture()
def order_ib(mock_ib: MagicMock) -> MagicMock:
    """mock_ib specialized for order flows."""
    mock_ib.isConnected = MagicMock(return_value=True)

    async def fake_qualify(*contracts: Any) -> list[Any]:
        out = []
        for c in contracts:
            key = (c.lastTradeDateOrContractMonth, c.strike, c.right)
            c.conId = CONID_BY_SPEC.get(key, 0)
            out.append(c if c.conId else None)
        return out

    mock_ib.qualifyContractsAsync = AsyncMock(side_effect=fake_qualify)

    next_id = iter(range(7, 100))

    def fake_place(contract: Any, order: Any) -> Any:
        from ib_async import Trade

        if not order.orderId:
            order.orderId = next(next_id)
        return Trade(contract=contract, order=order)

    mock_ib.placeOrder = MagicMock(side_effect=fake_place)
    mock_ib.cancelOrder = MagicMock(return_value=None)
    mock_ib.whatIfOrderAsync = AsyncMock()
    # Event hooks: real ib_async IB exposes Event objects supporting +=.
    mock_ib.orderStatusEvent = MagicMock()
    mock_ib.execDetailsEvent = MagicMock()
    mock_ib.commissionReportEvent = MagicMock()
    return mock_ib


@pytest.fixture()
def order_client(order_settings: Settings, order_ib: MagicMock) -> OrderClient:
    client = IBKRClient(role="exec", settings=order_settings, ib=order_ib)
    resolver = ContractResolver(client)
    return OrderClient(client, resolver)


# --- role / config -----------------------------------------------------------


def test_exec_role_uses_client_id_3(order_settings: Settings) -> None:
    client = IBKRClient(role="exec", settings=order_settings)
    assert client._client_id() == 3  # noqa: SLF001 — pinning the role mapping


# --- combo building + placement ----------------------------------------------


async def test_place_condor_builds_guaranteed_smart_bag(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=2, limit_price=-1.55, order_ref="obot-7",
    )
    assert isinstance(placed, PlacedOrder)
    contract, order = order_ib.placeOrder.call_args.args
    assert contract.secType == "BAG"
    assert contract.symbol == "SPY"
    assert contract.currency == "USD"
    assert contract.exchange == "SMART"
    legs = contract.comboLegs
    assert [(leg.conId, leg.action, leg.ratio, leg.exchange) for leg in legs] == [
        (1580, "SELL", 1, "SMART"),
        (1575, "BUY", 1, "SMART"),
        (1620, "SELL", 1, "SMART"),
        (1625, "BUY", 1, "SMART"),
    ]
    assert order.action == "BUY"  # buy the bag; credit = negative net price
    assert order.totalQuantity == 2
    assert order.lmtPrice == -1.55
    assert order.orderRef == "obot-7"
    assert order.tif == "DAY"
    assert placed.ib_order_id == order.orderId


async def test_stk_legs_are_filtered_out(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    covered_call = [
        {"symbol": "SPY", "side": "buy", "sec_type": "STK", "quantity": 100,
         "expiry": None, "strike": None, "right": None},
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 620.0, "right": "C", "quantity": 1},
    ]
    await order_client.place_combo_limit(
        "SPY", covered_call, quantity=1, limit_price=-1.20, order_ref="obot-8",
    )
    contract, order = order_ib.placeOrder.call_args.args
    # Single remaining OPT leg -> plain qualified Option, not a 1-leg BAG.
    assert contract.secType == "OPT"
    assert contract.conId == 1620
    assert order.action == "SELL"  # the leg's own side
    assert order.lmtPrice == 1.20  # positive premium for a single leg
    assert order.totalQuantity == 1


async def test_single_long_leg_buys_at_positive_price(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    long_put = [
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
         "strike": 580.0, "right": "P", "quantity": 1},
    ]
    await order_client.place_combo_limit(
        "SPY", long_put, quantity=1, limit_price=2.30, order_ref="obot-9",
    )
    contract, order = order_ib.placeOrder.call_args.args
    assert contract.secType == "OPT"
    assert order.action == "BUY"
    assert order.lmtPrice == 2.30


async def test_unqualified_leg_raises(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    bad_legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 999.0, "right": "P", "quantity": 1},  # not in CONID_BY_SPEC
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
         "strike": 575.0, "right": "P", "quantity": 1},
    ]
    with pytest.raises(ValueError, match="qualify"):
        await order_client.place_combo_limit(
            "SPY", bad_legs, quantity=1, limit_price=-0.5, order_ref="obot-10",
        )
    order_ib.placeOrder.assert_not_called()


async def test_no_option_legs_raises(order_client: OrderClient) -> None:
    with pytest.raises(ValueError, match="option leg"):
        await order_client.place_combo_limit(
            "SPY",
            [{"symbol": "SPY", "side": "buy", "sec_type": "STK", "quantity": 100,
              "expiry": None, "strike": None, "right": None}],
            quantity=1, limit_price=1.0, order_ref="obot-11",
        )


# --- paper interlock ----------------------------------------------------------


async def test_mutations_blocked_off_paper(order_ib: MagicMock) -> None:
    live = Settings()
    live.ibkr.paper = False
    client = IBKRClient(role="exec", settings=live, ib=order_ib)
    oc = OrderClient(client, ContractResolver(client))
    with pytest.raises(RuntimeError, match="paper"):
        await oc.place_combo_limit(
            "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.0, order_ref="obot-12",
        )
    with pytest.raises(RuntimeError, match="paper"):
        await oc.modify_price(7, new_limit_price=-0.9)
    with pytest.raises(RuntimeError, match="paper"):
        await oc.cancel(7)
    order_ib.placeOrder.assert_not_called()


async def test_mutations_blocked_on_live_port(order_ib: MagicMock) -> None:
    live_port = Settings()
    live_port.ibkr.port = 4001  # paper flag True but live Gateway port
    client = IBKRClient(role="exec", settings=live_port, ib=order_ib)
    oc = OrderClient(client, ContractResolver(client))
    with pytest.raises(RuntimeError, match="4001"):
        await oc.place_combo_limit(
            "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.0, order_ref="obot-13",
        )


# --- modify / cancel ------------------------------------------------------------


async def test_modify_price_reuses_same_order_id(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55, order_ref="obot-14",
    )
    await order_client.modify_price(placed.ib_order_id, new_limit_price=-1.45)
    assert order_ib.placeOrder.call_count == 2
    _, first_order = order_ib.placeOrder.call_args_list[0].args
    _, second_order = order_ib.placeOrder.call_args_list[1].args
    assert second_order is first_order  # SAME order object, price-only change
    assert second_order.orderId == placed.ib_order_id
    assert second_order.lmtPrice == -1.45


async def test_modify_single_leg_normalizes_to_positive_premium(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    # Live Error 201: the walk passes signed net targets (credit = negative);
    # a single-leg option must convert to a positive premium on modify just
    # like it does at placement.
    short_put = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 580.0, "right": "P", "quantity": 1},
    ]
    placed = await order_client.place_combo_limit(
        "SPY", short_put, quantity=1, limit_price=-6.15, order_ref="obot-16",
    )
    await order_client.modify_price(placed.ib_order_id, new_limit_price=-6.14)
    _, modified_order = order_ib.placeOrder.call_args_list[1].args
    assert modified_order.lmtPrice == pytest.approx(+6.14)  # positive premium


async def test_modify_bag_keeps_signed_limit(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55, order_ref="obot-17",
    )
    await order_client.modify_price(placed.ib_order_id, new_limit_price=-1.45)
    _, modified_order = order_ib.placeOrder.call_args_list[1].args
    assert modified_order.lmtPrice == pytest.approx(-1.45)  # signed BAG net


async def test_modify_unknown_order_raises(order_client: OrderClient) -> None:
    with pytest.raises(ValueError, match="unknown"):
        await order_client.modify_price(31337, new_limit_price=-1.0)


async def test_cancel_uses_registered_order(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55, order_ref="obot-15",
    )
    await order_client.cancel(placed.ib_order_id)
    (cancelled_order,) = order_ib.cancelOrder.call_args.args
    assert cancelled_order.orderId == placed.ib_order_id


# --- whatIf ---------------------------------------------------------------------


async def test_whatif_parses_order_state(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = OrderState(
        initMarginChange="2345.67",
        maintMarginChange="2100.00",
        equityWithLoanChange="-12.5",
        commission=1.31,
        maxCommission=1.7976931348623157e308,  # IBKR's UNSET_DOUBLE sentinel
        warningText="",
    )
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert isinstance(preview, MarginPreview)
    assert preview.init_margin_change == pytest.approx(2345.67)
    assert preview.maint_margin_change == pytest.approx(2100.00)
    assert preview.equity_with_loan_change == pytest.approx(-12.5)
    assert preview.commission == pytest.approx(1.31)
    assert preview.max_commission is None  # UNSET sentinel -> None
    assert preview.warning is None  # empty string -> None


async def test_whatif_handles_unparseable_margins(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = OrderState(warningText="check this")
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert preview.init_margin_change is None
    assert preview.warning == "check this"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_whatif_rejects_non_finite_margins(
    order_client: OrderClient,
    order_ib: MagicMock,
    value: float,
) -> None:
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = OrderState(initMarginChange=value)
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert preview.init_margin_change is None


async def test_whatif_tolerates_list_result(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    # ib_async 2.1.0 returns a LIST of OrderState from whatIfOrderAsync
    # despite its annotation — observed against a live paper Gateway.
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = [OrderState(initMarginChange="100.5")]
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert preview.init_margin_change == pytest.approx(100.5)

    order_ib.whatIfOrderAsync.return_value = []
    empty = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert empty.init_margin_change is None
    assert empty.warning is not None


# --- event translation ------------------------------------------------------------


def _make_trade(order_id: int = 7, ref: str = "obot-7") -> Any:
    from ib_async import Contract, Order, OrderStatus, Trade

    order = Order(orderId=order_id, permId=4242, orderRef=ref)
    status = OrderStatus(
        orderId=order_id, status="Submitted", filled=1.0, remaining=1.0,
        avgFillPrice=1.5,
    )
    return Trade(contract=Contract(secType="BAG", symbol="SPY"), order=order,
                 orderStatus=status)


def test_status_event_translates(order_client: OrderClient) -> None:
    seen: list[OrderStatusUpdate] = []
    order_client.on_status(seen.append)
    order_client._handle_order_status(_make_trade())  # noqa: SLF001
    [update] = seen
    assert update.ib_order_id == 7
    assert update.perm_id == 4242
    assert update.order_ref == "obot-7"
    assert update.status == "Submitted"
    assert update.filled == 1.0
    assert update.remaining == 1.0
    assert update.avg_fill_price == 1.5


def test_fill_event_translates_and_normalizes_side(
    order_client: OrderClient,
) -> None:
    from datetime import UTC, datetime

    from ib_async import Contract, Execution, Fill

    seen: list[ExecutionFill] = []
    order_client.on_fill(seen.append)
    execution = Execution(
        execId="0001.aa.01", time=datetime(2026, 6, 10, 15, 30, tzinfo=UTC),
        side="BOT", shares=2.0, price=0.40, permId=4242, orderId=7,
        orderRef="obot-7",
    )
    leg_contract = Contract(secType="OPT", symbol="SPY", conId=1575)
    fill = Fill(
        contract=leg_contract, execution=execution,
        commissionReport=None, time=execution.time,  # type: ignore[arg-type]
    )
    order_client._handle_exec_details(_make_trade(), fill)  # noqa: SLF001
    [record] = seen
    assert record.exec_id == "0001.aa.01"
    assert record.side == "BUY"  # BOT normalized
    assert record.qty == 2
    assert record.price == 0.40
    assert record.con_id == 1575
    assert record.sec_type == "OPT"
    assert record.order_ref == "obot-7"
    assert record.ts.tzinfo is not None


def test_status_event_tolerates_none_perm_id(order_client: OrderClient) -> None:
    # Order.permId defaults to 0 but the dataclass accepts None; a None must
    # not blow up the handler (the status update would be silently dropped).
    from ib_async import Contract, Order, OrderStatus, Trade

    seen: list[OrderStatusUpdate] = []
    order_client.on_status(seen.append)
    trade = Trade(
        contract=Contract(secType="BAG", symbol="SPY"),
        order=Order(orderId=7, permId=None, orderRef="obot-7"),  # type: ignore[arg-type]
        orderStatus=OrderStatus(orderId=7, status="Submitted", filled=0, remaining=1),
    )
    order_client._handle_order_status(trade)  # noqa: SLF001
    [update] = seen
    assert update.perm_id is None


async def test_adopt_open_orders_rebinds_registry_for_our_orders_only(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    ours = Trade(
        contract=Contract(secType="BAG", symbol="SPY"),
        order=Order(orderId=44, permId=99, orderRef="obot-12"),
        orderStatus=OrderStatus(orderId=44, status="Submitted"),
    )
    manual = Trade(
        contract=Contract(secType="OPT", symbol="SPY"),
        order=Order(orderId=0, permId=7, orderRef=""),
        orderStatus=OrderStatus(orderId=0, status="Submitted"),
    )
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[ours, manual])

    adopted = await order_client.adopt_open_orders()
    assert (44, "obot-12", "Submitted") in adopted
    assert (0, None, "Submitted") in adopted  # reported for classification
    # Registry holds OURS only — modify works again after a restart...
    await order_client.modify_price(44, new_limit_price=-1.10)
    order_ib.placeOrder.assert_called_once()
    # ...but a manual TWS order (orderId 0) must never be modifiable.
    with pytest.raises(ValueError, match="unknown"):
        await order_client.modify_price(0, new_limit_price=-1.0)


async def test_recent_executions_translates_with_commission(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from datetime import UTC, datetime

    from ib_async import CommissionReport, Contract, Execution, Fill

    execution = Execution(
        execId="0009.aa.01", time=datetime(2026, 6, 11, 15, 0, tzinfo=UTC),
        side="SLD", shares=1.0, price=1.55, permId=99, orderId=44,
        orderRef="obot-12",
    )
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=1580),
        execution=execution,
        commissionReport=CommissionReport(execId="0009.aa.01", commission=0.66),
        time=execution.time,
    )
    order_ib.reqExecutionsAsync = AsyncMock(return_value=[fill])

    [record] = await order_client.recent_executions()
    # The filter must carry a lookback time — an empty filter returns TODAY
    # only and would hide weekend-outage fills from reconciliation.
    (exec_filter,) = order_ib.reqExecutionsAsync.call_args.args
    assert exec_filter.time  # non-empty lookback
    assert record.exec_id == "0009.aa.01"
    assert record.side == "SELL"
    assert record.order_ref == "obot-12"
    assert record.commission == pytest.approx(0.66)
    assert record.sec_type == "OPT"


def test_commission_event_translates(order_client: OrderClient) -> None:
    from ib_async import CommissionReport

    seen: list[CommissionUpdate] = []
    order_client.on_commission(seen.append)
    report = CommissionReport(execId="0001.aa.01", commission=1.31, currency="USD")
    order_client._handle_commission(_make_trade(), None, report)  # noqa: SLF001
    [update] = seen
    assert update.exec_id == "0001.aa.01"
    assert update.commission == pytest.approx(1.31)
