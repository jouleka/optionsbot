"""SQLAlchemy Core table definitions. Single source of truth for the DB schema."""

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
)

metadata = MetaData()


watchlist = Table(
    "watchlist",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("view_override_dir", Text),  # nullable; values like 'bull'|'neutral'|'bear'
    Column("view_override_iv", Text),   # nullable; 'high'|'neutral'|'low'
    Column("notes", Text),
    Column("added_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "view_override_dir IN ('bull','neutral','bear') OR view_override_dir IS NULL",
        name="ck_watchlist_view_override_dir",
    ),
    CheckConstraint(
        "view_override_iv IN ('high','neutral','low') OR view_override_iv IS NULL",
        name="ck_watchlist_view_override_iv",
    ),
)


snapshots = Table(
    "snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", Text, nullable=False, index=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("spot", Float),
    Column("iv_rank", Float),
    Column("hv20", Float),
    Column("iv_hv_ratio", Float),
    Column("expected_move", Float),
    Column("regime_dir", Text),   # 'bull'|'neutral'|'bear'
    Column("regime_iv", Text),    # 'high'|'neutral'|'low'
    Column("raw_json", JSON),     # serialized blob of supporting metrics
    CheckConstraint(
        "regime_dir IN ('bull','neutral','bear') OR regime_dir IS NULL",
        name="ck_snapshots_regime_dir",
    ),
    CheckConstraint(
        "regime_iv IN ('high','neutral','low') OR regime_iv IS NULL",
        name="ck_snapshots_regime_iv",
    ),
)


iv_history = Table(
    "iv_history",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("date", Date, primary_key=True),
    Column("atm_iv", Float, nullable=False),
)


strategy_scores = Table(
    "strategy_scores",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "snapshot_id",
        Integer,
        ForeignKey("snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("strategy", Text, nullable=False, index=True),
    Column("score", Float, nullable=False),
    Column("rationale", Text),
    Column("legs_json", JSON),
    # Suggestion blob: defined_risk, credit_or_debit, max_loss, max_profit,
    # prob_profit, suggested_quantity. Persisted so retry alerts (see
    # daemon/alert_pipeline._reconstruct_scored) can render the same UNDEFINED
    # RISK warning + financial figures as the first attempt.
    Column("suggestion_json", JSON),
)


alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("symbol", Text, nullable=False),
    Column("strategy", Text, nullable=False),
    Column("score", Float, nullable=False),
    Column("status", Text, nullable=False),
    Column("sent_ts", DateTime(timezone=True)),
    Column("telegram_msg_id", Integer),
    Column("retry_count", Integer, nullable=False, server_default="0"),
    Column("next_retry_ts", DateTime(timezone=True)),
    Column("last_error", Text),
    CheckConstraint(
        "status IN ('pending','sent','failed','dropped')",
        name="ck_alerts_status",
    ),
)


Index("ix_alerts_symbol_strategy", alerts.c.symbol, alerts.c.strategy)
Index("ix_alerts_status_next_retry_ts", alerts.c.status, alerts.c.next_retry_ts)


scan_runs = Table(
    "scan_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started", DateTime(timezone=True), nullable=False),
    Column("finished", DateTime(timezone=True)),
    Column("tickers_scanned", Integer),
    Column("alerts_fired", Integer),
    Column("errors_json", JSON),
)


pick_outcomes = Table(
    "pick_outcomes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "strategy_score_id",
        Integer,
        ForeignKey("strategy_scores.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    Column("symbol", Text, nullable=False),
    Column("strategy", Text, nullable=False),
    Column("expiry", Text, nullable=False),
    Column("entry_spot", Float, nullable=False),
    Column("predicted_prob_profit", Float),
    Column("score", Float),
    Column("credit_or_debit", Float),
    Column("max_profit", Float),
    Column("max_loss", Float),
    Column("risk_tier", Text),
    Column("terminal_spot", Float, nullable=False),
    Column("realized_pnl", Float, nullable=False),
    Column("win", Integer, nullable=False),
    Column("evaluated_at", DateTime(timezone=True), nullable=False),
)


symbol_news = Table(
    "symbol_news",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("headlines_json", JSON),
)


# IBK-113: one row per management alert sent, keyed by dedup_key
# (symbol:expiry:strike:right:trigger[+trigger], IBK-119 merges a leg's triggers into one
# key), so should_manage_alert suppresses re-fires within the cooldown across daemon restarts.
position_alerts = Table(
    "position_alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dedup_key", Text, nullable=False, index=True),
    Column("ts", DateTime(timezone=True), nullable=False),
)


# IBK-123: singleton-row (id=1) execution kill switch. Persisted (unlike
# DaemonContext.alerting_paused) so a tripped switch survives daemon restarts;
# cleared only by an explicit /arm.
execution_state = Table(
    "execution_state",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("killed", Integer, nullable=False, server_default="0"),
    Column("reason", Text),
    Column("ts", DateTime(timezone=True)),
)


# IBK-124: the order ledger. One row per order INTENT (entry or exit), staged
# before any network call so a crash mid-submit is recoverable (IBK-128
# resolves `submitting` rows against the broker). order_ref ("obot-{id}") is
# stamped into IBKR's Order.orderRef so broker-side orders map back to rows
# unambiguously; limit_price is per combo unit, negative = net credit under
# the BUY-bag convention.
orders = Table(
    "orders",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "strategy_score_id",
        Integer,
        ForeignKey("strategy_scores.id", ondelete="SET NULL"),
        index=True,  # nullable: close orders (IBK-129) won't reference a pick
    ),
    Column("intent", Text, nullable=False),
    Column("symbol", Text, nullable=False, index=True),
    Column("strategy", Text, nullable=False),
    # [{symbol, side('buy'|'sell'), sec_type, expiry, strike, right, quantity}]
    # copied verbatim from strategy_scores — carries NO conIds; contract
    # qualification is the submitter's job at order time (IBK-125).
    Column("legs_json", JSON),
    Column("quantity", Integer, nullable=False),
    Column("limit_price", Float),
    Column("ib_order_id", Integer),
    Column("ib_perm_id", Integer),
    Column("order_ref", Text, unique=True),
    # IBK-129: a closing order points at the entry it closes (nullable; only
    # intent='close' rows carry it). Entry "position open" = filled open-intent
    # order with no filled close.
    Column("closes_order_id", Integer, ForeignKey("orders.id", ondelete="SET NULL")),
    Column("status", Text, nullable=False, index=True),
    Column("staged_ts", DateTime(timezone=True), nullable=False),
    Column("submitted_ts", DateTime(timezone=True)),
    Column("terminal_ts", DateTime(timezone=True)),
    Column("last_error", Text),
    Column("reprice_count", Integer, nullable=False, server_default="0"),
    CheckConstraint("intent IN ('open','close')", name="ck_orders_intent"),
    CheckConstraint(
        "status IN ('staged','submitting','submitted','partial','filled',"
        "'cancelled','rejected','abandoned','skipped')",
        name="ck_orders_status",
    ),
)


# IBK-124: per-LEG executions (combo orders report one execution per leg, each
# with its own execId; IBKR re-sends executions on reconnect, hence the UNIQUE
# ib_exec_id dedupe). commission arrives separately via commissionReport,
# keyed by the same execId.
# IBK-127: decision-quote journal — the implementation-shortfall baseline.
# One row at decision time plus one per price-walk step, carrying the exact
# per-leg NBBO the bot acted on, so realized fills can be compared against
# the quotes that justified them (slippage measurement on paper AND live).
order_quotes = Table(
    "order_quotes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "order_id",
        Integer,
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("kind", Text, nullable=False),  # decision | step | final
    Column("step", Integer, nullable=False, server_default="0"),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("combo_bid", Float),
    Column("combo_ask", Float),
    Column("combo_mid", Float),
    Column("target_net", Float),
    Column("limit_price", Float),
    Column("legs_json", JSON),  # per-leg {expiry, strike, right, side, bid, ask, mid, delayed}
    CheckConstraint("kind IN ('decision','step','final')", name="ck_order_quotes_kind"),
)


fills = Table(
    "fills",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "order_id",
        Integer,
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("ib_exec_id", Text, nullable=False, unique=True),
    Column("side", Text, nullable=False),
    Column("price", Float, nullable=False),
    Column("qty", Integer, nullable=False),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("commission", Float),
    Column("leg_con_id", Integer),
    CheckConstraint("side IN ('BUY','SELL')", name="ck_fills_side"),
)
