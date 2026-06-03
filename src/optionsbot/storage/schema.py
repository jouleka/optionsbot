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
