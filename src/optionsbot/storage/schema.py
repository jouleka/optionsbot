"""SQLAlchemy Core table definitions. Single source of truth for the DB schema."""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
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
    Column("added_at", DateTime, nullable=False),
)


snapshots = Table(
    "snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", Text, nullable=False, index=True),
    Column("ts", DateTime, nullable=False, index=True),
    Column("spot", Float),
    Column("iv_rank", Float),
    Column("hv20", Float),
    Column("iv_hv_ratio", Float),
    Column("expected_move", Float),
    Column("regime_dir", Text),   # 'bull'|'neutral'|'bear'
    Column("regime_iv", Text),    # 'high'|'neutral'|'low'
    Column("raw_json", Text),     # serialized blob of supporting metrics
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
    Column("legs_json", Text),
)


alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, nullable=False, index=True),
    Column("symbol", Text, nullable=False),
    Column("strategy", Text, nullable=False),
    Column("score", Float, nullable=False),
    Column("status", Text, nullable=False),
    Column("sent_ts", DateTime),
    Column("telegram_msg_id", Integer),
    Column("retry_count", Integer, nullable=False, server_default="0"),
    Column("next_retry_ts", DateTime),
    Column("last_error", Text),
    CheckConstraint(
        "status IN ('pending','sent','failed','dropped')",
        name="ck_alerts_status",
    ),
)


scan_runs = Table(
    "scan_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started", DateTime, nullable=False),
    Column("finished", DateTime),
    Column("tickers_scanned", Integer),
    Column("alerts_fired", Integer),
    Column("errors_json", Text),
)
