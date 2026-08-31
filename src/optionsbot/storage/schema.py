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
    text,
)

metadata = MetaData()


watchlist = Table(
    "watchlist",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("view_override_dir", Text),  # nullable; values like 'bull'|'neutral'|'bear'
    Column("view_override_iv", Text),  # nullable; 'high'|'neutral'|'low'
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
    Column("regime_dir", Text),  # 'bull'|'neutral'|'bear'
    Column("regime_iv", Text),  # 'high'|'neutral'|'low'
    Column("raw_json", JSON),  # serialized blob of supporting metrics
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

# A scan persists at most one row for each strategy. This turns the
# (snapshot_id, strategy) lookup used by alerts into an exact identity instead
# of a "latest matching row" guess.
Index(
    "uq_strategy_scores_snapshot_strategy",
    strategy_scores.c.snapshot_id,
    strategy_scores.c.strategy,
    unique=True,
)


alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "strategy_score_id",
        Integer,
        ForeignKey("strategy_scores.id", ondelete="SET NULL"),
        index=True,
    ),
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


# Prospective managed-outcome journal.  A row is one immutable structure for
# one deterministic signal.  It is created before alert, EV, affordability, or
# Hermes admission so rejected ideas remain available to an unbiased shadow
# evaluation.  Repeated scans of the same signal/strategy resolve to the same
# ``opportunity_key`` and may never retarget the frozen legs.
managed_opportunities = Table(
    "managed_opportunities",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("opportunity_key", Text, nullable=False, unique=True),
    Column("signal_id", Text, nullable=False, index=True),
    Column("session", Text, nullable=False, index=True),
    Column("symbol", Text, nullable=False, index=True),
    Column("direction", Text, nullable=False),
    Column("setup_type", Text, nullable=False),
    Column("strategy", Text, nullable=False),
    Column(
        "strategy_score_id",
        Integer,
        ForeignKey("strategy_scores.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        unique=True,
    ),
    Column("structure_hash", Text, nullable=False),
    Column("legs_json", JSON, nullable=False),
    Column("features_json", JSON, nullable=False),
    Column("policy_version", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("detected_at", DateTime(timezone=True), nullable=False),
    # Immutable inputs needed to replay the production scan selector. Equity is
    # frozen later, in the same first-writer-wins transition as bot_action,
    # because the daemon fetches one account summary only after the scan batch.
    Column("decision_batch_id", Text, nullable=False),
    Column("decision_score", Float, nullable=False),
    Column("decision_defined_risk", Integer, nullable=False),
    Column("decision_max_loss", Float),
    Column("decision_account_value_available", Integer),
    Column("decision_account_value_usd", Float),
    Column("baseline_action", Text, nullable=False),
    Column("baseline_reason", Text, nullable=False),
    Column("admission_eligible", Integer, nullable=False, server_default="0"),
    Column("shadow_only", Integer, nullable=False, server_default="0"),
    Column("bot_action", Text),
    Column("bot_reason", Text),
    Column("bot_decided_at", DateTime(timezone=True)),
    Column("session_close_at", DateTime(timezone=True), nullable=False),
    Column("entry_cutoff_at", DateTime(timezone=True), nullable=False),
    Column("timeout_at", DateTime(timezone=True), nullable=False),
    Column("entry_ts", DateTime(timezone=True)),
    Column("entry_combo_bid", Float),
    Column("entry_combo_ask", Float),
    Column("entry_net", Float),
    Column("basis_dollars", Float),
    Column("stop_pct", Float, nullable=False),
    Column("target_pct", Float, nullable=False),
    Column("commission_estimate", Float, nullable=False),
    Column("status", Text, nullable=False),
    Column("outcome", Text),
    Column("resolved_at", DateTime(timezone=True)),
    Column("exit_net", Float),
    Column("gross_pnl", Float),
    Column("net_pnl", Float),
    Column("mfe_dollars", Float),
    Column("mae_dollars", Float),
    Column("last_valid_mark_at", DateTime(timezone=True)),
    Column("valid_marks", Integer, nullable=False, server_default="0"),
    Column("max_mark_gap_seconds", Float),
    Column("training_eligible", Integer, nullable=False, server_default="0"),
    Column("resolution_reason", Text),
    CheckConstraint(
        "direction IN ('bull','bear')",
        name="ck_managed_opportunities_direction",
    ),
    CheckConstraint(
        "status IN ('pending_entry','active','resolved','censored','unobservable')",
        name="ck_managed_opportunities_status",
    ),
    CheckConstraint(
        "outcome IN ('target','stop','timeout','censored') OR outcome IS NULL",
        name="ck_managed_opportunities_outcome",
    ),
    CheckConstraint(
        "stop_pct > 0.0 AND stop_pct < 1.0",
        name="ck_managed_opportunities_stop_pct",
    ),
    CheckConstraint(
        "target_pct > 0.0",
        name="ck_managed_opportunities_target_pct",
    ),
    CheckConstraint(
        "commission_estimate >= 0.0",
        name="ck_managed_opportunities_commission",
    ),
    CheckConstraint(
        "length(decision_batch_id) > 0",
        name="ck_managed_opportunities_decision_batch",
    ),
    CheckConstraint(
        "decision_score >= 0.0 AND decision_score <= 100.0",
        name="ck_managed_opportunities_decision_score",
    ),
    CheckConstraint(
        "decision_defined_risk IN (0, 1)",
        name="ck_managed_opportunities_decision_defined_risk",
    ),
    CheckConstraint(
        "decision_max_loss IS NULL OR decision_max_loss > 0.0",
        name="ck_managed_opportunities_decision_max_loss",
    ),
    CheckConstraint(
        "(bot_decided_at IS NULL AND decision_account_value_available IS NULL "
        "AND decision_account_value_usd IS NULL) OR "
        "(bot_decided_at IS NOT NULL AND "
        "((decision_account_value_available = 0 AND decision_account_value_usd IS NULL) OR "
        "(decision_account_value_available = 1 AND decision_account_value_usd IS NOT NULL)))",
        name="ck_managed_opportunities_decision_account",
    ),
    CheckConstraint(
        "training_eligible IN (0, 1)",
        name="ck_managed_opportunities_training_eligible",
    ),
    CheckConstraint(
        "baseline_action IN ('candidate','hold')",
        name="ck_managed_opportunities_baseline_action",
    ),
    CheckConstraint(
        "admission_eligible IN (0, 1)",
        name="ck_managed_opportunities_admission_eligible",
    ),
    CheckConstraint(
        "shadow_only IN (0, 1)",
        name="ck_managed_opportunities_shadow_only",
    ),
    CheckConstraint(
        "shadow_only = 0 OR admission_eligible = 0",
        name="ck_managed_opportunities_shadow_not_admission_eligible",
    ),
    CheckConstraint(
        "bot_action != 'candidate' OR (admission_eligible = 1 AND shadow_only = 0)",
        name="ck_managed_opportunities_candidate_is_executable",
    ),
    CheckConstraint(
        "bot_action != 'candidate' OR "
        "(decision_defined_risk = 1 AND decision_max_loss IS NOT NULL "
        "AND decision_account_value_available = 1)",
        name="ck_managed_opportunities_candidate_has_risk_evidence",
    ),
    CheckConstraint(
        "bot_action != 'candidate' OR "
        "(bot_decided_at >= detected_at AND bot_decided_at < entry_cutoff_at)",
        name="ck_managed_opportunities_candidate_timing",
    ),
    CheckConstraint(
        "bot_action IN ('candidate','hold') OR bot_action IS NULL",
        name="ck_managed_opportunities_bot_action",
    ),
    CheckConstraint(
        "(bot_action IS NULL AND bot_reason IS NULL AND bot_decided_at IS NULL) OR "
        "(bot_action IS NOT NULL AND bot_reason IS NOT NULL AND bot_decided_at IS NOT NULL)",
        name="ck_managed_opportunities_bot_disposition_complete",
    ),
    CheckConstraint(
        "status != 'resolved' OR "
        "(outcome IN ('target','stop','timeout') AND resolved_at IS NOT NULL "
        "AND entry_ts IS NOT NULL AND basis_dollars IS NOT NULL "
        "AND gross_pnl IS NOT NULL AND net_pnl IS NOT NULL)",
        name="ck_managed_opportunities_resolved_complete",
    ),
    CheckConstraint(
        "training_eligible = 0 OR "
        "(status = 'resolved' AND outcome IN ('target','stop','timeout') "
        "AND bot_decided_at IS NOT NULL AND entry_ts IS NOT NULL "
        "AND bot_decided_at <= entry_ts AND entry_ts < entry_cutoff_at "
        "AND basis_dollars IS NOT NULL AND gross_pnl IS NOT NULL "
        "AND net_pnl IS NOT NULL AND resolved_at IS NOT NULL)",
        name="ck_managed_opportunities_training_complete",
    ),
)

Index(
    "ix_managed_opportunities_signal_strategy",
    managed_opportunities.c.signal_id,
    managed_opportunities.c.strategy,
)
Index(
    "ix_managed_opportunities_status_timeout",
    managed_opportunities.c.status,
    managed_opportunities.c.timeout_at,
)


# One executable synthetic-combo observation per opportunity and scheduler
# bucket.  Bad/missing quotes are records too: silently omitting them would
# make a later target/stop ordering look more certain than it was.
managed_opportunity_marks = Table(
    "managed_opportunity_marks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "opportunity_id",
        Integer,
        ForeignKey("managed_opportunities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("poll_bucket", Integer, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False, index=True),
    Column("leg_quote_min_ts", DateTime(timezone=True)),
    Column("leg_quote_max_ts", DateTime(timezone=True)),
    Column("combo_bid", Float),
    Column("combo_ask", Float),
    Column("combo_mid", Float),
    Column("liquidation_net", Float),
    Column("gross_pnl", Float),
    Column("net_pnl", Float),
    Column("usable", Integer, nullable=False),
    Column("issue", Text),
    Column("legs_json", JSON, nullable=False),
    CheckConstraint(
        "usable IN (0, 1)",
        name="ck_managed_opportunity_marks_usable",
    ),
)

Index(
    "uq_managed_opportunity_marks_bucket",
    managed_opportunity_marks.c.opportunity_id,
    managed_opportunity_marks.c.poll_bucket,
    unique=True,
)


# First-class structured Hermes context record.  It is advisory evidence only;
# no field here is execution authority or a production probability.
managed_context_reviews = Table(
    "managed_context_reviews",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "opportunity_id",
        Integer,
        ForeignKey("managed_opportunities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("received_at", DateTime(timezone=True), nullable=False, index=True),
    Column("timing", Text, nullable=False),
    Column("response_json", JSON, nullable=False),
    Column("response_hash", Text, nullable=False),
    Column("context_probability", Float),
    Column("event_conflict", Integer),
    Column("anomaly_json", JSON, nullable=False),
    Column("evidence_json", JSON, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("prompt_version", Text, nullable=False),
    CheckConstraint(
        "timing IN ('pretrade','post_entry','post_cutoff','post_outcome')",
        name="ck_managed_context_reviews_timing",
    ),
    CheckConstraint(
        "context_probability IS NULL OR "
        "(context_probability >= 0.0 AND context_probability <= 1.0)",
        name="ck_managed_context_reviews_probability",
    ),
    CheckConstraint(
        "event_conflict IN (0, 1) OR event_conflict IS NULL",
        name="ck_managed_context_reviews_event_conflict",
    ),
)

Index(
    "uq_managed_context_reviews_response",
    managed_context_reviews.c.opportunity_id,
    managed_context_reviews.c.response_hash,
    unique=True,
)
Index(
    "uq_managed_context_reviews_critic",
    managed_context_reviews.c.opportunity_id,
    managed_context_reviews.c.model_version,
    managed_context_reviews.c.prompt_version,
    unique=True,
)


# Immutable model artifacts and their fold/holdout evidence.  Capture never
# reads these tables for trade admission; a later controlled promotion layer
# may do so only after explicit validation.
managed_models = Table(
    "managed_models",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_version", Text, nullable=False, unique=True),
    Column("artifact_hash", Text, nullable=False),
    Column("feature_schema_version", Text, nullable=False),
    Column("outcome_policy_version", Text, nullable=False),
    Column("trained_from_session", Text, nullable=False),
    Column("trained_through_session", Text, nullable=False),
    Column("metrics_json", JSON, nullable=False),
    Column("status", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("promoted_at", DateTime(timezone=True)),
    CheckConstraint(
        "status IN ('challenger','promoted','rejected','retired')",
        name="ck_managed_models_status",
    ),
)

Index(
    "uq_managed_models_one_promoted",
    managed_models.c.status,
    unique=True,
    sqlite_where=managed_models.c.status == "promoted",
    postgresql_where=managed_models.c.status == "promoted",
)
Index(
    "uq_managed_models_one_base_challenger",
    managed_models.c.status,
    unique=True,
    sqlite_where=text(
        "status = 'challenger' AND json_extract(metrics_json, '$.model_role') = 'causal_base'"
    ),
    postgresql_where=text(
        "status = 'challenger' AND (metrics_json ->> 'model_role') = 'causal_base'"
    ),
)


managed_model_evaluations = Table(
    "managed_model_evaluations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "model_id",
        Integer,
        ForeignKey("managed_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("evaluation_kind", Text, nullable=False),
    Column("fold_index", Integer, nullable=False),
    Column("train_from_session", Text),
    Column("train_through_session", Text),
    Column("test_from_session", Text, nullable=False),
    Column("test_through_session", Text, nullable=False),
    Column("metrics_json", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "evaluation_kind IN ('walk_forward','holdout','paper_shadow')",
        name="ck_managed_model_evaluations_kind",
    ),
    CheckConstraint(
        "fold_index >= 0",
        name="ck_managed_model_evaluations_fold",
    ),
)

Index(
    "uq_managed_model_evaluations_fold",
    managed_model_evaluations.c.model_id,
    managed_model_evaluations.c.evaluation_kind,
    managed_model_evaluations.c.fold_index,
    unique=True,
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
    Column("day_start_net_liq", Float),
    Column("day_start_session", Text),  # NYSE session date "YYYY-MM-DD" the baseline belongs to
)


# Hermes is an advisory entry overlay, so its correctness breaker is kept
# separate from the global execution kill switch.  A disabled overlay blocks
# only Hermes-vetted entries and survives daemon restarts until an operator
# explicitly resets it.
hermes_overlay_state = Table(
    "hermes_overlay_state",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("enabled", Integer, nullable=False, server_default="1"),
    Column("reason", Text),
    Column("ts", DateTime(timezone=True)),
    Column("judgeable", Integer, nullable=False, server_default="0"),
    Column("accuracy", Float),
    CheckConstraint("id = 1", name="ck_hermes_overlay_state_singleton"),
    CheckConstraint("enabled IN (0, 1)", name="ck_hermes_overlay_state_enabled"),
    CheckConstraint("judgeable >= 0", name="ck_hermes_overlay_state_judgeable"),
    CheckConstraint(
        "accuracy IS NULL OR (accuracy >= 0.0 AND accuracy <= 1.0)",
        name="ck_hermes_overlay_state_accuracy",
    ),
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

# Permanent admission receipt for an exact entry candidate. The receipt is
# inserted in the same transaction as the first staged open order and is never
# deleted or retargeted, so terminal outcomes and mutable order status cannot
# re-authorize the review.
entry_intent_consumptions = Table(
    "entry_intent_consumptions",
    metadata,
    Column(
        "strategy_score_id",
        Integer,
        ForeignKey("strategy_scores.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column(
        "first_order_id",
        Integer,
        ForeignKey("orders.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("consumed_at", DateTime(timezone=True), nullable=False),
)

Index(
    "uq_orders_ib_order_id",
    orders.c.ib_order_id,
    unique=True,
    # IBKR orderId is local to a client session and may be reused after a
    # Gateway reset.  It is unique only among orders the broker can still act
    # on; terminal history is durably identified by order_ref / permId / fills.
    sqlite_where=(
        orders.c.ib_order_id.is_not(None)
        & orders.c.status.in_(["staged", "submitting", "submitted", "partial"])
    ),
    postgresql_where=(
        orders.c.ib_order_id.is_not(None)
        & orders.c.status.in_(["staged", "submitting", "submitted", "partial"])
    ),
)

Index(
    "uq_orders_active_close_per_entry",
    orders.c.closes_order_id,
    unique=True,
    sqlite_where=(
        orders.c.closes_order_id.is_not(None)
        & orders.c.status.in_(["staged", "submitting", "submitted", "partial"])
    ),
    postgresql_where=(
        orders.c.closes_order_id.is_not(None)
        & orders.c.status.in_(["staged", "submitting", "submitted", "partial"])
    ),
)

# Durable high-water mark for adaptive winner management. The value is stored
# per combo unit in the same signed-net space used by execution.exits, so a
# daemon restart cannot forget that a 0DTE debit trade had already armed its
# profit trail.
position_exit_state = Table(
    "position_exit_state",
    metadata,
    Column(
        "entry_order_id",
        Integer,
        ForeignKey("orders.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("peak_pnl_per_unit", Float, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


# A filled option entry can expire without a broker close fill.  Keep that
# economic terminal state separate from executions: fills remain broker facts,
# while this row records a post-clearing expiration settlement.  ITM structures
# use their intrinsic payoff at the official terminal spot; any resulting stock
# assignment is a separate broker position that must be flattened/reconciled.
position_settlements = Table(
    "position_settlements",
    metadata,
    Column(
        "entry_order_id",
        Integer,
        ForeignKey("orders.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("kind", Text, nullable=False),
    Column("expiry", Text, nullable=False),
    Column("terminal_spot", Float, nullable=False),
    Column("pnl", Float, nullable=False),
    Column("commissions", Float, nullable=False),
    Column("settled_at", DateTime(timezone=True), nullable=False, index=True),
    CheckConstraint(
        "kind IN ('expired_worthless','expired_intrinsic')",
        name="ck_position_settlements_kind",
    ),
)


# IBK-138: audited Hermes-originated close requests. MCP writes only a request;
# the daemon owns the trading-soundness gate and converts at most approved
# requests into close orders.
exit_requests = Table(
    "exit_requests",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "position_id",
        Integer,
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("requested_at", DateTime(timezone=True), nullable=False, index=True),
    Column("catalyst_type", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("sources_json", JSON, nullable=False),
    Column("reason", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="requested", index=True),
    Column("decision_reason", Text),
    Column("processed_at", DateTime(timezone=True)),
    Column("close_order_id", Integer, ForeignKey("orders.id", ondelete="SET NULL")),
    CheckConstraint(
        "status IN ('requested','refused','submitted','failed')",
        name="ck_exit_requests_status",
    ),
)

Index("ix_exit_requests_status_requested_at", exit_requests.c.status, exit_requests.c.requested_at)
Index(
    "ix_exit_requests_position_requested_at",
    exit_requests.c.position_id,
    exit_requests.c.requested_at,
)


# Hermes-originated pre-trade reviews. A review is evidence only: the daemon
# may consume a fresh vetted row, but every deterministic execute_pick gate
# remains authoritative. Non-vetted verdicts are persisted as terminal holds.
entry_reviews = Table(
    "entry_reviews",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "strategy_score_id",
        Integer,
        ForeignKey("strategy_scores.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    ),
    Column(
        "alert_id",
        Integer,
        ForeignKey("alerts.id", ondelete="CASCADE"),
        nullable=True,  # legacy unmatched rows are terminalized by migration 0015
        unique=True,
        index=True,
    ),
    Column("reviewed_at", DateTime(timezone=True), nullable=False),
    Column("verdict", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("sources_json", JSON, nullable=False),
    Column("reason", Text, nullable=False),
    Column("checks_json", JSON, nullable=False),
    Column("status", Text, nullable=False, server_default="requested", index=True),
    Column("decision_reason", Text),
    Column("claimed_at", DateTime(timezone=True)),
    Column("processed_at", DateTime(timezone=True)),
    Column("order_id", Integer, ForeignKey("orders.id", ondelete="SET NULL")),
    CheckConstraint(
        "verdict IN ('vetted_paper_candidate','watch_only','no_trade')",
        name="ck_entry_reviews_verdict",
    ),
    CheckConstraint(
        "status IN ('requested','processing','held','refused','submitted','expired','failed')",
        name="ck_entry_reviews_status",
    ),
    CheckConstraint(
        "confidence >= 0.0 AND confidence <= 1.0",
        name="ck_entry_reviews_confidence",
    ),
)

Index(
    "ix_entry_reviews_status_reviewed_at",
    entry_reviews.c.status,
    entry_reviews.c.reviewed_at,
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


# Work-stream D1: in-flight price-walk state, one row per walking order.
# Persisted on every walk step so a daemon restart can re-attach (resume the
# walk) or issue one corrective reprice instead of orphaning the in-memory
# asyncio task until the TTL watcher cancels. Deleted when the walk ends
# (fill/cancel/exhaustion) — a stale row is harmless (load_walk_states joins
# orders and skips terminal rows) but the row's purpose is "actively walking".
walk_state = Table(
    "walk_state",
    metadata,
    Column(
        "order_id",
        Integer,
        ForeignKey("orders.id", ondelete="CASCADE"),
        primary_key=True,  # one walk per order
    ),
    Column("ib_order_id", Integer, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("legs_json", JSON, nullable=False),  # the full legs list, incl. non-OPT
    Column("decision_mid", Float, nullable=False),
    Column("budget", Float, nullable=False),
    Column("increment", Float, nullable=False),
    Column("step", Integer, nullable=False),  # last completed step
    Column("prev_target", Float, nullable=False),  # last signed net target
    Column("updated_ts", DateTime(timezone=True), nullable=False),
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
