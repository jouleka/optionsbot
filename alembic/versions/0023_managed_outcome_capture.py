"""Add prospective managed-outcome capture and model registries.

Revision ID: 0023
Revises: 0022
Create Date: 2026-08-29 16:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0023"
down_revision: str | Sequence[str] | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "managed_opportunities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("opportunity_key", sa.Text(), nullable=False),
        sa.Column("signal_id", sa.Text(), nullable=False),
        sa.Column("session", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("setup_type", sa.Text(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("strategy_score_id", sa.Integer(), nullable=False),
        sa.Column("structure_hash", sa.Text(), nullable=False),
        sa.Column("legs_json", sa.JSON(), nullable=False),
        sa.Column("features_json", sa.JSON(), nullable=False),
        sa.Column("policy_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_batch_id", sa.Text(), nullable=False),
        sa.Column("decision_score", sa.Float(), nullable=False),
        sa.Column("decision_defined_risk", sa.Integer(), nullable=False),
        sa.Column("decision_max_loss", sa.Float(), nullable=True),
        sa.Column("decision_account_value_available", sa.Integer(), nullable=True),
        sa.Column("decision_account_value_usd", sa.Float(), nullable=True),
        sa.Column("baseline_action", sa.Text(), nullable=False),
        sa.Column("baseline_reason", sa.Text(), nullable=False),
        sa.Column("admission_eligible", sa.Integer(), server_default="0", nullable=False),
        sa.Column("shadow_only", sa.Integer(), server_default="0", nullable=False),
        sa.Column("bot_action", sa.Text(), nullable=True),
        sa.Column("bot_reason", sa.Text(), nullable=True),
        sa.Column("bot_decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("session_close_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timeout_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("entry_combo_bid", sa.Float(), nullable=True),
        sa.Column("entry_combo_ask", sa.Float(), nullable=True),
        sa.Column("entry_net", sa.Float(), nullable=True),
        sa.Column("basis_dollars", sa.Float(), nullable=True),
        sa.Column("stop_pct", sa.Float(), nullable=False),
        sa.Column("target_pct", sa.Float(), nullable=False),
        sa.Column("commission_estimate", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_net", sa.Float(), nullable=True),
        sa.Column("gross_pnl", sa.Float(), nullable=True),
        sa.Column("net_pnl", sa.Float(), nullable=True),
        sa.Column("mfe_dollars", sa.Float(), nullable=True),
        sa.Column("mae_dollars", sa.Float(), nullable=True),
        sa.Column("last_valid_mark_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_marks", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_mark_gap_seconds", sa.Float(), nullable=True),
        sa.Column("training_eligible", sa.Integer(), server_default="0", nullable=False),
        sa.Column("resolution_reason", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "direction IN ('bull','bear')",
            name="ck_managed_opportunities_direction",
        ),
        sa.CheckConstraint(
            "status IN ('pending_entry','active','resolved','censored','unobservable')",
            name="ck_managed_opportunities_status",
        ),
        sa.CheckConstraint(
            "outcome IN ('target','stop','timeout','censored') OR outcome IS NULL",
            name="ck_managed_opportunities_outcome",
        ),
        sa.CheckConstraint(
            "stop_pct > 0.0 AND stop_pct < 1.0",
            name="ck_managed_opportunities_stop_pct",
        ),
        sa.CheckConstraint(
            "target_pct > 0.0",
            name="ck_managed_opportunities_target_pct",
        ),
        sa.CheckConstraint(
            "commission_estimate >= 0.0",
            name="ck_managed_opportunities_commission",
        ),
        sa.CheckConstraint(
            "length(decision_batch_id) > 0",
            name="ck_managed_opportunities_decision_batch",
        ),
        sa.CheckConstraint(
            "decision_score >= 0.0 AND decision_score <= 100.0",
            name="ck_managed_opportunities_decision_score",
        ),
        sa.CheckConstraint(
            "decision_defined_risk IN (0, 1)",
            name="ck_managed_opportunities_decision_defined_risk",
        ),
        sa.CheckConstraint(
            "decision_max_loss IS NULL OR decision_max_loss > 0.0",
            name="ck_managed_opportunities_decision_max_loss",
        ),
        sa.CheckConstraint(
            "(bot_decided_at IS NULL AND decision_account_value_available IS NULL "
            "AND decision_account_value_usd IS NULL) OR "
            "(bot_decided_at IS NOT NULL AND "
            "((decision_account_value_available = 0 AND decision_account_value_usd IS NULL) OR "
            "(decision_account_value_available = 1 AND decision_account_value_usd IS NOT NULL)))",
            name="ck_managed_opportunities_decision_account",
        ),
        sa.CheckConstraint(
            "training_eligible IN (0, 1)",
            name="ck_managed_opportunities_training_eligible",
        ),
        sa.CheckConstraint(
            "baseline_action IN ('candidate','hold')",
            name="ck_managed_opportunities_baseline_action",
        ),
        sa.CheckConstraint(
            "admission_eligible IN (0, 1)",
            name="ck_managed_opportunities_admission_eligible",
        ),
        sa.CheckConstraint(
            "shadow_only IN (0, 1)",
            name="ck_managed_opportunities_shadow_only",
        ),
        sa.CheckConstraint(
            "shadow_only = 0 OR admission_eligible = 0",
            name="ck_managed_opportunities_shadow_not_admission_eligible",
        ),
        sa.CheckConstraint(
            "bot_action != 'candidate' OR (admission_eligible = 1 AND shadow_only = 0)",
            name="ck_managed_opportunities_candidate_is_executable",
        ),
        sa.CheckConstraint(
            "bot_action != 'candidate' OR "
            "(decision_defined_risk = 1 AND decision_max_loss IS NOT NULL "
            "AND decision_account_value_available = 1)",
            name="ck_managed_opportunities_candidate_has_risk_evidence",
        ),
        sa.CheckConstraint(
            "bot_action != 'candidate' OR "
            "(bot_decided_at >= detected_at AND bot_decided_at < entry_cutoff_at)",
            name="ck_managed_opportunities_candidate_timing",
        ),
        sa.CheckConstraint(
            "bot_action IN ('candidate','hold') OR bot_action IS NULL",
            name="ck_managed_opportunities_bot_action",
        ),
        sa.CheckConstraint(
            "(bot_action IS NULL AND bot_reason IS NULL AND bot_decided_at IS NULL) OR "
            "(bot_action IS NOT NULL AND bot_reason IS NOT NULL "
            "AND bot_decided_at IS NOT NULL)",
            name="ck_managed_opportunities_bot_disposition_complete",
        ),
        sa.CheckConstraint(
            "status != 'resolved' OR "
            "(outcome IN ('target','stop','timeout') AND resolved_at IS NOT NULL "
            "AND entry_ts IS NOT NULL AND basis_dollars IS NOT NULL "
            "AND gross_pnl IS NOT NULL AND net_pnl IS NOT NULL)",
            name="ck_managed_opportunities_resolved_complete",
        ),
        sa.CheckConstraint(
            "training_eligible = 0 OR "
            "(status = 'resolved' AND outcome IN ('target','stop','timeout') "
            "AND bot_decided_at IS NOT NULL AND entry_ts IS NOT NULL "
            "AND bot_decided_at <= entry_ts AND entry_ts < entry_cutoff_at "
            "AND basis_dollars IS NOT NULL AND gross_pnl IS NOT NULL "
            "AND net_pnl IS NOT NULL AND resolved_at IS NOT NULL)",
            name="ck_managed_opportunities_training_complete",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_score_id"],
            ["strategy_scores.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("opportunity_key"),
    )
    op.create_index(
        "ix_managed_opportunities_signal_id",
        "managed_opportunities",
        ["signal_id"],
        unique=False,
    )
    op.create_index(
        "ix_managed_opportunities_session",
        "managed_opportunities",
        ["session"],
        unique=False,
    )
    op.create_index(
        "ix_managed_opportunities_symbol",
        "managed_opportunities",
        ["symbol"],
        unique=False,
    )
    op.create_index(
        "ix_managed_opportunities_strategy_score_id",
        "managed_opportunities",
        ["strategy_score_id"],
        unique=True,
    )
    op.create_index(
        "ix_managed_opportunities_signal_strategy",
        "managed_opportunities",
        ["signal_id", "strategy"],
        unique=False,
    )
    op.create_index(
        "ix_managed_opportunities_status_timeout",
        "managed_opportunities",
        ["status", "timeout_at"],
        unique=False,
    )

    op.create_table(
        "managed_opportunity_marks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("opportunity_id", sa.Integer(), nullable=False),
        sa.Column("poll_bucket", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("leg_quote_min_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("leg_quote_max_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("combo_bid", sa.Float(), nullable=True),
        sa.Column("combo_ask", sa.Float(), nullable=True),
        sa.Column("combo_mid", sa.Float(), nullable=True),
        sa.Column("liquidation_net", sa.Float(), nullable=True),
        sa.Column("gross_pnl", sa.Float(), nullable=True),
        sa.Column("net_pnl", sa.Float(), nullable=True),
        sa.Column("usable", sa.Integer(), nullable=False),
        sa.Column("issue", sa.Text(), nullable=True),
        sa.Column("legs_json", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "usable IN (0, 1)",
            name="ck_managed_opportunity_marks_usable",
        ),
        sa.ForeignKeyConstraint(
            ["opportunity_id"],
            ["managed_opportunities.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_managed_opportunity_marks_opportunity_id",
        "managed_opportunity_marks",
        ["opportunity_id"],
        unique=False,
    )
    op.create_index(
        "ix_managed_opportunity_marks_observed_at",
        "managed_opportunity_marks",
        ["observed_at"],
        unique=False,
    )
    op.create_index(
        "uq_managed_opportunity_marks_bucket",
        "managed_opportunity_marks",
        ["opportunity_id", "poll_bucket"],
        unique=True,
    )

    op.create_table(
        "managed_context_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("opportunity_id", sa.Integer(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timing", sa.Text(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("response_hash", sa.Text(), nullable=False),
        sa.Column("context_probability", sa.Float(), nullable=True),
        sa.Column("event_conflict", sa.Integer(), nullable=True),
        sa.Column("anomaly_json", sa.JSON(), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "timing IN ('pretrade','post_entry','post_cutoff','post_outcome')",
            name="ck_managed_context_reviews_timing",
        ),
        sa.CheckConstraint(
            "context_probability IS NULL OR "
            "(context_probability >= 0.0 AND context_probability <= 1.0)",
            name="ck_managed_context_reviews_probability",
        ),
        sa.CheckConstraint(
            "event_conflict IN (0, 1) OR event_conflict IS NULL",
            name="ck_managed_context_reviews_event_conflict",
        ),
        sa.ForeignKeyConstraint(
            ["opportunity_id"],
            ["managed_opportunities.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_managed_context_reviews_opportunity_id",
        "managed_context_reviews",
        ["opportunity_id"],
        unique=False,
    )
    op.create_index(
        "ix_managed_context_reviews_received_at",
        "managed_context_reviews",
        ["received_at"],
        unique=False,
    )
    op.create_index(
        "uq_managed_context_reviews_response",
        "managed_context_reviews",
        ["opportunity_id", "response_hash"],
        unique=True,
    )
    op.create_index(
        "uq_managed_context_reviews_critic",
        "managed_context_reviews",
        ["opportunity_id", "model_version", "prompt_version"],
        unique=True,
    )
    if op.get_bind().dialect.name == "sqlite":
        op.execute(
            """
            CREATE TRIGGER managed_context_reviews_no_update
            BEFORE UPDATE ON managed_context_reviews
            BEGIN
                SELECT RAISE(ABORT, 'managed_context_reviews are immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_context_reviews_no_delete
            BEFORE DELETE ON managed_context_reviews
            BEGIN
                SELECT RAISE(ABORT, 'managed_context_reviews are immutable');
            END
            """
        )

    op.create_table(
        "managed_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("artifact_hash", sa.Text(), nullable=False),
        sa.Column("feature_schema_version", sa.Text(), nullable=False),
        sa.Column("outcome_policy_version", sa.Text(), nullable=False),
        sa.Column("trained_from_session", sa.Text(), nullable=False),
        sa.Column("trained_through_session", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('challenger','promoted','rejected','retired')",
            name="ck_managed_models_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_version"),
    )
    op.create_index(
        "uq_managed_models_one_promoted",
        "managed_models",
        ["status"],
        unique=True,
        sqlite_where=sa.text("status = 'promoted'"),
        postgresql_where=sa.text("status = 'promoted'"),
    )
    op.create_index(
        "uq_managed_models_one_base_challenger",
        "managed_models",
        ["status"],
        unique=True,
        sqlite_where=sa.text(
            "status = 'challenger' AND json_extract(metrics_json, '$.model_role') = 'causal_base'"
        ),
        postgresql_where=sa.text(
            "status = 'challenger' AND (metrics_json ->> 'model_role') = 'causal_base'"
        ),
    )

    op.create_table(
        "managed_model_evaluations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("evaluation_kind", sa.Text(), nullable=False),
        sa.Column("fold_index", sa.Integer(), nullable=False),
        sa.Column("train_from_session", sa.Text(), nullable=True),
        sa.Column("train_through_session", sa.Text(), nullable=True),
        sa.Column("test_from_session", sa.Text(), nullable=False),
        sa.Column("test_through_session", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "evaluation_kind IN ('walk_forward','holdout','paper_shadow')",
            name="ck_managed_model_evaluations_kind",
        ),
        sa.CheckConstraint(
            "fold_index >= 0",
            name="ck_managed_model_evaluations_fold",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["managed_models.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_managed_model_evaluations_model_id",
        "managed_model_evaluations",
        ["model_id"],
        unique=False,
    )
    op.create_index(
        "uq_managed_model_evaluations_fold",
        "managed_model_evaluations",
        ["model_id", "evaluation_kind", "fold_index"],
        unique=True,
    )
    if op.get_bind().dialect.name == "sqlite":
        op.execute(
            """
            CREATE TRIGGER managed_opportunities_protect_identity
            BEFORE UPDATE ON managed_opportunities
            WHEN OLD.opportunity_key IS NOT NEW.opportunity_key
              OR OLD.signal_id IS NOT NEW.signal_id
              OR OLD.session IS NOT NEW.session
              OR OLD.symbol IS NOT NEW.symbol
              OR OLD.direction IS NOT NEW.direction
              OR OLD.setup_type IS NOT NEW.setup_type
              OR OLD.strategy IS NOT NEW.strategy
              OR OLD.strategy_score_id IS NOT NEW.strategy_score_id
              OR OLD.structure_hash IS NOT NEW.structure_hash
              OR OLD.legs_json IS NOT NEW.legs_json
              OR OLD.features_json IS NOT NEW.features_json
              OR OLD.policy_version IS NOT NEW.policy_version
              OR OLD.created_at IS NOT NEW.created_at
              OR OLD.detected_at IS NOT NEW.detected_at
              OR OLD.decision_batch_id IS NOT NEW.decision_batch_id
              OR OLD.decision_score IS NOT NEW.decision_score
              OR OLD.decision_defined_risk IS NOT NEW.decision_defined_risk
              OR OLD.decision_max_loss IS NOT NEW.decision_max_loss
              OR ((OLD.decision_account_value_available IS NOT NEW.decision_account_value_available
                   OR OLD.decision_account_value_usd IS NOT NEW.decision_account_value_usd)
                  AND NOT (OLD.bot_decided_at IS NULL AND NEW.bot_decided_at IS NOT NULL))
              OR OLD.baseline_action IS NOT NEW.baseline_action
              OR OLD.baseline_reason IS NOT NEW.baseline_reason
              OR OLD.admission_eligible IS NOT NEW.admission_eligible
              OR OLD.shadow_only IS NOT NEW.shadow_only
              OR (OLD.bot_decided_at IS NOT NULL AND (
                    OLD.bot_action IS NOT NEW.bot_action
                 OR OLD.bot_reason IS NOT NEW.bot_reason
                 OR OLD.bot_decided_at IS NOT NEW.bot_decided_at
              ))
              OR OLD.session_close_at IS NOT NEW.session_close_at
              OR OLD.entry_cutoff_at IS NOT NEW.entry_cutoff_at
              OR OLD.timeout_at IS NOT NEW.timeout_at
              OR OLD.stop_pct IS NOT NEW.stop_pct
              OR OLD.target_pct IS NOT NEW.target_pct
              OR OLD.commission_estimate IS NOT NEW.commission_estimate
            BEGIN
                SELECT RAISE(ABORT, 'managed opportunity identity is immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_opportunities_no_delete
            BEFORE DELETE ON managed_opportunities
            BEGIN
                SELECT RAISE(ABORT, 'managed opportunities are immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_strategy_scores_protect_structure
            BEFORE UPDATE OF snapshot_id, strategy, score, rationale, legs_json, suggestion_json
            ON strategy_scores
            WHEN EXISTS (
                SELECT 1 FROM managed_opportunities
                 WHERE managed_opportunities.strategy_score_id = OLD.id
                   AND (
                        OLD.snapshot_id IS NOT NEW.snapshot_id
                     OR OLD.strategy IS NOT NEW.strategy
                     OR OLD.legs_json IS NOT NEW.legs_json
                     OR (
                         managed_opportunities.bot_decided_at IS NOT NULL
                         AND (
                              OLD.score IS NOT NEW.score
                           OR OLD.rationale IS NOT NEW.rationale
                           OR OLD.suggestion_json IS NOT NEW.suggestion_json
                         )
                     )
                   )
            )
            BEGIN
                SELECT RAISE(ABORT, 'managed strategy score structure is immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_opportunities_terminal_immutable
            BEFORE UPDATE ON managed_opportunities
            WHEN OLD.status IN ('resolved','censored','unobservable')
             AND (
                   OLD.status IS NOT NEW.status
                OR OLD.outcome IS NOT NEW.outcome
                OR OLD.resolved_at IS NOT NEW.resolved_at
                OR OLD.entry_ts IS NOT NEW.entry_ts
                OR OLD.entry_combo_bid IS NOT NEW.entry_combo_bid
                OR OLD.entry_combo_ask IS NOT NEW.entry_combo_ask
                OR OLD.entry_net IS NOT NEW.entry_net
                OR OLD.basis_dollars IS NOT NEW.basis_dollars
                OR OLD.exit_net IS NOT NEW.exit_net
                OR OLD.gross_pnl IS NOT NEW.gross_pnl
                OR OLD.net_pnl IS NOT NEW.net_pnl
                OR OLD.mfe_dollars IS NOT NEW.mfe_dollars
                OR OLD.mae_dollars IS NOT NEW.mae_dollars
                OR OLD.last_valid_mark_at IS NOT NEW.last_valid_mark_at
                OR OLD.valid_marks IS NOT NEW.valid_marks
                OR OLD.max_mark_gap_seconds IS NOT NEW.max_mark_gap_seconds
                OR OLD.training_eligible IS NOT NEW.training_eligible
                OR OLD.resolution_reason IS NOT NEW.resolution_reason
             )
            BEGIN
                SELECT RAISE(ABORT, 'managed opportunity terminal label is immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_strategy_scores_no_delete
            BEFORE DELETE ON strategy_scores
            WHEN EXISTS (
                SELECT 1 FROM managed_opportunities
                 WHERE managed_opportunities.strategy_score_id = OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'managed strategy score cannot be deleted');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_opportunity_marks_no_update
            BEFORE UPDATE ON managed_opportunity_marks
            BEGIN
                SELECT RAISE(ABORT, 'managed opportunity marks are immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_opportunity_marks_no_delete
            BEFORE DELETE ON managed_opportunity_marks
            BEGIN
                SELECT RAISE(ABORT, 'managed opportunity marks are immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_models_protect_artifact
            BEFORE UPDATE ON managed_models
            WHEN OLD.model_version IS NOT NEW.model_version
              OR OLD.artifact_hash IS NOT NEW.artifact_hash
              OR OLD.feature_schema_version IS NOT NEW.feature_schema_version
              OR OLD.outcome_policy_version IS NOT NEW.outcome_policy_version
              OR OLD.trained_from_session IS NOT NEW.trained_from_session
              OR OLD.trained_through_session IS NOT NEW.trained_through_session
              OR OLD.metrics_json IS NOT NEW.metrics_json
              OR OLD.created_at IS NOT NEW.created_at
            BEGIN
                SELECT RAISE(ABORT, 'managed model artifacts are immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_models_no_delete
            BEFORE DELETE ON managed_models
            BEGIN
                SELECT RAISE(ABORT, 'managed models must be retired, not deleted');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_model_evaluations_no_update
            BEFORE UPDATE ON managed_model_evaluations
            BEGIN
                SELECT RAISE(ABORT, 'managed model evaluations are immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER managed_model_evaluations_no_delete
            BEFORE DELETE ON managed_model_evaluations
            BEGIN
                SELECT RAISE(ABORT, 'managed model evaluations are immutable');
            END
            """
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS managed_model_evaluations_no_delete")
        op.execute("DROP TRIGGER IF EXISTS managed_model_evaluations_no_update")
        op.execute("DROP TRIGGER IF EXISTS managed_models_no_delete")
        op.execute("DROP TRIGGER IF EXISTS managed_models_protect_artifact")
        op.execute("DROP TRIGGER IF EXISTS managed_opportunity_marks_no_delete")
        op.execute("DROP TRIGGER IF EXISTS managed_opportunity_marks_no_update")
        op.execute("DROP TRIGGER IF EXISTS managed_strategy_scores_no_delete")
        op.execute("DROP TRIGGER IF EXISTS managed_strategy_scores_protect_structure")
        op.execute("DROP TRIGGER IF EXISTS managed_opportunities_terminal_immutable")
        op.execute("DROP TRIGGER IF EXISTS managed_opportunities_no_delete")
        op.execute("DROP TRIGGER IF EXISTS managed_opportunities_protect_identity")
    op.drop_index(
        "uq_managed_model_evaluations_fold",
        table_name="managed_model_evaluations",
    )
    op.drop_index(
        "ix_managed_model_evaluations_model_id",
        table_name="managed_model_evaluations",
    )
    op.drop_table("managed_model_evaluations")
    op.drop_index(
        "uq_managed_models_one_base_challenger",
        table_name="managed_models",
    )
    op.drop_index("uq_managed_models_one_promoted", table_name="managed_models")
    op.drop_table("managed_models")
    if op.get_bind().dialect.name == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS managed_context_reviews_no_delete")
        op.execute("DROP TRIGGER IF EXISTS managed_context_reviews_no_update")
    op.drop_index(
        "uq_managed_context_reviews_critic",
        table_name="managed_context_reviews",
    )
    op.drop_index(
        "uq_managed_context_reviews_response",
        table_name="managed_context_reviews",
    )
    op.drop_index(
        "ix_managed_context_reviews_received_at",
        table_name="managed_context_reviews",
    )
    op.drop_index(
        "ix_managed_context_reviews_opportunity_id",
        table_name="managed_context_reviews",
    )
    op.drop_table("managed_context_reviews")
    op.drop_index(
        "uq_managed_opportunity_marks_bucket",
        table_name="managed_opportunity_marks",
    )
    op.drop_index(
        "ix_managed_opportunity_marks_observed_at",
        table_name="managed_opportunity_marks",
    )
    op.drop_index(
        "ix_managed_opportunity_marks_opportunity_id",
        table_name="managed_opportunity_marks",
    )
    op.drop_table("managed_opportunity_marks")
    op.drop_index(
        "ix_managed_opportunities_status_timeout",
        table_name="managed_opportunities",
    )
    op.drop_index(
        "ix_managed_opportunities_signal_strategy",
        table_name="managed_opportunities",
    )
    op.drop_index(
        "ix_managed_opportunities_strategy_score_id",
        table_name="managed_opportunities",
    )
    op.drop_index("ix_managed_opportunities_symbol", table_name="managed_opportunities")
    op.drop_index("ix_managed_opportunities_session", table_name="managed_opportunities")
    op.drop_index("ix_managed_opportunities_signal_id", table_name="managed_opportunities")
    op.drop_table("managed_opportunities")
