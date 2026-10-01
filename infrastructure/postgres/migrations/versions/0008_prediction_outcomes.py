"""Store delayed observed RUL outcomes for model-performance monitoring."""

from alembic import op

revision = "0008_prediction_outcomes"
down_revision = "0007_alert_deduplication"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE prediction_outcome (
            outcome_id BIGSERIAL PRIMARY KEY,
            prediction_id BIGINT NOT NULL REFERENCES prediction(prediction_id),
            actual_rul_cycles DOUBLE PRECISION NOT NULL CHECK (actual_rul_cycles >= 0),
            observed_at TIMESTAMPTZ NOT NULL,
            source TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (prediction_id)
        );

        CREATE INDEX ix_prediction_outcome_observed_at
            ON prediction_outcome (observed_at DESC);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS prediction_outcome")
