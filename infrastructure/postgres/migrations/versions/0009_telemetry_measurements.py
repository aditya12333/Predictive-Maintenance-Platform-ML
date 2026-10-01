"""Persist validated sensor measurements for feature-drift monitoring."""

from alembic import op

revision = "0009_telemetry_measurements"
down_revision = "0008_prediction_outcomes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE telemetry_event
            ADD COLUMN measurements JSONB;

        ALTER TABLE telemetry_event
            ADD CONSTRAINT ck_telemetry_measurements_object
            CHECK (measurements IS NULL OR jsonb_typeof(measurements) = 'object');
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE telemetry_event
            DROP CONSTRAINT IF EXISTS ck_telemetry_measurements_object,
            DROP COLUMN IF EXISTS measurements;
        """
    )
