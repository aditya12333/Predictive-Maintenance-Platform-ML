"""Add evidence, acknowledgement and optimistic concurrency to alerts."""

from alembic import op

revision = "0005_alert_lifecycle"
down_revision = "0004_prediction_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE alert
            ADD COLUMN title TEXT,
            ADD COLUMN message TEXT,
            ADD COLUMN source_event_id TEXT REFERENCES telemetry_event(event_id),
            ADD COLUMN source_prediction_id BIGINT REFERENCES prediction(prediction_id),
            ADD COLUMN acknowledged_at TIMESTAMPTZ,
            ADD COLUMN acknowledged_by TEXT,
            ADD COLUMN version INTEGER NOT NULL DEFAULT 1;

        UPDATE alert
        SET
            title = initcap(replace(alert_type, '_', ' ')),
            message = initcap(replace(alert_type, '_', ' '));

        ALTER TABLE alert
            ALTER COLUMN title SET NOT NULL,
            ALTER COLUMN message SET NOT NULL,
            ADD CONSTRAINT ck_alert_state
                CHECK (upper(state) IN ('OPEN', 'ACKNOWLEDGED', 'RESOLVED', 'DISMISSED')),
            ADD CONSTRAINT ck_alert_severity
                CHECK (lower(severity) IN ('info', 'warning', 'error', 'critical')),
            ADD CONSTRAINT ck_alert_version_positive
                CHECK (version > 0);

        CREATE INDEX ix_alert_state_updated
            ON alert (state, updated_at DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS ix_alert_state_updated;
        ALTER TABLE alert
            DROP CONSTRAINT IF EXISTS ck_alert_version_positive,
            DROP CONSTRAINT IF EXISTS ck_alert_severity,
            DROP CONSTRAINT IF EXISTS ck_alert_state,
            DROP COLUMN IF EXISTS version,
            DROP COLUMN IF EXISTS acknowledged_by,
            DROP COLUMN IF EXISTS acknowledged_at,
            DROP COLUMN IF EXISTS source_prediction_id,
            DROP COLUMN IF EXISTS source_event_id,
            DROP COLUMN IF EXISTS message,
            DROP COLUMN IF EXISTS title;
        """
    )
