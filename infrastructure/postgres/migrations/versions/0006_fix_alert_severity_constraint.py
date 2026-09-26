"""Correct the alert severity constraint introduced in the lifecycle migration."""

from alembic import op

revision = "0006_alert_severity_fix"
down_revision = "0005_alert_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE alert
            DROP CONSTRAINT IF EXISTS ck_alert_severity,
            ADD CONSTRAINT ck_alert_severity
                CHECK (lower(severity) IN ('info', 'warning', 'error', 'critical'));
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE alert
            DROP CONSTRAINT IF EXISTS ck_alert_severity,
            ADD CONSTRAINT ck_alert_severity
                CHECK (lower(severity) IN ('info', 'warning', 'error', 'critical'));
        """
    )
