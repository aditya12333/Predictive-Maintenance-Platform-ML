"""Add a fingerprint for idempotent data-quality conflicts."""

from alembic import op

revision = "0002_quality_conflict_fp"
down_revision = "0001_initial_operational_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE data_quality_issue ADD COLUMN fingerprint TEXT")
    op.execute(
        """
        CREATE UNIQUE INDEX ux_quality_issue_conflict
            ON data_quality_issue (event_id, issue_type, fingerprint)
            WHERE issue_type = 'conflicting_duplicate' AND resolved_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ux_quality_issue_conflict")
    op.execute("ALTER TABLE data_quality_issue DROP COLUMN IF EXISTS fingerprint")
