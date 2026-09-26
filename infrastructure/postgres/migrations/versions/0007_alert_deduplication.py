"""Add active-alert deduplication keys."""

import sqlalchemy as sa
from alembic import op

revision = "0007_alert_deduplication"
down_revision = "0006_alert_severity_fix"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("alert", sa.Column("deduplication_key", sa.Text(), nullable=True))
    op.execute("UPDATE alert SET deduplication_key = 'legacy:' || alert_id")
    op.alter_column("alert", "deduplication_key", nullable=False)
    op.create_index(
        "uq_alert_active_deduplication_key",
        "alert",
        ["deduplication_key"],
        unique=True,
        postgresql_where=sa.text("resolved_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_alert_active_deduplication_key", table_name="alert")
    op.drop_column("alert", "deduplication_key")
