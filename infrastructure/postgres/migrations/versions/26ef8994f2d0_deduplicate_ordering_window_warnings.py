"""Deduplicate ordering window warnings

Revision ID: 26ef8994f2d0
Revises: 0003_pending_events
Create Date: 2026-09-12 15:27:24.147469
"""

from collections.abc import Sequence

from alembic import op

revision: str = "26ef8994f2d0"
down_revision: str | None = "0003_pending_events"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX ux_quality_issue_ordering_window
            ON data_quality_issue (event_id, issue_type)
            WHERE issue_type = 'ordering_window_expired'
              AND resolved_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ux_quality_issue_ordering_window")
