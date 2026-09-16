"""Create durable storage for events awaiting event-time ordering."""

from alembic import op

revision = "0003_pending_events"
down_revision = "0002_quality_conflict_fp"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE pending_event (
            event_id TEXT PRIMARY KEY,
            engine_id INTEGER NOT NULL CHECK (engine_id > 0),
            cycle INTEGER NOT NULL CHECK (cycle > 0),
            event_timestamp TIMESTAMPTZ NOT NULL,
            received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            payload_digest TEXT NOT NULL,
            payload JSONB NOT NULL
        );
        CREATE INDEX ix_pending_event_engine_cycle
            ON pending_event (engine_id, cycle);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pending_event")
