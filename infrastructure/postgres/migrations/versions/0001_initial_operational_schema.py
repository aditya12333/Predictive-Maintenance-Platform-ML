"""Create the initial operational schema.

Revision ID: 0001_initial_operational_schema
"""

from alembic import op

revision = "0001_initial_operational_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE telemetry_event (
            event_id TEXT PRIMARY KEY,
            engine_id INTEGER NOT NULL CHECK (engine_id > 0),
            cycle INTEGER NOT NULL CHECK (cycle > 0),
            event_timestamp TIMESTAMPTZ NOT NULL,
            ingestion_timestamp TIMESTAMPTZ NOT NULL,
            schema_version TEXT NOT NULL,
            source_id TEXT NOT NULL,
            payload_digest TEXT NOT NULL,
            processing_status TEXT NOT NULL DEFAULT 'RECEIVED',
            quality_status TEXT NOT NULL DEFAULT 'UNKNOWN',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX ix_telemetry_event_engine_time
            ON telemetry_event (engine_id, event_timestamp DESC);

        CREATE TABLE prediction (
            prediction_id BIGSERIAL PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES telemetry_event(event_id),
            model_release TEXT NOT NULL,
            generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            rul_cycles DOUBLE PRECISION,
            risk_score DOUBLE PRECISION,
            status TEXT NOT NULL,
            withheld_reason TEXT,
            UNIQUE (event_id, model_release)
        );

        CREATE TABLE data_quality_issue (
            issue_id BIGSERIAL PRIMARY KEY,
            event_id TEXT REFERENCES telemetry_event(event_id),
            engine_id INTEGER CHECK (engine_id IS NULL OR engine_id > 0),
            issue_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved_at TIMESTAMPTZ
        );

        CREATE INDEX ix_quality_issue_open
            ON data_quality_issue (engine_id, detected_at DESC)
            WHERE resolved_at IS NULL;

        CREATE TABLE alert (
            alert_id BIGSERIAL PRIMARY KEY,
            engine_id INTEGER NOT NULL CHECK (engine_id > 0),
            alert_type TEXT NOT NULL,
            state TEXT NOT NULL,
            severity TEXT NOT NULL,
            opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved_at TIMESTAMPTZ
        );

        CREATE INDEX ix_alert_active_by_engine
            ON alert (engine_id, updated_at DESC)
            WHERE resolved_at IS NULL;

        CREATE TABLE outbox_event (
            outbox_id BIGSERIAL PRIMARY KEY,
            event_type TEXT NOT NULL,
            aggregate_type TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            published_at TIMESTAMPTZ,
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            last_error TEXT
        );

        CREATE INDEX ix_outbox_pending
            ON outbox_event (created_at)
            WHERE published_at IS NULL;

        CREATE TABLE audit_event (
            audit_id BIGSERIAL PRIMARY KEY,
            event_type TEXT NOT NULL,
            actor_id TEXT,
            subject_type TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            details JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS audit_event;
        DROP TABLE IF EXISTS outbox_event;
        DROP TABLE IF EXISTS alert;
        DROP TABLE IF EXISTS data_quality_issue;
        DROP TABLE IF EXISTS prediction;
        DROP TABLE IF EXISTS telemetry_event;
        """
    )
