"""Add feature and data-quality lineage to predictions.

Revision ID: 0004_prediction_lineage
Revises: 26ef8994f2d0
"""

from alembic import op

revision = "0004_prediction_lineage"
down_revision = "26ef8994f2d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE prediction
            ADD COLUMN feature_version TEXT,
            ADD COLUMN data_quality_status TEXT,
            ADD COLUMN quality_flags JSONB NOT NULL DEFAULT '[]'::jsonb;

        UPDATE prediction
        SET
            feature_version = 'features-v1',
            data_quality_status = CASE
                WHEN lower(status) = 'withheld' THEN 'invalid'
                WHEN lower(status) = 'degraded' THEN 'degraded'
                ELSE 'valid'
            END;

        ALTER TABLE prediction
            ALTER COLUMN feature_version SET NOT NULL,
            ALTER COLUMN data_quality_status SET NOT NULL,
            ADD CONSTRAINT ck_prediction_data_quality_status
                CHECK (data_quality_status IN ('valid', 'degraded', 'invalid')),
            ADD CONSTRAINT ck_prediction_quality_flags_array
                CHECK (jsonb_typeof(quality_flags) = 'array');
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE prediction
            DROP CONSTRAINT IF EXISTS ck_prediction_quality_flags_array,
            DROP CONSTRAINT IF EXISTS ck_prediction_data_quality_status,
            DROP COLUMN IF EXISTS quality_flags,
            DROP COLUMN IF EXISTS data_quality_status,
            DROP COLUMN IF EXISTS feature_version;
        """
    )
