"""Tests for the Alembic migration graph."""

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migrations_form_one_complete_chain() -> None:
    script = ScriptDirectory.from_config(Config("alembic.ini"))

    assert script.get_heads() == ["0009_telemetry_measurements"]
    assert [revision.revision for revision in script.walk_revisions()] == [
        "0009_telemetry_measurements",
        "0008_prediction_outcomes",
        "0007_alert_deduplication",
        "0006_alert_severity_fix",
        "0005_alert_lifecycle",
        "0004_prediction_lineage",
        "26ef8994f2d0",
        "0003_pending_events",
        "0002_quality_conflict_fp",
        "0001_initial_operational_schema",
    ]
