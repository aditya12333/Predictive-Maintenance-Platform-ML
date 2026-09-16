"""Tests for the Alembic migration graph."""

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migrations_form_one_complete_chain() -> None:
    script = ScriptDirectory.from_config(Config("alembic.ini"))

    assert script.get_heads() == ["0004_prediction_lineage"]
    assert [revision.revision for revision in script.walk_revisions()] == [
        "0004_prediction_lineage",
        "26ef8994f2d0",
        "0003_pending_events",
        "0002_quality_conflict_fp",
        "0001_initial_operational_schema",
    ]
