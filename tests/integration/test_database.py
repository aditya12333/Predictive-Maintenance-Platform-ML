from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text

from predictive_maintenance.core.settings import load_settings
from predictive_maintenance.storage.database import (
    EventPersistenceOutcome,
    TelemetryEventRecord,
    create_database_engine,
    persist_telemetry_event,
    record_ordering_window_expired_issue,
)


@pytest.fixture()
def database_engine():
    engine = create_database_engine(load_settings())
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - depends on local infrastructure
        pytest.skip(f"local PostgreSQL is unavailable: {exc}")
    yield engine
    engine.dispose()


def test_telemetry_insert_is_idempotent_and_conflicts_are_detected(database_engine) -> None:
    event_id = f"integration-{uuid4()}"
    event = TelemetryEventRecord(
        event_id=event_id,
        engine_id=1,
        cycle=1,
        event_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        schema_version="telemetry-v1",
        source_id="integration-test",
        payload_digest="digest-a",
    )
    try:
        assert persist_telemetry_event(database_engine, event) == EventPersistenceOutcome.INSERTED
        assert persist_telemetry_event(database_engine, event) == EventPersistenceOutcome.DUPLICATE
        conflicting = TelemetryEventRecord(**{**event.__dict__, "payload_digest": "digest-b"})
        assert (
            persist_telemetry_event(database_engine, conflicting)
            == EventPersistenceOutcome.CONFLICT
        )
        with database_engine.connect() as connection:
            issue_count = connection.execute(
                text(
                    "SELECT count(*) FROM data_quality_issue "
                    "WHERE event_id = :event_id AND issue_type = 'conflicting_duplicate'"
                ),
                {"event_id": event_id},
            ).scalar_one()
        assert issue_count == 1
        assert (
            persist_telemetry_event(database_engine, conflicting)
            == EventPersistenceOutcome.CONFLICT
        )
        with database_engine.connect() as connection:
            repeated_issue_count = connection.execute(
                text(
                    "SELECT count(*) FROM data_quality_issue "
                    "WHERE event_id = :event_id AND issue_type = 'conflicting_duplicate'"
                ),
                {"event_id": event_id},
            ).scalar_one()
        assert repeated_issue_count == 1
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM data_quality_issue WHERE event_id = :event_id"),
                {"event_id": event_id},
            )
            connection.execute(
                text("DELETE FROM telemetry_event WHERE event_id = :event_id"),
                {"event_id": event_id},
            )


def test_missing_cycle_is_recorded_without_blocking_ingestion(database_engine) -> None:
    first_id = f"gap-first-{uuid4()}"
    later_id = f"gap-later-{uuid4()}"
    base = dict(
        engine_id=999,
        event_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        schema_version="telemetry-v1",
        source_id="integration-test",
    )
    try:
        assert (
            persist_telemetry_event(
                database_engine,
                TelemetryEventRecord(event_id=first_id, cycle=1, payload_digest="first", **base),
            )
            == EventPersistenceOutcome.INSERTED
        )
        assert (
            persist_telemetry_event(
                database_engine,
                TelemetryEventRecord(event_id=later_id, cycle=3, payload_digest="later", **base),
            )
            == EventPersistenceOutcome.INSERTED
        )
        with database_engine.connect() as connection:
            issue = connection.execute(
                text(
                    "SELECT message FROM data_quality_issue "
                    "WHERE event_id = :event_id AND issue_type = 'cycle_gap'"
                ),
                {"event_id": later_id},
            ).scalar_one()
        assert "Cycles 2 through 2" in issue
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM data_quality_issue WHERE event_id IN (:first, :later)"),
                {"first": first_id, "later": later_id},
            )
            connection.execute(
                text("DELETE FROM telemetry_event WHERE event_id IN (:first, :later)"),
                {"first": first_id, "later": later_id},
            )


def test_ordering_window_warning_is_idempotent(database_engine) -> None:
    event_id = f"idle-warning-{uuid4()}"
    event = TelemetryEventRecord(
        event_id=event_id,
        engine_id=9101,
        cycle=1,
        event_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        schema_version="telemetry-v1",
        source_id="integration-test",
        payload_digest="idle-warning-digest",
    )

    try:
        assert persist_telemetry_event(database_engine, event) == EventPersistenceOutcome.INSERTED

        record_ordering_window_expired_issue(
            database_engine,
            event,
            idle_seconds=301.0,
        )
        record_ordering_window_expired_issue(
            database_engine,
            event,
            idle_seconds=301.0,
        )

        with database_engine.connect() as connection:
            issue_count = connection.execute(
                text(
                    "SELECT count(*) FROM data_quality_issue "
                    "WHERE event_id = :event_id "
                    "AND issue_type = 'ordering_window_expired'"
                ),
                {"event_id": event_id},
            ).scalar_one()

        assert issue_count == 1
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM data_quality_issue WHERE event_id = :event_id"),
                {"event_id": event_id},
            )
            connection.execute(
                text("DELETE FROM telemetry_event WHERE event_id = :event_id"),
                {"event_id": event_id},
            )
