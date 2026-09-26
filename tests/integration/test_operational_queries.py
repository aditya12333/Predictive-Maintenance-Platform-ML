"""PostgreSQL integration tests for dashboard read models."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from predictive_maintenance.api.contracts import (
    AlertState,
    AlertType,
    FleetHealthStatus,
)
from predictive_maintenance.core.settings import load_settings
from predictive_maintenance.storage.database import create_database_engine
from predictive_maintenance.storage.operational import (
    load_alerts,
    load_equipment_detail,
    load_fleet_health,
    load_prediction_history,
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


def test_operational_queries_assemble_fleet_and_engine_views(database_engine) -> None:
    suffix = uuid4().hex
    engine_id = 90_000 + uuid4().int % 9_000
    first_event = f"operational-query-{suffix}-1"
    second_event = f"operational-query-{suffix}-2"
    first_time = datetime(2026, 9, 24, 10, tzinfo=UTC)
    second_time = first_time + timedelta(minutes=1)

    try:
        with database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO telemetry_event (
                        event_id, engine_id, cycle, event_timestamp, ingestion_timestamp,
                        schema_version, source_id, payload_digest, processing_status, quality_status
                    ) VALUES
                        (:first_event, :engine_id, 1, :first_time, :first_time,
                         'telemetry-v1', 'query-test', :first_digest, 'PROCESSED', 'VALID'),
                        (:second_event, :engine_id, 2, :second_time, :second_time,
                         'telemetry-v1', 'query-test', :second_digest, 'PROCESSED', 'DEGRADED')
                    """
                ),
                {
                    "first_event": first_event,
                    "second_event": second_event,
                    "engine_id": engine_id,
                    "first_time": first_time,
                    "second_time": second_time,
                    "first_digest": f"digest-{first_event}",
                    "second_digest": f"digest-{second_event}",
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO prediction (
                        event_id, model_release, generated_at, rul_cycles, status,
                        feature_version, data_quality_status, quality_flags
                    ) VALUES
                        (:first_event, 'rul-lightgbm-v1', :first_time, 20.0, 'available',
                         'features-v1', 'valid', '[]'::jsonb),
                        (:second_event, 'rul-lightgbm-v1', :second_time, NULL, 'withheld',
                         'features-v1', 'invalid', '["missing_sensor_7"]'::jsonb)
                    """
                ),
                {
                    "first_event": first_event,
                    "second_event": second_event,
                    "first_time": first_time,
                    "second_time": second_time,
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO data_quality_issue (
                        event_id, engine_id, issue_type, severity, message
                    ) VALUES (
                        :event_id, :engine_id, 'missing_sensor', 'WARNING', 'Sensor 7 missing'
                    )
                    """
                ),
                {"event_id": second_event, "engine_id": engine_id},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO alert (
                        engine_id, alert_type, state, severity, deduplication_key, title, message
                    )
                    VALUES (
                        :engine_id, 'data_quality', 'OPEN', 'WARNING', :deduplication_key,
                        'Data quality', 'Sensor 7 missing'
                    )
                    """
                ),
                {"engine_id": engine_id, "deduplication_key": f"test-data-quality:{engine_id}"},
            )

        as_of = datetime(2026, 9, 24, 10, 5, tzinfo=UTC)
        detail = load_equipment_detail(
            database_engine,
            engine_id=engine_id,
            as_of=as_of,
            stale_after=timedelta(hours=1),
        )
        fleet = load_fleet_health(
            database_engine,
            as_of=as_of,
            stale_after=timedelta(hours=1),
        )
        history = load_prediction_history(database_engine, engine_id=engine_id)
        alerts = load_alerts(
            database_engine,
            engine_id=engine_id,
            state=AlertState.OPEN,
        )

        assert detail.health_status is FleetHealthStatus.WITHHELD
        assert detail.latest_prediction is not None
        assert detail.latest_prediction.estimated_rul is None
        assert detail.last_valid_prediction is not None
        assert detail.last_valid_prediction.estimated_rul == 20.0
        assert detail.open_data_quality_issues[0].issue_type == "missing_sensor"
        assert detail.active_alerts[0].alert_type is AlertType.DATA_QUALITY
        assert history.items[0].status.value == "withheld"
        assert len(history.items) == 2
        assert alerts.items[0].state is AlertState.OPEN
        fleet_row = next(row for row in fleet.engines if row.engine_id == engine_id)
        assert fleet_row.health_status is FleetHealthStatus.WITHHELD
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM alert WHERE engine_id = :engine_id"),
                {"engine_id": engine_id},
            )
            connection.execute(
                text("DELETE FROM data_quality_issue WHERE event_id = :event_id"),
                {"event_id": second_event},
            )
            connection.execute(
                text("DELETE FROM prediction WHERE event_id IN (:first_event, :second_event)"),
                {"first_event": first_event, "second_event": second_event},
            )
            connection.execute(
                text(
                    "DELETE FROM telemetry_event WHERE event_id IN (:first_event, :second_event)"
                ),
                {"first_event": first_event, "second_event": second_event},
            )
