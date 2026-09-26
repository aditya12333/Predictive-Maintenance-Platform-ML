from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from predictive_maintenance.api.app import create_app
from predictive_maintenance.core.settings import load_settings
from predictive_maintenance.storage.database import create_database_engine


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


def test_ingestion_api_validates_and_deduplicates(database_engine) -> None:
    event_id = f"api-integration-{uuid4()}"
    client = TestClient(create_app(engine=database_engine))
    payload = {
        "event_id": event_id,
        "engine_id": 1,
        "cycle": 1,
        "event_timestamp": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "schema_version": "telemetry-v1",
        "source_id": "integration-test",
        "measurements": {"sensor_1": 0.5},
    }
    try:
        first = client.post("/v1/telemetry", json=payload)
        assert first.status_code == 202
        assert first.json()["persistence_outcome"] == "inserted"

        duplicate = client.post("/v1/telemetry", json=payload)
        assert duplicate.status_code == 202
        assert duplicate.json()["persistence_outcome"] == "duplicate"

        invalid = {**payload, "event_id": f"{event_id}-invalid", "engine_id": 0}
        assert client.post("/v1/telemetry", json=invalid).status_code == 422
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM telemetry_event WHERE event_id LIKE :prefix"),
                {"prefix": f"{event_id}%"},
            )


def test_readiness_checks_database(database_engine) -> None:
    client = TestClient(create_app(engine=database_engine))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_operational_endpoints_expose_unscored_engine_state(database_engine) -> None:
    event_id = f"api-operational-{uuid4()}"
    engine_id = 90_000 + uuid4().int % 9_000
    client = TestClient(create_app(engine=database_engine))
    payload = {
        "event_id": event_id,
        "engine_id": engine_id,
        "cycle": 1,
        "event_timestamp": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "schema_version": "telemetry-v1",
        "source_id": "integration-test",
        "measurements": {"sensor_1": 0.5},
    }
    try:
        assert client.post("/v1/telemetry", json=payload).status_code == 202

        fleet = client.get("/v1/fleet/health")
        assert fleet.status_code == 200
        fleet_engine = next(
            row for row in fleet.json()["engines"] if row["engine_id"] == engine_id
        )
        assert fleet_engine["health_status"] == "unavailable"

        detail = client.get(f"/v1/equipment/{engine_id}")
        assert detail.status_code == 200
        assert detail.json()["health_status"] == "unavailable"
        assert detail.json()["latest_prediction"] is None

        history = client.get(f"/v1/equipment/{engine_id}/predictions")
        assert history.status_code == 200
        assert history.json()["items"] == []

        alerts = client.get("/v1/alerts", params={"engine_id": engine_id})
        assert alerts.status_code == 200
        assert alerts.json()["items"] == []
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM telemetry_event WHERE event_id = :event_id"),
                {"event_id": event_id},
            )


def test_alert_transition_is_version_checked_and_audited(database_engine) -> None:
    engine_id = 90_000 + uuid4().int % 9_000
    client = TestClient(create_app(engine=database_engine))
    with database_engine.begin() as connection:
        alert_id = connection.execute(
            text(
                """
                INSERT INTO alert (
                    engine_id, alert_type, state, severity, deduplication_key, title, message
                ) VALUES (
                    :engine_id, 'data_quality', 'OPEN', 'WARNING', :deduplication_key,
                    'Missing sensor', 'Sensor 7 is missing'
                )
                RETURNING alert_id
                """
            ),
            {"engine_id": engine_id, "deduplication_key": f"test-api-data-quality:{engine_id}"},
        ).scalar_one()

    try:
        acknowledged = client.patch(
            f"/v1/alerts/{alert_id}",
            json={
                "action": "acknowledge",
                "expected_version": 1,
                "actor_id": "maintenance-user-1",
                "reason": "Inspection started.",
            },
        )
        assert acknowledged.status_code == 200
        assert acknowledged.json()["state"] == "acknowledged"
        assert acknowledged.json()["version"] == 2
        assert acknowledged.json()["acknowledged_by"] == "maintenance-user-1"

        stale = client.patch(
            f"/v1/alerts/{alert_id}",
            json={
                "action": "dismiss",
                "expected_version": 1,
                "actor_id": "another-user",
            },
        )
        assert stale.status_code == 409

        resolved = client.patch(
            f"/v1/alerts/{alert_id}",
            json={
                "action": "resolve",
                "expected_version": 2,
                "actor_id": "maintenance-user-1",
                "reason": "Sensor replaced.",
            },
        )
        assert resolved.status_code == 200
        assert resolved.json()["state"] == "resolved"
        assert resolved.json()["version"] == 3
        assert resolved.json()["resolved_at"] is not None

        closed_transition = client.patch(
            f"/v1/alerts/{alert_id}",
            json={
                "action": "acknowledge",
                "expected_version": 3,
                "actor_id": "maintenance-user-1",
            },
        )
        assert closed_transition.status_code == 409

        with database_engine.connect() as connection:
            audit_count = connection.execute(
                text(
                    """
                    SELECT count(*)
                    FROM audit_event
                    WHERE subject_type = 'alert' AND subject_id = :subject_id
                    """
                ),
                {"subject_id": str(alert_id)},
            ).scalar_one()
        assert audit_count == 2
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM audit_event "
                    "WHERE subject_type = 'alert' AND subject_id = :subject_id"
                ),
                {"subject_id": str(alert_id)},
            )
            connection.execute(
                text("DELETE FROM alert WHERE alert_id = :alert_id"),
                {"alert_id": alert_id},
            )
