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
