"""PostgreSQL integration tests for automatic alert evaluation."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text

from predictive_maintenance.core.settings import load_settings
from predictive_maintenance.inference.contracts import (
    DataQualityStatus,
    PredictionRecord,
    PredictionStatus,
)
from predictive_maintenance.storage.alerts import AlertPolicy, persist_prediction_alerts_transaction
from predictive_maintenance.storage.database import create_database_engine
from predictive_maintenance.storage.predictions import persist_prediction_transaction


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


def test_prediction_alerts_are_thresholded_and_deduplicated(database_engine) -> None:
    suffix = uuid4().hex
    event_id = f"alert-evaluation-{suffix}"
    engine_id = 90_000 + uuid4().int % 9_000
    generated_at = datetime(2026, 9, 24, 10, tzinfo=UTC)
    prediction = PredictionRecord(
        event_id=event_id,
        engine_id=engine_id,
        cycle=12,
        estimated_rul=8.0,
        status=PredictionStatus.AVAILABLE,
        data_quality_status=DataQualityStatus.DEGRADED,
        quality_flags=["cycle_gap", "cycle_gap"],
        model_release="rul-test-v1",
        feature_version="features-test-v1",
        generated_at=generated_at,
    )
    try:
        with database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO telemetry_event (
                        event_id, engine_id, cycle, event_timestamp, ingestion_timestamp,
                        schema_version, source_id, payload_digest, processing_status, quality_status
                    ) VALUES (
                        :event_id, :engine_id, 12, :timestamp, :timestamp,
                        'telemetry-v1', 'alert-test', :digest, 'PROCESSED', 'DEGRADED'
                    )
                    """
                ),
                {
                    "event_id": event_id,
                    "engine_id": engine_id,
                    "timestamp": generated_at,
                    "digest": f"digest-{event_id}",
                },
            )
            persist_prediction_transaction(connection, prediction)
            policy = AlertPolicy(warning_rul_cycles=30, critical_rul_cycles=10)
            assert (
                persist_prediction_alerts_transaction(
                    connection, prediction=prediction, policy=policy, now=generated_at
                )
                == 2
            )
            assert (
                persist_prediction_alerts_transaction(
                    connection, prediction=prediction, policy=policy, now=generated_at
                )
                == 2
            )
            rows = connection.execute(
                text(
                    """
                    SELECT deduplication_key, severity, version
                    FROM alert
                    WHERE engine_id = :engine_id AND resolved_at IS NULL
                    ORDER BY deduplication_key
                    """
                ),
                {"engine_id": engine_id},
            ).mappings().all()
        assert len(rows) == 2
        assert rows[0]["deduplication_key"] == f"data-quality:{engine_id}:cycle_gap"
        assert rows[0]["version"] == 2
        assert rows[1]["deduplication_key"] == f"equipment-risk:{engine_id}"
        assert rows[1]["severity"] == "CRITICAL"
        assert rows[1]["version"] == 2
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM alert WHERE engine_id = :engine_id"),
                {"engine_id": engine_id},
            )
            connection.execute(
                text("DELETE FROM prediction WHERE event_id = :event_id"),
                {"event_id": event_id},
            )
            connection.execute(
                text("DELETE FROM telemetry_event WHERE event_id = :event_id"),
                {"event_id": event_id},
            )
