"""PostgreSQL integration tests for prediction persistence."""

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
from predictive_maintenance.storage.database import (
    TelemetryEventRecord,
    create_database_engine,
    persist_telemetry_event,
)
from predictive_maintenance.storage.predictions import (
    PredictionPersistenceOutcome,
    load_last_valid_prediction,
    persist_prediction,
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


def _telemetry(event_id: str, *, engine_id: int, cycle: int) -> TelemetryEventRecord:
    return TelemetryEventRecord(
        event_id=event_id,
        engine_id=engine_id,
        cycle=cycle,
        event_timestamp=datetime(2026, 9, 16, 12, cycle, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 9, 16, 13, cycle, tzinfo=UTC),
        schema_version="telemetry-v1",
        source_id="prediction-integration-test",
        payload_digest=f"digest-{event_id}",
    )


def _prediction(
    event_id: str,
    *,
    engine_id: int,
    cycle: int,
    estimated_rul: float | None,
    status: PredictionStatus,
    quality_status: DataQualityStatus,
) -> PredictionRecord:
    return PredictionRecord(
        event_id=event_id,
        engine_id=engine_id,
        cycle=cycle,
        estimated_rul=estimated_rul,
        status=status,
        data_quality_status=quality_status,
        quality_flags=["missing_sensor_7"] if status is PredictionStatus.WITHHELD else [],
        model_release="rul-model-v1",
        feature_version="features-v1",
        generated_at=datetime(2026, 9, 16, 14, cycle, tzinfo=UTC),
    )


def test_prediction_is_idempotent_and_loads_last_valid(database_engine) -> None:
    engine_id = 80_000 + uuid4().int % 10_000
    available_event_id = f"prediction-available-{uuid4()}"
    withheld_event_id = f"prediction-withheld-{uuid4()}"

    persist_telemetry_event(
        database_engine,
        _telemetry(available_event_id, engine_id=engine_id, cycle=1),
    )
    persist_telemetry_event(
        database_engine,
        _telemetry(withheld_event_id, engine_id=engine_id, cycle=2),
    )
    available = _prediction(
        available_event_id,
        engine_id=engine_id,
        cycle=1,
        estimated_rul=26.0,
        status=PredictionStatus.AVAILABLE,
        quality_status=DataQualityStatus.VALID,
    )
    withheld = _prediction(
        withheld_event_id,
        engine_id=engine_id,
        cycle=2,
        estimated_rul=None,
        status=PredictionStatus.WITHHELD,
        quality_status=DataQualityStatus.INVALID,
    )

    try:
        assert (
            persist_prediction(database_engine, available) is PredictionPersistenceOutcome.INSERTED
        )
        assert (
            persist_prediction(database_engine, available) is PredictionPersistenceOutcome.DUPLICATE
        )
        changed = available.model_copy(update={"estimated_rul": 12.0})
        assert persist_prediction(database_engine, changed) is PredictionPersistenceOutcome.CONFLICT
        assert (
            persist_prediction(database_engine, withheld) is PredictionPersistenceOutcome.INSERTED
        )

        last_valid = load_last_valid_prediction(
            database_engine,
            engine_id=engine_id,
            model_release="rul-model-v1",
        )

        assert last_valid is not None
        assert last_valid.event_id == available_event_id
        assert last_valid.cycle == 1
        assert last_valid.estimated_rul == 26.0
    finally:
        with database_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM prediction WHERE event_id IN (:available, :withheld)"),
                {"available": available_event_id, "withheld": withheld_event_id},
            )
            connection.execute(
                text("DELETE FROM telemetry_event WHERE event_id IN (:available, :withheld)"),
                {"available": available_event_id, "withheld": withheld_event_id},
            )
