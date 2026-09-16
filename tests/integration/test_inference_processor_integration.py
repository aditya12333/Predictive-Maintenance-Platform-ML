"""PostgreSQL integration tests for atomic telemetry-to-prediction processing."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text

from predictive_maintenance.core.settings import load_settings
from predictive_maintenance.features.contracts import FeatureVector
from predictive_maintenance.inference.processor import TransactionalInferenceProcessor
from predictive_maintenance.inference.worker import InferenceExecutionError, InferenceWorker
from predictive_maintenance.storage.database import (
    EventPersistenceOutcome,
    TelemetryEventRecord,
    create_database_engine,
)
from predictive_maintenance.storage.predictions import PredictionPersistenceOutcome


class ConstantPredictor:
    def predict(self, features: FeatureVector) -> float:
        del features
        return 18.5


class FailingPredictor:
    def predict(self, features: FeatureVector) -> float:
        del features
        raise RuntimeError("model service unavailable")


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


def event(event_id: str, *, engine_id: int) -> TelemetryEventRecord:
    return TelemetryEventRecord(
        event_id=event_id,
        engine_id=engine_id,
        cycle=1,
        event_timestamp=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 9, 16, 12, 0, 1, tzinfo=UTC),
        schema_version="telemetry-v1",
        source_id="processor-integration-test",
        payload_digest=f"digest-{event_id}",
        payload={"measurements": {f"sensor_{index}": 1.0 for index in range(1, 22)}},
    )


def delete_event(database_engine, event_id: str) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text("DELETE FROM prediction WHERE event_id = :event_id"),
            {"event_id": event_id},
        )
        connection.execute(
            text("DELETE FROM telemetry_event WHERE event_id = :event_id"),
            {"event_id": event_id},
        )


def test_processor_commits_telemetry_and_prediction_idempotently(database_engine) -> None:
    event_id = f"processor-success-{uuid4()}"
    engine_id = 90_000 + uuid4().int % 9_000
    processor = TransactionalInferenceProcessor(
        engine=database_engine,
        worker=InferenceWorker(
            predictor=ConstantPredictor(),
            model_release="integration-model-v1",
            clock=lambda: datetime(2026, 9, 16, 12, 0, 2, tzinfo=UTC),
        ),
    )
    telemetry = event(event_id, engine_id=engine_id)

    try:
        inserted = processor.process(telemetry)
        duplicate = processor.process(telemetry)

        assert inserted.telemetry_outcome is EventPersistenceOutcome.INSERTED
        assert inserted.prediction_outcome is PredictionPersistenceOutcome.INSERTED
        assert duplicate.telemetry_outcome is EventPersistenceOutcome.DUPLICATE
        assert duplicate.prediction_outcome is PredictionPersistenceOutcome.DUPLICATE
        with database_engine.connect() as connection:
            stored = connection.execute(
                text(
                    "SELECT t.quality_status, p.rul_cycles, p.status "
                    "FROM telemetry_event AS t "
                    "JOIN prediction AS p ON p.event_id = t.event_id "
                    "WHERE t.event_id = :event_id"
                ),
                {"event_id": event_id},
            ).one()
        assert stored.quality_status == "VALID"
        assert stored.rul_cycles == 18.5
        assert stored.status == "available"
    finally:
        delete_event(database_engine, event_id)


def test_model_failure_rolls_back_telemetry_insert(database_engine) -> None:
    event_id = f"processor-rollback-{uuid4()}"
    processor = TransactionalInferenceProcessor(
        engine=database_engine,
        worker=InferenceWorker(
            predictor=FailingPredictor(),
            model_release="integration-model-v1",
        ),
    )

    with pytest.raises(InferenceExecutionError):
        processor.process(event(event_id, engine_id=99_999))

    with database_engine.connect() as connection:
        stored_count = connection.execute(
            text("SELECT count(*) FROM telemetry_event WHERE event_id = :event_id"),
            {"event_id": event_id},
        ).scalar_one()
    assert stored_count == 0
