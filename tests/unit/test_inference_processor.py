"""Tests for atomic ordered-event inference orchestration."""

from datetime import UTC, datetime
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy import Engine

import predictive_maintenance.inference.processor as processor_module
from predictive_maintenance.inference.contracts import (
    DataQualityStatus,
    PredictionRecord,
    PredictionStatus,
)
from predictive_maintenance.inference.processor import TransactionalInferenceProcessor
from predictive_maintenance.inference.worker import InferenceWorker
from predictive_maintenance.storage.database import (
    EventPersistenceOutcome,
    TelemetryEventRecord,
    TelemetryPersistenceResult,
)
from predictive_maintenance.storage.predictions import (
    PredictionPersistenceError,
    PredictionPersistenceOutcome,
)


class TransactionContext:
    def __init__(self, connection: object) -> None:
        self.connection = connection
        self.exception_type: type[BaseException] | None = None

    def __enter__(self) -> object:
        return self.connection

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        del exception, traceback
        self.exception_type = exception_type


class FakeEngine:
    def __init__(self) -> None:
        self.connection = object()
        self.transaction = TransactionContext(self.connection)

    def begin(self) -> TransactionContext:
        return self.transaction


def event() -> TelemetryEventRecord:
    return TelemetryEventRecord(
        event_id="engine-4-cycle-9",
        engine_id=4,
        cycle=9,
        event_timestamp=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 9, 16, 12, 0, 1, tzinfo=UTC),
        schema_version="telemetry-v1",
        source_id="unit-test",
        payload_digest="digest",
        payload={"measurements": {f"sensor_{index}": 1.0 for index in range(1, 22)}},
    )


def prediction() -> PredictionRecord:
    return PredictionRecord(
        event_id="engine-4-cycle-9",
        engine_id=4,
        cycle=9,
        estimated_rul=20.0,
        status=PredictionStatus.AVAILABLE,
        data_quality_status=DataQualityStatus.VALID,
        quality_flags=[],
        model_release="rul-model-v1",
        feature_version="features-v1",
        generated_at=datetime(2026, 9, 16, 12, 0, 2, tzinfo=UTC),
    )


def build_processor() -> tuple[TransactionalInferenceProcessor, FakeEngine, Mock]:
    engine = FakeEngine()
    worker = Mock(spec=InferenceWorker)
    worker.process.return_value = prediction()
    processor = TransactionalInferenceProcessor(
        engine=cast(Engine, engine),
        worker=worker,
    )
    return processor, engine, worker


def test_process_persists_telemetry_and_prediction_in_same_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, engine, worker = build_processor()
    telemetry_persist = Mock(
        return_value=TelemetryPersistenceResult(
            outcome=EventPersistenceOutcome.INSERTED,
            is_degraded=False,
            quality_flags=(),
        )
    )
    prediction_persist = Mock(return_value=PredictionPersistenceOutcome.INSERTED)
    monkeypatch.setattr(
        processor_module,
        "persist_telemetry_event_transaction",
        telemetry_persist,
    )
    monkeypatch.setattr(
        processor_module,
        "persist_prediction_transaction",
        prediction_persist,
    )

    result = processor.process(event())

    assert result.telemetry_outcome is EventPersistenceOutcome.INSERTED
    assert result.prediction_outcome is PredictionPersistenceOutcome.INSERTED
    telemetry_persist.assert_called_once_with(engine.connection, event(), quality_flags=())
    prediction_persist.assert_called_once_with(engine.connection, prediction())
    inference_input = worker.process.call_args.args[0]
    assert inference_input.data_quality_status is DataQualityStatus.VALID
    assert engine.transaction.exception_type is None


def test_process_propagates_degraded_quality_to_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, worker = build_processor()
    monkeypatch.setattr(
        processor_module,
        "persist_telemetry_event_transaction",
        Mock(
            return_value=TelemetryPersistenceResult(
                outcome=EventPersistenceOutcome.INSERTED,
                is_degraded=True,
                quality_flags=("cycle_gap",),
            )
        ),
    )
    monkeypatch.setattr(
        processor_module,
        "persist_prediction_transaction",
        Mock(return_value=PredictionPersistenceOutcome.INSERTED),
    )

    processor.process(event())

    inference_input = worker.process.call_args.args[0]
    assert inference_input.data_quality_status is DataQualityStatus.DEGRADED
    assert inference_input.quality_flags == ("cycle_gap",)


def test_conflicting_telemetry_does_not_score_the_changed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _, worker = build_processor()
    prediction_persist = Mock()
    monkeypatch.setattr(
        processor_module,
        "persist_telemetry_event_transaction",
        Mock(
            return_value=TelemetryPersistenceResult(
                outcome=EventPersistenceOutcome.CONFLICT,
                is_degraded=True,
                quality_flags=("conflicting_duplicate",),
            )
        ),
    )
    monkeypatch.setattr(
        processor_module,
        "persist_prediction_transaction",
        prediction_persist,
    )

    result = processor.process(event())

    assert result.prediction_outcome is None
    worker.process.assert_not_called()
    prediction_persist.assert_not_called()


def test_prediction_conflict_fails_the_transaction_for_investigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, engine, _ = build_processor()
    monkeypatch.setattr(
        processor_module,
        "persist_telemetry_event_transaction",
        Mock(
            return_value=TelemetryPersistenceResult(
                outcome=EventPersistenceOutcome.DUPLICATE,
                is_degraded=False,
                quality_flags=(),
            )
        ),
    )
    monkeypatch.setattr(
        processor_module,
        "persist_prediction_transaction",
        Mock(return_value=PredictionPersistenceOutcome.CONFLICT),
    )

    with pytest.raises(PredictionPersistenceError, match="engine-4-cycle-9"):
        processor.process(event())

    assert engine.transaction.exception_type is PredictionPersistenceError


def test_process_requires_the_validated_payload() -> None:
    processor, _, _ = build_processor()
    missing_payload = TelemetryEventRecord(**{**event().__dict__, "payload": None})

    with pytest.raises(ValueError, match="validated payload"):
        processor.process(missing_payload)
