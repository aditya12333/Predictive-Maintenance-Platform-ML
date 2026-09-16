"""Atomic orchestration from an ordered telemetry event to a stored prediction."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine

from predictive_maintenance.inference.artifacts import load_approved_manifest
from predictive_maintenance.inference.contracts import DataQualityStatus, InferenceInput
from predictive_maintenance.inference.sklearn_predictor import SklearnRULPredictor
from predictive_maintenance.inference.worker import InferenceWorker
from predictive_maintenance.storage.database import (
    EventPersistenceOutcome,
    TelemetryEventRecord,
    persist_telemetry_event_transaction,
)
from predictive_maintenance.storage.predictions import (
    PredictionPersistenceError,
    PredictionPersistenceOutcome,
    persist_prediction_transaction,
)


@dataclass(frozen=True)
class EventProcessingResult:
    """Persistence outcomes produced for one ordered telemetry event."""

    telemetry_outcome: EventPersistenceOutcome
    prediction_outcome: PredictionPersistenceOutcome | None


class TransactionalInferenceProcessor:
    """Persist telemetry and its inference result in one database transaction."""

    def __init__(self, *, engine: Engine, worker: InferenceWorker) -> None:
        self._engine = engine
        self._worker = worker

    @classmethod
    def load_approved(
        cls,
        *,
        engine: Engine,
        manifest_path: Path,
    ) -> "TransactionalInferenceProcessor":
        """Build the processor from an explicitly approved model release."""

        manifest = load_approved_manifest(manifest_path)
        predictor = SklearnRULPredictor.load(manifest_path)
        return cls(
            engine=engine,
            worker=InferenceWorker(
                predictor=predictor,
                model_release=manifest.model_release,
            ),
        )

    def process(
        self,
        event: TelemetryEventRecord,
        *,
        quality_flags: tuple[str, ...] = (),
    ) -> EventProcessingResult:
        """Process one ordered event atomically, leaving failures available for retry."""

        if event.payload is None:
            raise ValueError("ordered telemetry event must include its validated payload")

        with self._engine.begin() as connection:
            telemetry_result = persist_telemetry_event_transaction(
                connection,
                event,
                quality_flags=quality_flags,
            )
            if telemetry_result.outcome is EventPersistenceOutcome.CONFLICT:
                return EventProcessingResult(
                    telemetry_outcome=telemetry_result.outcome,
                    prediction_outcome=None,
                )

            inference_input = InferenceInput.model_validate(
                {
                    "event_id": event.event_id,
                    "engine_id": event.engine_id,
                    "cycle": event.cycle,
                    "schema_version": event.schema_version,
                    "measurements": event.payload.get("measurements"),
                    "data_quality_status": (
                        DataQualityStatus.DEGRADED
                        if telemetry_result.is_degraded
                        else DataQualityStatus.VALID
                    ),
                    "quality_flags": telemetry_result.quality_flags,
                }
            )
            prediction = self._worker.process(inference_input)
            prediction_outcome = persist_prediction_transaction(connection, prediction)
            if prediction_outcome is PredictionPersistenceOutcome.CONFLICT:
                raise PredictionPersistenceError(
                    f"stored prediction conflicts with event {event.event_id}"
                )
            return EventProcessingResult(
                telemetry_outcome=telemetry_result.outcome,
                prediction_outcome=prediction_outcome,
            )
