"""Idempotent prediction persistence and last-valid prediction lookup."""

from collections.abc import Mapping
from enum import StrEnum

from psycopg.types.json import Jsonb
from pydantic import ValidationError
from sqlalchemy import Connection, Engine, text
from sqlalchemy.engine import RowMapping

from predictive_maintenance.inference.contracts import PredictionRecord, PredictionStatus


class PredictionPersistenceOutcome(StrEnum):
    """Result of inserting one prediction under its business identity."""

    INSERTED = "inserted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


class PredictionPersistenceError(ValueError):
    """Raised when stored prediction data violates the application contract."""


def persist_prediction(
    engine: Engine,
    prediction: PredictionRecord,
) -> PredictionPersistenceOutcome:
    """Insert a prediction idempotently while preserving the original on conflict."""

    with engine.begin() as connection:
        return persist_prediction_transaction(connection, prediction)


def persist_prediction_transaction(
    connection: Connection,
    prediction: PredictionRecord,
) -> PredictionPersistenceOutcome:
    """Persist a prediction inside a caller-managed database transaction."""

    insert = text(
        """
        INSERT INTO prediction (
            event_id,
            model_release,
            generated_at,
            rul_cycles,
            status,
            withheld_reason,
            feature_version,
            data_quality_status,
            quality_flags
        ) VALUES (
            :event_id,
            :model_release,
            :generated_at,
            :rul_cycles,
            :status,
            :withheld_reason,
            :feature_version,
            :data_quality_status,
            :quality_flags
        )
        ON CONFLICT (event_id, model_release) DO NOTHING
        RETURNING prediction_id
        """
    )
    lookup = text(
        """
        SELECT
            rul_cycles,
            status,
            feature_version,
            data_quality_status,
            quality_flags
        FROM prediction
        WHERE event_id = :event_id
          AND model_release = :model_release
        """
    )
    parameters = _persistence_parameters(prediction)

    inserted_id = connection.execute(insert, parameters).scalar_one_or_none()
    if inserted_id is not None:
        return PredictionPersistenceOutcome.INSERTED

    existing = (
        connection.execute(
            lookup,
            {
                "event_id": prediction.event_id,
                "model_release": prediction.model_release,
            },
        )
        .mappings()
        .one_or_none()
    )
    if existing is None:
        raise PredictionPersistenceError(
            "prediction conflict was reported but the stored row could not be loaded"
        )
    if _matches_existing(existing, prediction):
        return PredictionPersistenceOutcome.DUPLICATE
    return PredictionPersistenceOutcome.CONFLICT


def load_last_valid_prediction(
    engine: Engine,
    *,
    engine_id: int,
    model_release: str,
) -> PredictionRecord | None:
    """Load the latest usable prediction for one engine and model release."""

    if engine_id <= 0:
        raise ValueError("engine_id must be positive")
    if not model_release.strip():
        raise ValueError("model_release must not be empty")

    statement = text(
        """
        SELECT
            p.event_id,
            t.engine_id,
            t.cycle,
            p.rul_cycles AS estimated_rul,
            p.status,
            p.data_quality_status,
            p.quality_flags,
            p.model_release,
            p.feature_version,
            p.generated_at
        FROM prediction AS p
        JOIN telemetry_event AS t ON t.event_id = p.event_id
        WHERE t.engine_id = :engine_id
          AND p.model_release = :model_release
          AND p.status IN ('available', 'degraded')
          AND p.rul_cycles IS NOT NULL
        ORDER BY t.cycle DESC, p.generated_at DESC, p.prediction_id DESC
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        row = (
            connection.execute(
                statement,
                {"engine_id": engine_id, "model_release": model_release},
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        return None

    try:
        return PredictionRecord.model_validate(dict(row))
    except ValidationError as error:
        raise PredictionPersistenceError(
            "stored last-valid prediction failed contract validation"
        ) from error


def _persistence_parameters(prediction: PredictionRecord) -> dict[str, object]:
    withheld_reason = (
        "; ".join(prediction.quality_flags)
        if prediction.status is PredictionStatus.WITHHELD and prediction.quality_flags
        else None
    )
    return {
        "event_id": prediction.event_id,
        "model_release": prediction.model_release,
        "generated_at": prediction.generated_at,
        "rul_cycles": prediction.estimated_rul,
        "status": prediction.status.value,
        "withheld_reason": withheld_reason,
        "feature_version": prediction.feature_version,
        "data_quality_status": prediction.data_quality_status.value,
        "quality_flags": Jsonb(prediction.quality_flags),
    }


def _matches_existing(
    existing: Mapping[str, object] | RowMapping,
    prediction: PredictionRecord,
) -> bool:
    quality_flags = existing.get("quality_flags")
    return (
        existing.get("rul_cycles") == prediction.estimated_rul
        and existing.get("status") == prediction.status.value
        and existing.get("feature_version") == prediction.feature_version
        and existing.get("data_quality_status") == prediction.data_quality_status.value
        and isinstance(quality_flags, list)
        and quality_flags == prediction.quality_flags
    )
