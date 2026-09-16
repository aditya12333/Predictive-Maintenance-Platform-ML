"""Tests for prediction output contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from predictive_maintenance.inference.contracts import (
    DataQualityStatus,
    InferenceInput,
    PredictionRecord,
    PredictionStatus,
)


def valid_prediction_payload() -> dict:
    return {
        "event_id": "engine-17-cycle-82",
        "engine_id": 17,
        "cycle": 82,
        "estimated_rul": 26.0,
        "status": PredictionStatus.AVAILABLE,
        "data_quality_status": DataQualityStatus.VALID,
        "quality_flags": [],
        "model_release": "rul-model-v1",
        "feature_version": "features-v1",
        "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }


def test_available_prediction_is_valid() -> None:
    prediction = PredictionRecord(**valid_prediction_payload())

    assert prediction.status == PredictionStatus.AVAILABLE
    assert prediction.estimated_rul == 26.0


def test_withheld_prediction_can_have_no_rul() -> None:
    payload = valid_prediction_payload()
    payload["estimated_rul"] = None
    payload["status"] = PredictionStatus.WITHHELD
    payload["data_quality_status"] = DataQualityStatus.INVALID

    prediction = PredictionRecord(**payload)

    assert prediction.status == PredictionStatus.WITHHELD
    assert prediction.estimated_rul is None


def test_degraded_prediction_requires_and_accepts_rul() -> None:
    payload = valid_prediction_payload()
    payload["status"] = PredictionStatus.DEGRADED
    payload["data_quality_status"] = DataQualityStatus.DEGRADED
    payload["quality_flags"] = ["cycle_gap"]

    prediction = PredictionRecord(**payload)

    assert prediction.status is PredictionStatus.DEGRADED
    assert prediction.estimated_rul == 26.0


@pytest.mark.parametrize(
    ("status", "estimated_rul", "message"),
    [
        (PredictionStatus.WITHHELD, 26.0, "must not contain estimated_rul"),
        (PredictionStatus.AVAILABLE, None, "require estimated_rul"),
        (PredictionStatus.DEGRADED, None, "require estimated_rul"),
    ],
)
def test_prediction_status_must_match_rul_presence(
    status: PredictionStatus,
    estimated_rul: float | None,
    message: str,
) -> None:
    payload = valid_prediction_payload()
    payload["status"] = status
    payload["estimated_rul"] = estimated_rul

    with pytest.raises(ValidationError, match=message):
        PredictionRecord(**payload)


def test_negative_rul_is_rejected() -> None:
    payload = valid_prediction_payload()
    payload["estimated_rul"] = -1.0

    with pytest.raises(ValidationError):
        PredictionRecord(**payload)


def test_invalid_engine_id_is_rejected() -> None:
    payload = valid_prediction_payload()
    payload["engine_id"] = 0

    with pytest.raises(ValidationError):
        PredictionRecord(**payload)


def test_inference_input_carries_measurements_and_quality_context() -> None:
    inference_input = InferenceInput(
        event_id="engine-17-cycle-82",
        engine_id=17,
        cycle=82,
        schema_version="telemetry-v1",
        measurements={"sensor_1": 1.5},
        data_quality_status=DataQualityStatus.DEGRADED,
        quality_flags=("cycle_gap",),
    )

    assert inference_input.measurements == {"sensor_1": 1.5}
    assert inference_input.data_quality_status is DataQualityStatus.DEGRADED
    assert inference_input.quality_flags == ("cycle_gap",)


def test_inference_input_preserves_incompatible_schema_for_worker_decision() -> None:
    inference_input = InferenceInput(
        event_id="engine-17-cycle-82",
        engine_id=17,
        cycle=82,
        schema_version="telemetry-v0",
        measurements={"sensor_1": 1.5},
        data_quality_status=DataQualityStatus.VALID,
    )

    assert inference_input.schema_version == "telemetry-v0"
