"""Tests for model-independent inference decisions."""

import math
from datetime import UTC, datetime

import pytest

from predictive_maintenance.features.contracts import FEATURE_NAMES, FeatureVector
from predictive_maintenance.inference.contracts import (
    DataQualityStatus,
    InferenceInput,
    PredictionStatus,
)
from predictive_maintenance.inference.worker import InferenceExecutionError, InferenceWorker

GENERATED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


class FakePredictor:
    def __init__(self, prediction: float = 26.0, error: Exception | None = None) -> None:
        self.prediction = prediction
        self.error = error
        self.received_features: list[FeatureVector] = []

    def predict(self, features: FeatureVector) -> float:
        self.received_features.append(features)
        if self.error is not None:
            raise self.error
        return self.prediction


def complete_measurements() -> dict[str, float]:
    return {name: float(index) for index, name in enumerate(FEATURE_NAMES[1:], start=1)}


def inference_input(**updates: object) -> InferenceInput:
    payload: dict[str, object] = {
        "event_id": "engine-17-cycle-82",
        "engine_id": 17,
        "cycle": 82,
        "schema_version": "telemetry-v1",
        "measurements": complete_measurements(),
        "data_quality_status": DataQualityStatus.VALID,
        "quality_flags": (),
    }
    payload.update(updates)
    return InferenceInput(**payload)


def worker(predictor: FakePredictor) -> InferenceWorker:
    return InferenceWorker(
        predictor=predictor,
        model_release="rul-model-v1",
        clock=lambda: GENERATED_AT,
    )


def test_valid_input_produces_available_prediction() -> None:
    predictor = FakePredictor(prediction=26.0)

    prediction = worker(predictor).process(inference_input())

    assert prediction.status is PredictionStatus.AVAILABLE
    assert prediction.estimated_rul == 26.0
    assert prediction.data_quality_status is DataQualityStatus.VALID
    assert prediction.quality_flags == []
    assert prediction.model_release == "rul-model-v1"
    assert prediction.generated_at == GENERATED_AT
    assert predictor.received_features[0].feature_names == FEATURE_NAMES
    assert predictor.received_features[0].values[0] == 82.0


def test_degraded_input_produces_degraded_prediction() -> None:
    predictor = FakePredictor(prediction=18.0)
    event = inference_input(
        data_quality_status=DataQualityStatus.DEGRADED,
        quality_flags=("cycle_gap",),
    )

    prediction = worker(predictor).process(event)

    assert prediction.status is PredictionStatus.DEGRADED
    assert prediction.estimated_rul == 18.0
    assert prediction.data_quality_status is DataQualityStatus.DEGRADED
    assert prediction.quality_flags == ["cycle_gap"]
    assert len(predictor.received_features) == 1


def test_invalid_input_is_withheld_without_calling_model() -> None:
    predictor = FakePredictor()
    event = inference_input(data_quality_status=DataQualityStatus.INVALID)

    prediction = worker(predictor).process(event)

    assert prediction.status is PredictionStatus.WITHHELD
    assert prediction.estimated_rul is None
    assert prediction.data_quality_status is DataQualityStatus.INVALID
    assert prediction.quality_flags == ["invalid_data_quality"]
    assert predictor.received_features == []


def test_incompatible_schema_is_withheld_without_calling_model() -> None:
    predictor = FakePredictor()

    prediction = worker(predictor).process(inference_input(schema_version="telemetry-v0"))

    assert prediction.status is PredictionStatus.WITHHELD
    assert prediction.data_quality_status is DataQualityStatus.INVALID
    assert prediction.quality_flags == ["incompatible_schema:telemetry-v0"]
    assert predictor.received_features == []


@pytest.mark.parametrize(
    ("measurements", "reason"),
    [
        (
            {name: value for name, value in complete_measurements().items() if name != "sensor_7"},
            "sensor_7 is missing",
        ),
        ({**complete_measurements(), "sensor_7": None}, "sensor_7 is null"),
        ({**complete_measurements(), "sensor_7": math.nan}, "sensor_7 must be finite"),
    ],
)
def test_invalid_features_are_withheld(
    measurements: dict[str, float | None],
    reason: str,
) -> None:
    predictor = FakePredictor()

    prediction = worker(predictor).process(inference_input(measurements=measurements))

    assert prediction.status is PredictionStatus.WITHHELD
    assert prediction.estimated_rul is None
    assert prediction.data_quality_status is DataQualityStatus.INVALID
    assert prediction.quality_flags == [f"feature_build_error:required feature {reason}"]
    assert predictor.received_features == []


def test_predictor_failure_is_raised_for_retry() -> None:
    predictor = FakePredictor(error=RuntimeError("model unavailable"))

    with pytest.raises(InferenceExecutionError, match="could not score") as raised:
        worker(predictor).process(inference_input())

    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.parametrize("invalid_prediction", [-1.0, math.nan, math.inf])
def test_invalid_model_output_is_raised_for_retry(invalid_prediction: float) -> None:
    predictor = FakePredictor(prediction=invalid_prediction)

    with pytest.raises(InferenceExecutionError, match="returned an invalid RUL"):
        worker(predictor).process(inference_input())


def test_worker_rejects_empty_model_release() -> None:
    with pytest.raises(ValueError, match="model_release must not be empty"):
        InferenceWorker(predictor=FakePredictor(), model_release=" ")
