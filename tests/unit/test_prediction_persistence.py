"""Unit tests for prediction persistence helpers."""

from datetime import UTC, datetime

import pytest

from predictive_maintenance.inference.contracts import (
    DataQualityStatus,
    PredictionRecord,
    PredictionStatus,
)
from predictive_maintenance.storage.predictions import _matches_existing, _persistence_parameters


def prediction() -> PredictionRecord:
    return PredictionRecord(
        event_id="engine-17-cycle-82",
        engine_id=17,
        cycle=82,
        estimated_rul=26.0,
        status=PredictionStatus.AVAILABLE,
        data_quality_status=DataQualityStatus.VALID,
        quality_flags=[],
        model_release="rul-model-v1",
        feature_version="features-v1",
        generated_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
    )


def test_persistence_parameters_preserve_prediction_lineage() -> None:
    parameters = _persistence_parameters(prediction())

    assert parameters["event_id"] == "engine-17-cycle-82"
    assert parameters["model_release"] == "rul-model-v1"
    assert parameters["rul_cycles"] == 26.0
    assert parameters["status"] == "available"
    assert parameters["feature_version"] == "features-v1"
    assert parameters["data_quality_status"] == "valid"
    assert parameters["withheld_reason"] is None


def test_withheld_flags_are_stored_as_readable_reason() -> None:
    withheld = prediction().model_copy(
        update={
            "estimated_rul": None,
            "status": PredictionStatus.WITHHELD,
            "data_quality_status": DataQualityStatus.INVALID,
            "quality_flags": ["missing_sensor_7", "invalid_data_quality"],
        }
    )

    parameters = _persistence_parameters(withheld)

    assert parameters["withheld_reason"] == "missing_sensor_7; invalid_data_quality"


def test_existing_prediction_matches_business_content() -> None:
    stored = {
        "rul_cycles": 26.0,
        "status": "available",
        "feature_version": "features-v1",
        "data_quality_status": "valid",
        "quality_flags": [],
    }

    assert _matches_existing(stored, prediction())


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("rul_cycles", 12.0),
        ("status", "degraded"),
        ("feature_version", "features-v2"),
        ("data_quality_status", "degraded"),
        ("quality_flags", ["cycle_gap"]),
    ],
)
def test_changed_business_content_is_a_conflict(field: str, different_value: object) -> None:
    stored = {
        "rul_cycles": 26.0,
        "status": "available",
        "feature_version": "features-v1",
        "data_quality_status": "valid",
        "quality_flags": [],
    }
    stored[field] = different_value

    assert not _matches_existing(stored, prediction())
