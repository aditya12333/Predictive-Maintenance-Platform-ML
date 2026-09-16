"""Tests for deterministic feature construction."""

import math

import pytest

from predictive_maintenance.features.builder import FeatureBuilder, FeatureBuildError
from predictive_maintenance.features.contracts import FEATURE_NAMES, FEATURE_VERSION


def complete_measurements() -> dict[str, float]:
    return {name: float(index) for index, name in enumerate(FEATURE_NAMES[1:], start=1)}


def test_builder_uses_canonical_feature_order() -> None:
    measurements = complete_measurements()
    measurements = dict(reversed(list(measurements.items())))

    vector = FeatureBuilder().build(cycle=82, measurements=measurements)

    assert vector.feature_version == FEATURE_VERSION
    assert vector.feature_names == FEATURE_NAMES
    assert vector.values[0] == 82.0
    assert vector.values[1:] == tuple(measurements[name] for name in FEATURE_NAMES[1:])


def test_builder_rejects_missing_sensor() -> None:
    measurements = complete_measurements()
    del measurements["sensor_7"]

    with pytest.raises(FeatureBuildError, match="sensor_7 is missing"):
        FeatureBuilder().build(cycle=82, measurements=measurements)


def test_builder_rejects_null_sensor() -> None:
    measurements = complete_measurements()
    measurements["sensor_7"] = None

    with pytest.raises(FeatureBuildError, match="sensor_7 is null"):
        FeatureBuilder().build(cycle=82, measurements=measurements)


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_builder_rejects_non_finite_sensor(invalid_value: float) -> None:
    measurements = complete_measurements()
    measurements["sensor_7"] = invalid_value

    with pytest.raises(FeatureBuildError, match="sensor_7 must be finite"):
        FeatureBuilder().build(cycle=82, measurements=measurements)


def test_builder_rejects_non_positive_cycle() -> None:
    with pytest.raises(FeatureBuildError, match="cycle must be positive"):
        FeatureBuilder().build(cycle=0, measurements=complete_measurements())
