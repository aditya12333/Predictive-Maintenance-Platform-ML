import math

import pytest
from pydantic import ValidationError

from predictive_maintenance.features.contracts import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FeatureVector,
)


def test_features_v1_has_stable_order() -> None:
    assert FEATURE_VERSION == "features-v1"
    assert FEATURE_NAMES[0] == "cycle"
    assert FEATURE_NAMES[-1] == "sensor_21"
    assert len(FEATURE_NAMES) == 22


def test_feature_vector_accepts_matching_finite_values() -> None:
    vector = FeatureVector(
        feature_version="features-v1",
        feature_names=("cycle", "sensor_1"),
        values=(82.0, 0.52),
    )

    assert vector.feature_names == ("cycle", "sensor_1")
    assert vector.values == (82.0, 0.52)


def test_feature_vector_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError, match="same length"):
        FeatureVector(
            feature_version="features-v1",
            feature_names=("cycle", "sensor_1"),
            values=(82.0,),
        )


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_feature_vector_rejects_non_finite_values(invalid_value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        FeatureVector(
            feature_version="features-v1",
            feature_names=("cycle",),
            values=(invalid_value,),
        )
