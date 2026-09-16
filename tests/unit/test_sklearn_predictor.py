"""Tests for the approved scikit-learn RUL runtime adapter."""

import numpy as np
import pytest

from predictive_maintenance.features.contracts import FEATURE_NAMES, FeatureVector
from predictive_maintenance.inference.sklearn_predictor import SklearnRULPredictor


class IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class ConstantModel:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(values.shape[0], self.prediction)


def valid_features() -> FeatureVector:
    return FeatureVector(
        feature_version="features-v1",
        feature_names=FEATURE_NAMES,
        values=tuple(float(index) for index in range(len(FEATURE_NAMES))),
    )


def test_predictor_returns_non_negative_rul() -> None:
    predictor = SklearnRULPredictor(scaler=IdentityScaler(), model=ConstantModel(-3.0))

    assert predictor.predict(valid_features()) == 0.0


def test_predictor_rejects_wrong_feature_order() -> None:
    features = valid_features().model_copy(
        update={"feature_names": tuple(reversed(FEATURE_NAMES))}
    )
    predictor = SklearnRULPredictor(scaler=IdentityScaler(), model=ConstantModel(10.0))

    with pytest.raises(ValueError, match="ordering"):
        predictor.predict(features)
