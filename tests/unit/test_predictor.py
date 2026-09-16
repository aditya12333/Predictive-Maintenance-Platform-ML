"""Tests for the model-independent predictor interface."""

from predictive_maintenance.features.contracts import FeatureVector
from predictive_maintenance.inference.predictor import RULPredictor


class FakeRULModel:
    def predict(self, features: FeatureVector) -> float:
        assert features.feature_version == "features-v1"
        return 26.0


def test_fake_model_implements_predictor_interface() -> None:
    predictor: RULPredictor = FakeRULModel()
    features = FeatureVector(
        feature_version="features-v1",
        feature_names=("cycle",),
        values=(82.0,),
    )

    assert predictor.predict(features) == 26.0
