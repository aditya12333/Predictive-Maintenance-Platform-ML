"""Model-independent prediction interfaces."""

from typing import Protocol

from predictive_maintenance.features.contracts import FeatureVector


class RULPredictor(Protocol):
    """Interface required by the inference worker to generate an RUL estimate."""

    def predict(self, features: FeatureVector) -> float:
        """Return estimated remaining useful life in operating cycles."""
