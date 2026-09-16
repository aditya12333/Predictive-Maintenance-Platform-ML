"""Scikit-learn runtime adapter for approved RUL model artifacts."""

import math
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from predictive_maintenance.features.contracts import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FeatureVector,
)
from predictive_maintenance.inference.artifacts import ArtifactLoadError, load_approved_manifest


class _Transformer(Protocol):
    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _Regressor(Protocol):
    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...


class SklearnRULPredictor:
    """Run an approved fitted scaler and regression model without refitting either."""

    def __init__(self, *, scaler: _Transformer, model: _Regressor) -> None:
        self._scaler = scaler
        self._model = model

    @classmethod
    def load(cls, manifest_path: Path) -> "SklearnRULPredictor":
        """Load a version-compatible, approved scikit-learn model package."""

        manifest = load_approved_manifest(manifest_path)
        if manifest.feature_version != FEATURE_VERSION:
            raise ArtifactLoadError(
                f"model requires {manifest.feature_version}, runtime provides {FEATURE_VERSION}"
            )
        try:
            scaler = cast(_Transformer, joblib.load(manifest.scaler_path))
            model = cast(_Regressor, joblib.load(manifest.model_path))
        except Exception as error:
            raise ArtifactLoadError("could not deserialize model artifacts") from error
        return cls(scaler=scaler, model=model)

    def predict(self, features: FeatureVector) -> float:
        """Return a finite, non-negative RUL estimate in operating cycles."""

        if features.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"expected feature version {FEATURE_VERSION}, received {features.feature_version}"
            )
        if features.feature_names != FEATURE_NAMES:
            raise ValueError("feature names or ordering do not match features-v1")

        values = np.asarray([features.values], dtype=np.float64)
        prediction = float(self._model.predict(self._scaler.transform(values))[0])
        if not math.isfinite(prediction):
            raise ValueError("model returned a non-finite RUL prediction")
        return max(0.0, prediction)
