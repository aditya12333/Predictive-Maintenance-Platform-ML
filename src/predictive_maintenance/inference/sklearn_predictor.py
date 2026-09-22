"""Runtime adapter for approved joblib-compatible RUL regressors."""

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
from predictive_maintenance.inference.artifacts import (
    ArtifactLoadError,
    ModelArtifactManifest,
    load_approved_manifest,
)


class _Transformer(Protocol):
    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _Regressor(Protocol):
    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...


class SklearnRULPredictor:
    """Run an approved fitted regressor with optional fitted preprocessing."""

    def __init__(
        self,
        *,
        model: _Regressor,
        preprocessor: _Transformer | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._model = model

    @classmethod
    def load(cls, manifest_path: Path) -> "SklearnRULPredictor":
        """Load an integrity-checked, runtime-compatible approved model package."""

        manifest = load_approved_manifest(manifest_path)
        return cls.from_verified_manifest(manifest)

    @classmethod
    def from_verified_manifest(
        cls,
        manifest: ModelArtifactManifest,
    ) -> "SklearnRULPredictor":
        """Load artifacts from a manifest already verified by a trusted lifecycle source."""

        if manifest.feature_version != FEATURE_VERSION:
            raise ArtifactLoadError(
                f"model requires {manifest.feature_version}, runtime provides {FEATURE_VERSION}"
            )
        if manifest.feature_names != FEATURE_NAMES:
            raise ArtifactLoadError("manifest feature names or ordering do not match features-v1")

        try:
            model = cast(_Regressor, joblib.load(manifest.model_path))
            preprocessor = (
                cast(_Transformer, joblib.load(manifest.preprocessing_path))
                if manifest.preprocessing_path is not None
                else None
            )
        except Exception as error:
            raise ArtifactLoadError("could not deserialize model artifacts") from error
        return cls(model=model, preprocessor=preprocessor)

    def predict(self, features: FeatureVector) -> float:
        """Return a finite, non-negative RUL estimate in operating cycles."""

        if features.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"expected feature version {FEATURE_VERSION}, received {features.feature_version}"
            )
        if features.feature_names != FEATURE_NAMES:
            raise ValueError("feature names or ordering do not match features-v1")

        values = np.asarray([features.values], dtype=np.float64)
        model_values = (
            self._preprocessor.transform(values)
            if self._preprocessor is not None
            else values
        )
        prediction = float(self._model.predict(model_values)[0])
        if not math.isfinite(prediction):
            raise ValueError("model returned a non-finite RUL prediction")
        return max(0.0, prediction)
