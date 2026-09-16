"""Deterministic construction of model feature vectors."""

import math
from collections.abc import Mapping

from predictive_maintenance.features.contracts import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FeatureVector,
)


class FeatureBuildError(ValueError):
    """Raised when telemetry cannot produce a valid feature vector."""


class FeatureBuilder:
    """Build the versioned baseline feature vector from telemetry measurements."""

    def build(
        self,
        *,
        cycle: int,
        measurements: Mapping[str, float | None],
    ) -> FeatureVector:
        """Build cycle-plus-sensor features in the canonical order."""

        if cycle <= 0:
            raise FeatureBuildError("cycle must be positive")

        values: list[float] = []
        for feature_name in FEATURE_NAMES:
            if feature_name == "cycle":
                values.append(float(cycle))
                continue

            if feature_name not in measurements:
                raise FeatureBuildError(f"required feature {feature_name} is missing")

            value = measurements[feature_name]
            if value is None:
                raise FeatureBuildError(f"required feature {feature_name} is null")
            if not math.isfinite(value):
                raise FeatureBuildError(
                    f"required feature {feature_name} must be finite"
                )
            values.append(float(value))

        return FeatureVector(
            feature_version=FEATURE_VERSION,
            feature_names=FEATURE_NAMES,
            values=tuple(values),
        )
