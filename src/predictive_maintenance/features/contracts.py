"""Feature vector contracts."""

import math
from typing import Final

from pydantic import BaseModel, Field, model_validator

FEATURE_VERSION: Final = "features-v1"
SENSOR_FEATURE_NAMES: Final = tuple(
    f"sensor_{index}" for index in range(1, 22)
)
FEATURE_NAMES: Final = ("cycle", *SENSOR_FEATURE_NAMES)


class FeatureVector(BaseModel):
    """An ordered numeric feature vector passed to a model."""

    feature_version: str = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    values: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape_and_values(self) -> "FeatureVector":
        if len(self.feature_names) != len(self.values):
            raise ValueError("feature_names and values must have the same length")

        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("feature values must be finite")

        return self
