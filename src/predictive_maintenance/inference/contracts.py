"""Prediction output contracts."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataQualityStatus(StrEnum):
    """Resolved input-quality state used by the inference decision."""

    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"


class InferenceInput(BaseModel):
    """Validated event data and quality context required for one prediction."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    engine_id: int = Field(gt=0)
    cycle: int = Field(gt=0)
    schema_version: str = Field(min_length=1)
    measurements: dict[str, float | None] = Field(min_length=1)
    data_quality_status: DataQualityStatus
    quality_flags: tuple[str, ...] = Field(default_factory=tuple)


class PredictionStatus(StrEnum):
    """Status of a prediction."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    WITHHELD = "withheld"


class PredictionRecord(BaseModel):
    """A prediction record."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    engine_id: int = Field(gt=0)
    cycle: int = Field(gt=0)
    estimated_rul: float | None = Field(default=None, ge=0)
    status: PredictionStatus
    data_quality_status: DataQualityStatus
    quality_flags: list[str] = Field(default_factory=list)
    model_release: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    generated_at: datetime

    @model_validator(mode="after")
    def validate_prediction_availability(self) -> "PredictionRecord":
        """Keep prediction status consistent with the presence of an RUL value."""

        if self.status is PredictionStatus.WITHHELD and self.estimated_rul is not None:
            raise ValueError("withheld predictions must not contain estimated_rul")
        if self.status is not PredictionStatus.WITHHELD and self.estimated_rul is None:
            raise ValueError("available or degraded predictions require estimated_rul")
        return self
