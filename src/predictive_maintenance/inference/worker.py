"""Model-independent inference decisions for one validated telemetry event."""

import math
from collections.abc import Callable
from datetime import UTC, datetime

from predictive_maintenance.data.contracts import TELEMETRY_SCHEMA_VERSION
from predictive_maintenance.features.builder import FeatureBuilder, FeatureBuildError
from predictive_maintenance.features.contracts import FEATURE_VERSION
from predictive_maintenance.inference.contracts import (
    DataQualityStatus,
    InferenceInput,
    PredictionRecord,
    PredictionStatus,
)
from predictive_maintenance.inference.predictor import RULPredictor


class InferenceExecutionError(RuntimeError):
    """Raised when valid input cannot be scored because model execution failed."""


class InferenceWorker:
    """Apply quality gates, build features, and generate one RUL prediction record."""

    def __init__(
        self,
        *,
        predictor: RULPredictor,
        model_release: str,
        feature_builder: FeatureBuilder | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not model_release.strip():
            raise ValueError("model_release must not be empty")
        self._predictor = predictor
        self._model_release = model_release
        self._feature_builder = feature_builder or FeatureBuilder()
        self._clock = clock or (lambda: datetime.now(UTC))

    def process(self, inference_input: InferenceInput) -> PredictionRecord:
        """Return a prediction decision for one event without persistence side effects."""

        quality_flags = list(inference_input.quality_flags)
        if inference_input.schema_version != TELEMETRY_SCHEMA_VERSION:
            _append_unique(
                quality_flags,
                f"incompatible_schema:{inference_input.schema_version}",
            )
        if inference_input.data_quality_status is DataQualityStatus.INVALID:
            _append_unique(quality_flags, "invalid_data_quality")

        if (
            inference_input.schema_version != TELEMETRY_SCHEMA_VERSION
            or inference_input.data_quality_status is DataQualityStatus.INVALID
        ):
            return self._withheld(inference_input, quality_flags)

        try:
            features = self._feature_builder.build(
                cycle=inference_input.cycle,
                measurements=inference_input.measurements,
            )
        except FeatureBuildError as error:
            _append_unique(quality_flags, f"feature_build_error:{error}")
            return self._withheld(inference_input, quality_flags)

        try:
            estimated_rul = float(self._predictor.predict(features))
        except Exception as error:
            raise InferenceExecutionError(
                f"model {self._model_release} could not score event {inference_input.event_id}"
            ) from error
        if not math.isfinite(estimated_rul) or estimated_rul < 0:
            raise InferenceExecutionError(
                f"model {self._model_release} returned an invalid RUL for "
                f"event {inference_input.event_id}"
            )

        prediction_status = (
            PredictionStatus.DEGRADED
            if inference_input.data_quality_status is DataQualityStatus.DEGRADED
            else PredictionStatus.AVAILABLE
        )
        return PredictionRecord(
            event_id=inference_input.event_id,
            engine_id=inference_input.engine_id,
            cycle=inference_input.cycle,
            estimated_rul=estimated_rul,
            status=prediction_status,
            data_quality_status=inference_input.data_quality_status,
            quality_flags=quality_flags,
            model_release=self._model_release,
            feature_version=features.feature_version,
            generated_at=self._clock(),
        )

    def _withheld(
        self,
        inference_input: InferenceInput,
        quality_flags: list[str],
    ) -> PredictionRecord:
        return PredictionRecord(
            event_id=inference_input.event_id,
            engine_id=inference_input.engine_id,
            cycle=inference_input.cycle,
            estimated_rul=None,
            status=PredictionStatus.WITHHELD,
            data_quality_status=DataQualityStatus.INVALID,
            quality_flags=quality_flags,
            model_release=self._model_release,
            feature_version=FEATURE_VERSION,
            generated_at=self._clock(),
        )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
