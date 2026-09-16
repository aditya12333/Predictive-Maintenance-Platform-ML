"""Evaluation metrics shared by every RUL model candidate."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class EvaluationError(ValueError):
    """Raised when RUL predictions cannot be evaluated safely."""


@dataclass(frozen=True)
class ModelEvaluation:
    """Comparable validation metrics for one model candidate."""

    model_name: str
    sample_count: int
    mae_cycles: float
    rmse_cycles: float
    nasa_score: float


def evaluate_predictions(
    *,
    model_name: str,
    actual_rul: ArrayLike,
    predicted_rul: ArrayLike,
) -> ModelEvaluation:
    """Calculate MAE, RMSE, and NASA score after RUL postprocessing."""

    if not model_name.strip():
        raise EvaluationError("model_name must not be empty")

    actual = _as_valid_vector(actual_rul, name="actual_rul")
    predicted = _as_valid_vector(predicted_rul, name="predicted_rul")
    if actual.shape != predicted.shape:
        raise EvaluationError("actual_rul and predicted_rul must have the same length")
    if np.any(actual < 0):
        raise EvaluationError("actual_rul must contain only non-negative values")

    processed_predictions = np.maximum(0.0, predicted)
    errors = processed_predictions - actual
    mae = float(mean_absolute_error(actual, processed_predictions))
    rmse = float(root_mean_squared_error(actual, processed_predictions))
    score = _nasa_score(errors)

    if not all(math.isfinite(metric) for metric in (mae, rmse, score)):
        raise EvaluationError("evaluation produced a non-finite metric")

    return ModelEvaluation(
        model_name=model_name,
        sample_count=actual.size,
        mae_cycles=mae,
        rmse_cycles=rmse,
        nasa_score=score,
    )


def _as_valid_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{name} must contain numeric values") from error

    if vector.ndim != 1:
        raise EvaluationError(f"{name} must be a one-dimensional sequence")
    if vector.size == 0:
        raise EvaluationError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise EvaluationError(f"{name} must contain only finite values")
    return vector


def _nasa_score(errors: NDArray[np.float64]) -> float:
    """Calculate the asymmetric C-MAPSS score from prediction errors."""

    early_predictions = errors < 0
    penalties = np.empty_like(errors)
    try:
        with np.errstate(over="raise", invalid="raise"):
            penalties[early_predictions] = np.expm1(-errors[early_predictions] / 13.0)
            penalties[~early_predictions] = np.expm1(errors[~early_predictions] / 10.0)
            return float(np.sum(penalties, dtype=np.float64))
    except FloatingPointError as error:
        raise EvaluationError("NASA score overflowed for the supplied predictions") from error
