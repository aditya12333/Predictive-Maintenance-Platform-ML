"""Evaluation metrics shared by every RUL model candidate."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)

FAILURE_HORIZON_CYCLES = 28


class EvaluationError(ValueError):
    """Raised when RUL predictions cannot be evaluated safely."""


@dataclass(frozen=True)
class FailureHorizonEvaluation:
    """Binary maintenance-warning metrics derived from RUL predictions."""

    horizon_cycles: int
    actual_warning_count: int
    predicted_warning_count: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_alert_rate: float
    missed_failure_rate: float


@dataclass(frozen=True)
class ModelEvaluation:
    """Comparable validation metrics for one model candidate."""

    model_name: str
    sample_count: int
    mae_cycles: float
    rmse_cycles: float
    nasa_score: float
    failure_horizon: FailureHorizonEvaluation


def evaluate_predictions(
    *,
    model_name: str,
    actual_rul: ArrayLike,
    predicted_rul: ArrayLike,
) -> ModelEvaluation:
    """Evaluate RUL accuracy and the warning derived at the 28-cycle horizon."""

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
    failure_horizon = _evaluate_failure_horizon(
        actual_rul=actual,
        predicted_rul=processed_predictions,
    )

    if not all(math.isfinite(metric) for metric in (mae, rmse, score)):
        raise EvaluationError("evaluation produced a non-finite metric")

    return ModelEvaluation(
        model_name=model_name,
        sample_count=actual.size,
        mae_cycles=mae,
        rmse_cycles=rmse,
        nasa_score=score,
        failure_horizon=failure_horizon,
    )


def _evaluate_failure_horizon(
    *,
    actual_rul: NDArray[np.float64],
    predicted_rul: NDArray[np.float64],
) -> FailureHorizonEvaluation:
    """Evaluate warnings where RUL at or below 28 cycles means attention is due."""

    actual_warning = actual_rul <= FAILURE_HORIZON_CYCLES
    predicted_warning = predicted_rul <= FAILURE_HORIZON_CYCLES
    true_negatives, false_positives, false_negatives, true_positives = (
        int(value)
        for value in confusion_matrix(
            actual_warning,
            predicted_warning,
            labels=[False, True],
        ).ravel()
    )

    negative_count = true_negatives + false_positives
    positive_count = true_positives + false_negatives
    return FailureHorizonEvaluation(
        horizon_cycles=FAILURE_HORIZON_CYCLES,
        actual_warning_count=positive_count,
        predicted_warning_count=true_positives + false_positives,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=float(precision_score(actual_warning, predicted_warning, zero_division=0)),
        recall=float(recall_score(actual_warning, predicted_warning, zero_division=0)),
        f1_score=float(f1_score(actual_warning, predicted_warning, zero_division=0)),
        false_alert_rate=(false_positives / negative_count if negative_count else 0.0),
        missed_failure_rate=(false_negatives / positive_count if positive_count else 0.0),
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
