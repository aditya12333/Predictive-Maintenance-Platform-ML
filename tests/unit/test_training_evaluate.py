"""Tests for shared RUL model evaluation."""

import numpy as np
import pytest

from predictive_maintenance.training.evaluate import (
    EvaluationError,
    evaluate_predictions,
)


def test_perfect_predictions_have_zero_error() -> None:
    evaluation = evaluate_predictions(
        model_name="perfect_model",
        actual_rul=[30.0, 20.0, 10.0],
        predicted_rul=[30.0, 20.0, 10.0],
    )

    assert evaluation.model_name == "perfect_model"
    assert evaluation.sample_count == 3
    assert evaluation.mae_cycles == 0.0
    assert evaluation.rmse_cycles == 0.0
    assert evaluation.nasa_score == 0.0


def test_nasa_score_penalizes_late_warning_more_than_early_warning() -> None:
    early_warning = evaluate_predictions(
        model_name="early_warning",
        actual_rul=[10.0],
        predicted_rul=[0.0],
    )
    late_warning = evaluate_predictions(
        model_name="late_warning",
        actual_rul=[10.0],
        predicted_rul=[20.0],
    )

    assert early_warning.nasa_score == pytest.approx(np.expm1(10.0 / 13.0))
    assert late_warning.nasa_score == pytest.approx(np.expm1(10.0 / 10.0))
    assert late_warning.nasa_score > early_warning.nasa_score


def test_negative_predictions_are_clipped_before_evaluation() -> None:
    evaluation = evaluate_predictions(
        model_name="candidate",
        actual_rul=[0.0, 10.0],
        predicted_rul=[-5.0, 8.0],
    )

    assert evaluation.mae_cycles == pytest.approx(1.0)
    assert evaluation.rmse_cycles == pytest.approx(np.sqrt(2.0))
    assert evaluation.nasa_score == pytest.approx(np.expm1(2.0 / 13.0))


@pytest.mark.parametrize(
    ("actual_rul", "predicted_rul", "message"),
    [
        ([], [], "actual_rul must not be empty"),
        ([1.0, 2.0], [1.0], "must have the same length"),
        ([[1.0], [2.0]], [1.0, 2.0], "actual_rul must be a one-dimensional sequence"),
        ([1.0, np.nan], [1.0, 2.0], "actual_rul must contain only finite values"),
        ([1.0, 2.0], [1.0, np.inf], "predicted_rul must contain only finite values"),
        ([-1.0, 2.0], [1.0, 2.0], "actual_rul must contain only non-negative values"),
    ],
)
def test_invalid_evaluation_inputs_are_rejected(
    actual_rul: object,
    predicted_rul: object,
    message: str,
) -> None:
    with pytest.raises(EvaluationError, match=message):
        evaluate_predictions(
            model_name="candidate",
            actual_rul=actual_rul,
            predicted_rul=predicted_rul,
        )


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(EvaluationError, match="model_name must not be empty"):
        evaluate_predictions(model_name=" ", actual_rul=[1.0], predicted_rul=[1.0])


def test_nasa_score_overflow_is_reported() -> None:
    with pytest.raises(EvaluationError, match="NASA score overflowed"):
        evaluate_predictions(
            model_name="invalid_candidate",
            actual_rul=[0.0],
            predicted_rul=[10_000.0],
        )
