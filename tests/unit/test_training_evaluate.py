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
    assert evaluation.failure_horizon.actual_warning_count == 2
    assert evaluation.failure_horizon.predicted_warning_count == 2
    assert evaluation.failure_horizon.precision == 1.0
    assert evaluation.failure_horizon.recall == 1.0
    assert evaluation.failure_horizon.f1_score == 1.0
    assert evaluation.failure_horizon.false_alert_rate == 0.0
    assert evaluation.failure_horizon.missed_failure_rate == 0.0


def test_failure_horizon_metrics_capture_false_alerts_and_missed_failures() -> None:
    evaluation = evaluate_predictions(
        model_name="candidate",
        actual_rul=[10.0, 20.0, 40.0, 50.0],
        predicted_rul=[15.0, 35.0, 25.0, 60.0],
    )

    warning = evaluation.failure_horizon
    assert warning.horizon_cycles == 28
    assert warning.actual_warning_count == 2
    assert warning.predicted_warning_count == 2
    assert warning.true_positives == 1
    assert warning.false_positives == 1
    assert warning.true_negatives == 1
    assert warning.false_negatives == 1
    assert warning.precision == pytest.approx(0.5)
    assert warning.recall == pytest.approx(0.5)
    assert warning.f1_score == pytest.approx(0.5)
    assert warning.false_alert_rate == pytest.approx(0.5)
    assert warning.missed_failure_rate == pytest.approx(0.5)


def test_failure_horizon_includes_exactly_28_cycles() -> None:
    evaluation = evaluate_predictions(
        model_name="boundary",
        actual_rul=[28.0, 29.0],
        predicted_rul=[28.0, 29.0],
    )

    warning = evaluation.failure_horizon
    assert warning.true_positives == 1
    assert warning.true_negatives == 1


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
