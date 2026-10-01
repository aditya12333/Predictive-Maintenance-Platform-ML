import pytest

from predictive_maintenance.monitoring.evaluation import (
    MonitoringStatus,
    evaluate_rate,
    evaluate_rul_error,
)


def test_evaluate_rate_classifies_quality_rate() -> None:
    result = evaluate_rate(
        "quality_issue_rate",
        2,
        20,
        warning_threshold=0.05,
        fail_threshold=0.10,
    )

    assert result.value == 0.1
    assert result.status is MonitoringStatus.FAIL


def test_evaluate_rate_classifies_lower_is_better_availability() -> None:
    result = evaluate_rate(
        "prediction_availability",
        17,
        20,
        warning_threshold=0.95,
        fail_threshold=0.90,
        higher_is_worse=False,
    )

    assert result.status is MonitoringStatus.FAIL


def test_evaluate_rul_error_returns_mae_and_rmse() -> None:
    mae, rmse = evaluate_rul_error([10, 20], [13, 16])

    assert mae.value == 3.5
    assert rmse.value == pytest.approx(3.5355339059)


def test_evaluate_rate_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="valid count"):
        evaluate_rate("issues", 3, 2, warning_threshold=0.1, fail_threshold=0.2)


def test_evaluate_rate_reports_no_data_for_empty_window() -> None:
    result = evaluate_rate(
        "prediction_availability",
        0,
        0,
        warning_threshold=0.95,
        fail_threshold=0.90,
        higher_is_worse=False,
    )

    assert result.status is MonitoringStatus.NO_DATA
