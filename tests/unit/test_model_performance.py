from predictive_maintenance.monitoring.evaluation import MonitoringStatus
from predictive_maintenance.storage.model_performance import (
    PerformanceWindow,
    evaluate_model_performance,
)


def test_model_performance_requires_minimum_labelled_sample() -> None:
    result = evaluate_model_performance(
        PerformanceWindow("rul-lightgbm-v1", 50, 25.0, 30.0),
        baseline_mae_cycles=20.0,
    )

    assert result.status is MonitoringStatus.NO_DATA


def test_model_performance_warns_when_mae_degrades() -> None:
    result = evaluate_model_performance(
        PerformanceWindow("rul-lightgbm-v1", 100, 23.0, 30.0),
        baseline_mae_cycles=20.0,
    )

    assert result.status is MonitoringStatus.WARNING


def test_model_performance_fails_when_mae_degrades_materially() -> None:
    result = evaluate_model_performance(
        PerformanceWindow("rul-lightgbm-v1", 100, 25.0, 30.0),
        baseline_mae_cycles=20.0,
    )

    assert result.status is MonitoringStatus.FAIL
