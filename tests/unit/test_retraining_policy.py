from predictive_maintenance.monitoring.evaluation import MonitoringCheck, MonitoringStatus
from predictive_maintenance.monitoring.retraining_policy import (
    RetrainingDecision,
    evaluate_retraining_policy,
)


def _window(status: MonitoringStatus) -> list[MonitoringCheck]:
    return [MonitoringCheck("model_mae_degradation", 25.0, status)]


def test_policy_waits_for_sustained_failures() -> None:
    result = evaluate_retraining_policy([_window(MonitoringStatus.FAIL)])

    assert result.decision is RetrainingDecision.WAIT


def test_policy_requests_candidate_after_two_failures() -> None:
    result = evaluate_retraining_policy(
        [_window(MonitoringStatus.FAIL), _window(MonitoringStatus.FAIL)]
    )

    assert result.decision is RetrainingDecision.REQUEST_CANDIDATE


def test_policy_does_not_trigger_on_no_data_or_warning() -> None:
    result = evaluate_retraining_policy(
        [_window(MonitoringStatus.FAIL), _window(MonitoringStatus.WARNING)]
    )

    assert result.decision is RetrainingDecision.WAIT
