"""Conservative policy for deciding when to request a retraining candidate."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from predictive_maintenance.monitoring.evaluation import MonitoringCheck, MonitoringStatus


class RetrainingDecision(StrEnum):
    """Outcome of the retraining eligibility policy."""

    WAIT = "wait"
    REQUEST_CANDIDATE = "request_candidate"


@dataclass(frozen=True)
class RetrainingPolicyResult:
    """Explainable retraining decision."""

    decision: RetrainingDecision
    reason: str
    consecutive_failed_windows: int
    required_failed_windows: int


def evaluate_retraining_policy(
    monitoring_windows: Sequence[Sequence[MonitoringCheck]],
    *,
    check_names: frozenset[str] = frozenset(
        {"model_mae_degradation", "feature_drift:sensor_1"}
    ),
    required_failed_windows: int = 2,
) -> RetrainingPolicyResult:
    """Request a candidate only after repeated performance or drift failures.

    A window counts as failed when any selected check has status ``fail``. A
    ``no_data`` or warning result never starts retraining.
    """

    if required_failed_windows <= 0:
        raise ValueError("required_failed_windows must be positive")
    if not check_names:
        raise ValueError("check_names must not be empty")

    consecutive = 0
    for window in reversed(monitoring_windows):
        selected = [
            check
            for check in window
            if check.name in check_names
            or ("feature_drift:sensor_1" in check_names and check.name.startswith("feature_drift:"))
        ]
        if not selected or not any(check.status is MonitoringStatus.FAIL for check in selected):
            break
        consecutive += 1
        if consecutive >= required_failed_windows:
            return RetrainingPolicyResult(
                decision=RetrainingDecision.REQUEST_CANDIDATE,
                reason="selected monitoring checks failed in consecutive windows",
                consecutive_failed_windows=consecutive,
                required_failed_windows=required_failed_windows,
            )
    return RetrainingPolicyResult(
        decision=RetrainingDecision.WAIT,
        reason="insufficient consecutive monitoring failures",
        consecutive_failed_windows=consecutive,
        required_failed_windows=required_failed_windows,
    )
