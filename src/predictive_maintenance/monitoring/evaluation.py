"""Pure monitoring checks used by scheduled observability jobs."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from statistics import mean

from predictive_maintenance.monitoring.drift import DriftResult


class MonitoringStatus(StrEnum):
    """Operational state of one monitoring check."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class MonitoringCheck:
    """A named metric together with thresholds and its operational status."""

    name: str
    value: float
    status: MonitoringStatus
    warning_threshold: float | None = None
    fail_threshold: float | None = None


def evaluate_rate(
    name: str,
    numerator: int,
    denominator: int,
    *,
    warning_threshold: float,
    fail_threshold: float,
    higher_is_worse: bool = True,
) -> MonitoringCheck:
    """Evaluate a bounded rate without allowing a zero denominator."""

    if not name.strip():
        raise ValueError("name must not be empty")
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("numerator and denominator must define a valid count")
    if warning_threshold < 0 or fail_threshold < 0:
        raise ValueError("thresholds must be non-negative")
    if higher_is_worse and warning_threshold > fail_threshold:
        raise ValueError("warning threshold must not exceed fail threshold")
    if not higher_is_worse and warning_threshold < fail_threshold:
        raise ValueError("warning threshold must not be below fail threshold")

    if denominator == 0:
        return MonitoringCheck(
            name=name,
            value=0.0,
            status=MonitoringStatus.NO_DATA,
            warning_threshold=warning_threshold,
            fail_threshold=fail_threshold,
        )

    value = numerator / denominator
    if higher_is_worse:
        status = (
            MonitoringStatus.FAIL
            if value >= fail_threshold
            else MonitoringStatus.WARNING
            if value >= warning_threshold
            else MonitoringStatus.PASS
        )
    else:
        status = (
            MonitoringStatus.FAIL
            if value <= fail_threshold
            else MonitoringStatus.WARNING
            if value <= warning_threshold
            else MonitoringStatus.PASS
        )
    return MonitoringCheck(
        name=name,
        value=value,
        status=status,
        warning_threshold=warning_threshold,
        fail_threshold=fail_threshold,
    )


def evaluate_drift(result: DriftResult, *, feature: str = "feature") -> MonitoringCheck:
    """Convert the existing PSI result into an operational check."""

    status = MonitoringStatus.WARNING if result.drifted else MonitoringStatus.PASS
    return MonitoringCheck(
        name=f"feature_drift:{feature}",
        value=result.psi,
        status=status,
        warning_threshold=result.threshold,
    )


def evaluate_rul_error(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    name: str = "rul_mae",
) -> tuple[MonitoringCheck, MonitoringCheck]:
    """Return MAE and RMSE checks for aligned actual and predicted RUL values."""

    if not actual or not predicted or len(actual) != len(predicted):
        raise ValueError("actual and predicted must be non-empty and equally sized")
    if any(value < 0 for value in actual) or any(value < 0 for value in predicted):
        raise ValueError("RUL values must be non-negative")

    errors = [
        prediction - observed
        for observed, prediction in zip(actual, predicted, strict=True)
    ]
    mae = mean(abs(error) for error in errors)
    rmse = sqrt(mean(error * error for error in errors))
    return (
        MonitoringCheck(name=name, value=mae, status=MonitoringStatus.PASS),
        MonitoringCheck(name=name.replace("mae", "rmse"), value=rmse, status=MonitoringStatus.PASS),
    )
