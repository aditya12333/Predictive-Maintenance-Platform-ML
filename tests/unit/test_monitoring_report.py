import pytest

from predictive_maintenance.monitoring.evaluation import MonitoringStatus
from predictive_maintenance.monitoring.report import (
    MonitoringWindowCounts,
    build_operational_checks,
)


def test_build_operational_checks_classifies_persisted_rates() -> None:
    checks = build_operational_checks(
        MonitoringWindowCounts(
            telemetry_events=100,
            quality_issues=2,
            predictions=100,
            available_predictions=96,
            degraded_predictions=3,
            withheld_predictions=0,
        )
    )

    assert [check.status for check in checks] == [
        MonitoringStatus.PASS,
        MonitoringStatus.PASS,
        MonitoringStatus.PASS,
        MonitoringStatus.PASS,
    ]


def test_build_operational_checks_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="valid count"):
        build_operational_checks(
            MonitoringWindowCounts(
                telemetry_events=10,
                quality_issues=0,
                predictions=2,
                available_predictions=3,
                degraded_predictions=0,
                withheld_predictions=0,
            )
        )
