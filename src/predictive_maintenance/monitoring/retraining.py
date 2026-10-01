"""Load monitoring history and apply the retraining policy."""

import json
from pathlib import Path

from predictive_maintenance.monitoring.evaluation import MonitoringCheck, MonitoringStatus
from predictive_maintenance.monitoring.retraining_policy import (
    RetrainingPolicyResult,
    evaluate_retraining_policy,
)


def evaluate_report_history(
    history_directory: Path,
    *,
    required_failed_windows: int = 2,
) -> RetrainingPolicyResult:
    """Evaluate the newest monitoring reports in chronological order."""

    reports = sorted(history_directory.glob("*.json"))
    windows: list[list[MonitoringCheck]] = []
    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        windows.append(
            [
                MonitoringCheck(
                    name=str(check["name"]),
                    value=float(check["value"]),
                    status=MonitoringStatus(str(check["status"])),
                    warning_threshold=check.get("warning_threshold"),
                    fail_threshold=check.get("fail_threshold"),
                )
                for check in payload.get("checks", [])
            ]
        )
    return evaluate_retraining_policy(
        windows,
        required_failed_windows=required_failed_windows,
    )
