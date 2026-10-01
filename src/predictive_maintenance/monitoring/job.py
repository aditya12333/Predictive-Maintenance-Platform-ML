"""Run the database-backed operational monitoring report."""

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from predictive_maintenance.core.settings import PlatformSettings
from predictive_maintenance.monitoring.feature_drift import (
    calculate_feature_drift_checks,
    load_sensor_windows,
)
from predictive_maintenance.monitoring.report import (
    build_operational_checks,
    load_monitoring_window_counts,
)
from predictive_maintenance.storage.database import create_database_engine
from predictive_maintenance.storage.model_performance import (
    evaluate_model_performance,
    load_performance_window,
)


def run_monitoring_job(
    settings: PlatformSettings,
    *,
    output_path: Path,
    window_hours: int = 24,
    now: datetime | None = None,
) -> Path:
    """Write one monitoring report for the requested trailing time window."""

    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    report_time = now or datetime.now(UTC)
    if report_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    window_start = report_time - timedelta(hours=window_hours)
    reference_start = window_start - timedelta(hours=window_hours)
    engine = create_database_engine(settings)
    try:
        counts = load_monitoring_window_counts(
            engine,
            window_start=window_start,
            window_end=report_time,
        )
        performance_windows = load_performance_window(
            engine,
            window_start=window_start,
            window_end=report_time,
        )
        sensor_windows = load_sensor_windows(
            engine,
            reference_start=reference_start,
            current_start=window_start,
            current_end=report_time,
        )
    finally:
        engine.dispose()

    checks = list(build_operational_checks(counts))
    checks.extend(calculate_feature_drift_checks(sensor_windows))
    if performance_windows:
        checks.extend(
            evaluate_model_performance(
                performance,
                baseline_mae_cycles=settings.monitoring_baseline_mae_cycles,
                minimum_samples=settings.monitoring_minimum_labelled_samples,
            )
            for performance in performance_windows
        )
    else:
        checks.append(
            evaluate_model_performance(
                None,
                baseline_mae_cycles=settings.monitoring_baseline_mae_cycles,
                minimum_samples=settings.monitoring_minimum_labelled_samples,
            )
        )
    report = {
        "generated_at": report_time.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": report_time.isoformat(),
        "counts": asdict(counts),
        "checks": [asdict(check) for check in checks],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2) + "\n"
    output_path.write_text(serialized, encoding="utf-8")
    history_path = output_path.parent / "history" / f"{report_time.strftime('%Y%m%dT%H%M%SZ')}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(serialized, encoding="utf-8")
    return output_path
