"""Delayed-outcome persistence and rolling model-performance checks."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Engine, text

from predictive_maintenance.monitoring.evaluation import MonitoringCheck, MonitoringStatus


@dataclass(frozen=True)
class PerformanceWindow:
    """Aligned predictions and observed RUL outcomes for one window."""

    model_release: str
    sample_count: int
    mae_cycles: float
    rmse_cycles: float


def record_prediction_outcome(
    engine: Engine,
    *,
    prediction_id: int,
    actual_rul_cycles: float,
    observed_at: datetime,
    source: str,
) -> None:
    """Record one delayed outcome idempotently for a prediction."""

    if prediction_id <= 0:
        raise ValueError("prediction_id must be positive")
    if actual_rul_cycles < 0:
        raise ValueError("actual_rul_cycles must be non-negative")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must include a timezone offset")
    if not source.strip():
        raise ValueError("source must not be empty")
    statement = text(
        """
        INSERT INTO prediction_outcome (
            prediction_id, actual_rul_cycles, observed_at, source
        ) VALUES (
            :prediction_id, :actual_rul_cycles, :observed_at, :source
        )
        ON CONFLICT (prediction_id) DO UPDATE SET
            actual_rul_cycles = EXCLUDED.actual_rul_cycles,
            observed_at = EXCLUDED.observed_at,
            source = EXCLUDED.source
        """
    )
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "prediction_id": prediction_id,
                "actual_rul_cycles": actual_rul_cycles,
                "observed_at": observed_at,
                "source": source,
            },
        )


def load_performance_window(
    engine: Engine,
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[PerformanceWindow]:
    """Calculate rolling MAE and RMSE grouped by model release."""

    if window_start.tzinfo is None or window_end.tzinfo is None:
        raise ValueError("performance window must use timezone-aware datetimes")
    if window_start >= window_end:
        raise ValueError("window_start must be before window_end")
    statement = text(
        """
        SELECT
            p.model_release,
            count(*) AS sample_count,
            avg(abs(p.rul_cycles - o.actual_rul_cycles)) AS mae_cycles,
            sqrt(avg((p.rul_cycles - o.actual_rul_cycles) *
                     (p.rul_cycles - o.actual_rul_cycles))) AS rmse_cycles
        FROM prediction AS p
        JOIN prediction_outcome AS o ON o.prediction_id = p.prediction_id
        WHERE p.rul_cycles IS NOT NULL
          AND o.observed_at >= :window_start
          AND o.observed_at < :window_end
        GROUP BY p.model_release
        ORDER BY p.model_release
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(
            statement,
            {"window_start": window_start, "window_end": window_end},
        ).mappings().all()
    return [
        PerformanceWindow(
            model_release=str(row["model_release"]),
            sample_count=int(row["sample_count"]),
            mae_cycles=float(row["mae_cycles"]),
            rmse_cycles=float(row["rmse_cycles"]),
        )
        for row in rows
    ]


def evaluate_model_performance(
    performance: PerformanceWindow | None,
    *,
    baseline_mae_cycles: float,
    minimum_samples: int = 100,
    warning_multiplier: float = 1.10,
    fail_multiplier: float = 1.20,
) -> MonitoringCheck:
    """Classify rolling MAE against an approved champion baseline."""

    if baseline_mae_cycles <= 0 or minimum_samples <= 0:
        raise ValueError("baseline and minimum samples must be positive")
    if not warning_multiplier < fail_multiplier:
        raise ValueError("warning multiplier must be below fail multiplier")
    if performance is None or performance.sample_count < minimum_samples:
        return MonitoringCheck(
            name=(
                f"model_mae_degradation:{performance.model_release}"
                if performance
                else "model_mae_degradation"
            ),
            value=performance.mae_cycles if performance else 0.0,
            status=MonitoringStatus.NO_DATA,
            warning_threshold=baseline_mae_cycles * warning_multiplier,
            fail_threshold=baseline_mae_cycles * fail_multiplier,
        )
    warning_threshold = baseline_mae_cycles * warning_multiplier
    fail_threshold = baseline_mae_cycles * fail_multiplier
    status = (
        MonitoringStatus.FAIL
        if performance.mae_cycles >= fail_threshold
        else MonitoringStatus.WARNING
        if performance.mae_cycles >= warning_threshold
        else MonitoringStatus.PASS
    )
    return MonitoringCheck(
        name=f"model_mae_degradation:{performance.model_release}",
        value=performance.mae_cycles,
        status=status,
        warning_threshold=warning_threshold,
        fail_threshold=fail_threshold,
    )
