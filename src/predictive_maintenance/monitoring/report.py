"""Database-backed monitoring summaries for a time window."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, text

from predictive_maintenance.api.contracts import MonitoringReportResponse
from predictive_maintenance.monitoring.evaluation import MonitoringCheck, evaluate_rate


@dataclass(frozen=True)
class MonitoringWindowCounts:
    """Persisted counts used to build operational monitoring checks."""

    telemetry_events: int
    quality_issues: int
    predictions: int
    available_predictions: int
    degraded_predictions: int
    withheld_predictions: int


def load_monitoring_window_counts(
    engine: Engine,
    *,
    window_start: datetime,
    window_end: datetime,
) -> MonitoringWindowCounts:
    """Load quality and prediction counts for an inclusive-exclusive window."""

    if window_start.tzinfo is None or window_end.tzinfo is None:
        raise ValueError("monitoring window must use timezone-aware datetimes")
    if window_start >= window_end:
        raise ValueError("window_start must be before window_end")

    statement = text(
        """
        SELECT
            (SELECT count(*) FROM telemetry_event
             WHERE ingestion_timestamp >= :window_start
               AND ingestion_timestamp < :window_end) AS telemetry_events,
            (SELECT count(*) FROM data_quality_issue
             WHERE detected_at >= :window_start
               AND detected_at < :window_end) AS quality_issues,
            (SELECT count(*) FROM prediction
             WHERE generated_at >= :window_start
               AND generated_at < :window_end) AS predictions,
            (SELECT count(*) FROM prediction
             WHERE generated_at >= :window_start
               AND generated_at < :window_end
               AND lower(status) = 'available') AS available_predictions,
            (SELECT count(*) FROM prediction
             WHERE generated_at >= :window_start
               AND generated_at < :window_end
               AND lower(status) = 'degraded') AS degraded_predictions,
            (SELECT count(*) FROM prediction
             WHERE generated_at >= :window_start
               AND generated_at < :window_end
               AND lower(status) = 'withheld') AS withheld_predictions
        """
    )
    with engine.connect() as connection:
        row = connection.execute(
            statement,
            {"window_start": window_start, "window_end": window_end},
        ).mappings().one()
    return MonitoringWindowCounts(
        telemetry_events=int(row["telemetry_events"]),
        quality_issues=int(row["quality_issues"]),
        predictions=int(row["predictions"]),
        available_predictions=int(row["available_predictions"]),
        degraded_predictions=int(row["degraded_predictions"]),
        withheld_predictions=int(row["withheld_predictions"]),
    )


def build_operational_checks(counts: MonitoringWindowCounts) -> tuple[MonitoringCheck, ...]:
    """Classify persisted quality and prediction rates using provisional thresholds."""

    return (
        evaluate_rate(
            "quality_issue_rate",
            counts.quality_issues,
            counts.telemetry_events,
            warning_threshold=0.05,
            fail_threshold=0.10,
        ),
        evaluate_rate(
            "prediction_availability",
            counts.available_predictions,
            counts.predictions,
            warning_threshold=0.95,
            fail_threshold=0.90,
            higher_is_worse=False,
        ),
        evaluate_rate(
            "degraded_prediction_rate",
            counts.degraded_predictions,
            counts.predictions,
            warning_threshold=0.05,
            fail_threshold=0.10,
        ),
        evaluate_rate(
            "withheld_prediction_rate",
            counts.withheld_predictions,
            counts.predictions,
            warning_threshold=0.01,
            fail_threshold=0.05,
        ),
    )


def load_monitoring_report(path: Path) -> MonitoringReportResponse:
    """Load and validate the latest scheduled report from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return MonitoringReportResponse.model_validate(payload)
