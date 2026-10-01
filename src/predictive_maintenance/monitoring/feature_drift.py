"""Load persisted sensor windows and calculate feature PSI checks."""

from datetime import datetime

from sqlalchemy import Engine, text

from predictive_maintenance.monitoring.drift import calculate_psi
from predictive_maintenance.monitoring.evaluation import MonitoringCheck, MonitoringStatus


def load_sensor_windows(
    engine: Engine,
    *,
    reference_start: datetime,
    current_start: datetime,
    current_end: datetime,
) -> dict[str, tuple[list[float], list[float]]]:
    """Load reference and current numeric sensor values from telemetry JSON."""

    if not all(
        value.tzinfo is not None and value.utcoffset() is not None
        for value in (reference_start, current_start, current_end)
    ):
        raise ValueError("drift windows must use timezone-aware datetimes")
    if not reference_start < current_start < current_end:
        raise ValueError("drift windows must be ordered")
    statement = text(
        """
        SELECT
            CASE
                WHEN ingestion_timestamp >= :reference_start
                 AND ingestion_timestamp < :current_start THEN 'reference'
                ELSE 'current'
            END AS window_name,
            item.key AS feature_name,
            item.value::double precision AS feature_value
        FROM telemetry_event
        CROSS JOIN LATERAL jsonb_each_text(measurements) AS item
        WHERE measurements IS NOT NULL
          AND ingestion_timestamp >= :reference_start
          AND ingestion_timestamp < :current_end
        """
    )
    windows: dict[str, tuple[list[float], list[float]]] = {}
    with engine.connect() as connection:
        rows = connection.execute(
            statement,
            {
                "reference_start": reference_start,
                "current_start": current_start,
                "current_end": current_end,
            },
        ).mappings().all()
    for row in rows:
        reference, current = windows.setdefault(str(row["feature_name"]), ([], []))
        (reference if row["window_name"] == "reference" else current).append(
            float(row["feature_value"])
        )
    return windows


def calculate_feature_drift_checks(
    windows: dict[str, tuple[list[float], list[float]]],
    *,
    minimum_samples: int = 20,
) -> tuple[MonitoringCheck, ...]:
    """Calculate PSI checks only when both windows have enough observations."""

    checks: list[MonitoringCheck] = []
    for feature_name, (reference, current) in sorted(windows.items()):
        if len(reference) < minimum_samples or len(current) < minimum_samples:
            continue
        result = calculate_psi(reference, current)
        checks.append(
            MonitoringCheck(
                name=f"feature_drift:{feature_name}",
                value=result.psi,
                status=MonitoringStatus.WARNING if result.drifted else MonitoringStatus.PASS,
                warning_threshold=result.threshold,
            )
        )
    return tuple(checks)
