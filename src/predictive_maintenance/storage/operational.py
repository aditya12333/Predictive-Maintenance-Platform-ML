"""Read-only queries that assemble dashboard and operational API responses."""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Engine, text

from predictive_maintenance.api.contracts import (
    AlertListResponse,
    AlertSeverity,
    AlertState,
    AlertType,
    AlertView,
    DataQualityIssueView,
    EquipmentDetailResponse,
    FleetEngineSummary,
    FleetHealthResponse,
    FleetHealthStatus,
    PredictionHistoryResponse,
    PredictionView,
)
from predictive_maintenance.inference.contracts import DataQualityStatus, PredictionStatus


def load_fleet_health(
    engine: Engine,
    *,
    as_of: datetime,
    stale_after: timedelta,
) -> FleetHealthResponse:
    """Return one current-state row per observed engine."""

    _require_aware_datetime(as_of)
    _require_positive_duration(stale_after)
    statement = text(
        """
        WITH latest_event AS (
            SELECT DISTINCT ON (engine_id)
                event_id,
                engine_id,
                cycle AS latest_received_cycle,
                event_timestamp AS latest_event_timestamp
            FROM telemetry_event
            ORDER BY engine_id, cycle DESC, event_timestamp DESC, ingestion_timestamp DESC
        ),
        latest_prediction AS (
            SELECT DISTINCT ON (t.engine_id)
                t.engine_id,
                p.event_id,
                t.cycle,
                t.event_timestamp,
                t.ingestion_timestamp,
                p.generated_at,
                p.rul_cycles,
                p.status,
                p.data_quality_status,
                p.quality_flags,
                p.model_release,
                p.feature_version
            FROM prediction AS p
            JOIN telemetry_event AS t ON t.event_id = p.event_id
            ORDER BY t.engine_id, t.cycle DESC, p.generated_at DESC, p.prediction_id DESC
        ),
        open_alerts AS (
            SELECT engine_id, count(*) AS open_alert_count
            FROM alert
            WHERE state IN ('OPEN', 'ACKNOWLEDGED') AND resolved_at IS NULL
            GROUP BY engine_id
        ),
        open_quality AS (
            SELECT engine_id, count(*) AS open_quality_issue_count
            FROM data_quality_issue
            WHERE resolved_at IS NULL
            GROUP BY engine_id
        )
        SELECT
            e.engine_id,
            e.latest_received_cycle,
            e.latest_event_timestamp,
            p.event_id,
            p.cycle AS latest_scored_cycle,
            p.event_timestamp AS prediction_event_timestamp,
            p.ingestion_timestamp AS prediction_ingestion_timestamp,
            p.generated_at,
            p.rul_cycles,
            p.status,
            p.data_quality_status,
            p.quality_flags,
            p.model_release,
            p.feature_version,
            COALESCE(a.open_alert_count, 0) AS open_alert_count,
            COALESCE(q.open_quality_issue_count, 0) AS open_quality_issue_count
        FROM latest_event AS e
        LEFT JOIN latest_prediction AS p ON p.event_id = e.event_id
        LEFT JOIN open_alerts AS a ON a.engine_id = e.engine_id
        LEFT JOIN open_quality AS q ON q.engine_id = e.engine_id
        ORDER BY e.engine_id
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    engines: list[FleetEngineSummary] = []
    counts = {status: 0 for status in FleetHealthStatus}
    for row in rows:
        latest_prediction = _prediction_from_row(row) if row["event_id"] is not None else None
        health_status = _derive_health_status(
            latest_prediction=latest_prediction,
            latest_event_timestamp=row["latest_event_timestamp"],
            as_of=as_of,
            stale_after=stale_after,
        )
        counts[health_status] += 1
        engines.append(
            FleetEngineSummary(
                engine_id=int(row["engine_id"]),
                health_status=health_status,
                latest_received_cycle=int(row["latest_received_cycle"]),
                latest_scored_cycle=(
                    int(row["latest_scored_cycle"])
                    if row["latest_scored_cycle"] is not None
                    else None
                ),
                latest_event_timestamp=row["latest_event_timestamp"],
                latest_prediction=latest_prediction,
                open_alert_count=int(row["open_alert_count"]),
                open_data_quality_issue_count=int(row["open_quality_issue_count"]),
            )
        )

    return FleetHealthResponse(
        generated_at=as_of,
        total_engines=len(engines),
        current_engines=counts[FleetHealthStatus.CURRENT],
        stale_engines=counts[FleetHealthStatus.STALE],
        degraded_engines=counts[FleetHealthStatus.DEGRADED],
        withheld_engines=counts[FleetHealthStatus.WITHHELD],
        unavailable_engines=counts[FleetHealthStatus.UNAVAILABLE],
        engines=engines,
    )


def load_equipment_detail(
    engine: Engine,
    *,
    engine_id: int,
    as_of: datetime,
    stale_after: timedelta,
) -> EquipmentDetailResponse:
    """Return current state, last valid prediction, alerts, and quality issues."""

    _require_engine_id(engine_id)
    _require_aware_datetime(as_of)
    _require_positive_duration(stale_after)
    current_statement = text(
        """
        SELECT
            t.engine_id,
            t.cycle AS latest_received_cycle,
            t.event_timestamp AS latest_event_timestamp,
            p.event_id,
            p.cycle AS latest_scored_cycle,
            p.event_timestamp AS prediction_event_timestamp,
            p.ingestion_timestamp AS prediction_ingestion_timestamp,
            p.generated_at,
            p.rul_cycles,
            p.status,
            p.data_quality_status,
            p.quality_flags,
            p.model_release,
            p.feature_version
        FROM telemetry_event AS t
        LEFT JOIN LATERAL (
            SELECT
                p.event_id,
                t2.cycle,
                t2.event_timestamp,
                t2.ingestion_timestamp,
                p.generated_at,
                p.rul_cycles,
                p.status,
                p.data_quality_status,
                p.quality_flags,
                p.model_release,
                p.feature_version
            FROM prediction AS p
            JOIN telemetry_event AS t2 ON t2.event_id = p.event_id
            WHERE p.event_id = t.event_id
            ORDER BY p.generated_at DESC, p.prediction_id DESC
            LIMIT 1
        ) AS p ON TRUE
        WHERE t.engine_id = :engine_id
        ORDER BY t.cycle DESC, t.event_timestamp DESC, t.ingestion_timestamp DESC
        LIMIT 1
        """
    )
    last_valid_statement = text(
        """
        SELECT
            t.event_id,
            t.engine_id,
            t.cycle,
            t.event_timestamp,
            t.ingestion_timestamp,
            p.generated_at,
            p.rul_cycles,
            p.status,
            p.data_quality_status,
            p.quality_flags,
            p.model_release,
            p.feature_version
        FROM prediction AS p
        JOIN telemetry_event AS t ON t.event_id = p.event_id
        WHERE t.engine_id = :engine_id
          AND p.status IN ('AVAILABLE', 'DEGRADED', 'available', 'degraded')
          AND p.rul_cycles IS NOT NULL
        ORDER BY t.cycle DESC, p.generated_at DESC, p.prediction_id DESC
        LIMIT 1
        """
    )
    quality_statement = text(
        """
        SELECT issue_id, event_id, engine_id, issue_type, severity, message,
               detected_at, resolved_at
        FROM data_quality_issue
        WHERE engine_id = :engine_id AND resolved_at IS NULL
        ORDER BY detected_at DESC, issue_id DESC
        """
    )
    alert_statement = text(
        """
        SELECT alert_id, engine_id, alert_type, state, severity,
               deduplication_key, title, message, source_event_id, source_prediction_id,
               opened_at, updated_at, acknowledged_at, acknowledged_by,
               resolved_at, version
        FROM alert
        WHERE engine_id = :engine_id
          AND resolved_at IS NULL
        ORDER BY updated_at DESC, alert_id DESC
        """
    )
    with engine.connect() as connection:
        current = (
            connection.execute(current_statement, {"engine_id": engine_id})
            .mappings()
            .one_or_none()
        )
        if current is None:
            raise LookupError(f"engine {engine_id} has no telemetry")
        last_valid = connection.execute(
            last_valid_statement,
            {"engine_id": engine_id},
        ).mappings().one_or_none()
        quality_rows = connection.execute(
            quality_statement,
            {"engine_id": engine_id},
        ).mappings().all()
        alert_rows = connection.execute(
            alert_statement,
            {"engine_id": engine_id},
        ).mappings().all()

    latest_prediction = _prediction_from_row(current) if current["event_id"] else None
    return EquipmentDetailResponse(
        engine_id=engine_id,
        health_status=_derive_health_status(
            latest_prediction=latest_prediction,
            latest_event_timestamp=current["latest_event_timestamp"],
            as_of=as_of,
            stale_after=stale_after,
        ),
        latest_received_cycle=int(current["latest_received_cycle"]),
        latest_scored_cycle=(
            int(current["latest_scored_cycle"])
            if current["latest_scored_cycle"] is not None
            else None
        ),
        latest_event_timestamp=current["latest_event_timestamp"],
        latest_prediction=latest_prediction,
        last_valid_prediction=(
            _prediction_from_row(last_valid) if last_valid is not None else None
        ),
        active_alerts=[_alert_from_row(row) for row in alert_rows],
        open_data_quality_issues=[
            DataQualityIssueView(
                issue_id=int(row["issue_id"]),
                event_id=row["event_id"],
                engine_id=(int(row["engine_id"]) if row["engine_id"] is not None else None),
                issue_type=str(row["issue_type"]),
                severity=_severity(row["severity"]),
                message=str(row["message"]),
                detected_at=row["detected_at"],
                resolved_at=row["resolved_at"],
            )
            for row in quality_rows
        ],
    )


def load_prediction_history(
    engine: Engine,
    *,
    engine_id: int,
    limit: int = 100,
) -> PredictionHistoryResponse:
    """Return immutable prediction history for one engine."""

    _require_engine_id(engine_id)
    _require_limit(limit)
    statement = text(
        """
        SELECT
            t.event_id,
            t.engine_id,
            t.cycle,
            t.event_timestamp,
            t.ingestion_timestamp,
            p.generated_at,
            p.rul_cycles,
            p.status,
            p.data_quality_status,
            p.quality_flags,
            p.model_release,
            p.feature_version
        FROM prediction AS p
        JOIN telemetry_event AS t ON t.event_id = p.event_id
        WHERE t.engine_id = :engine_id
        ORDER BY t.cycle DESC, p.generated_at DESC
        LIMIT :limit
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(
            statement,
            {"engine_id": engine_id, "limit": limit},
        ).mappings().all()
    return PredictionHistoryResponse(
        engine_id=engine_id,
        items=[_prediction_from_row(row) for row in rows],
    )


def load_alerts(
    engine: Engine,
    *,
    engine_id: int | None = None,
    state: AlertState | None = None,
    limit: int = 100,
) -> AlertListResponse:
    """Return active or historical alerts with optional engine and state filters."""

    if engine_id is not None:
        _require_engine_id(engine_id)
    _require_limit(limit)
    filters = []
    parameters: dict[str, Any] = {"limit": limit}
    if engine_id is not None:
        filters.append("engine_id = :engine_id")
        parameters["engine_id"] = engine_id
    if state is not None:
        filters.append("state = :state")
        parameters["state"] = state.value.upper()
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    statement = text(
        f"""
        SELECT alert_id, engine_id, alert_type, state, severity,
               deduplication_key, title, message, source_event_id, source_prediction_id,
               opened_at, updated_at, acknowledged_at, acknowledged_by,
               resolved_at, version
        FROM alert
        {where}
        ORDER BY updated_at DESC, alert_id DESC
        LIMIT :limit
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(statement, parameters).mappings().all()
    return AlertListResponse(items=[_alert_from_row(row) for row in rows])


def _prediction_from_row(row: Any) -> PredictionView:
    """Convert a joined SQL row into the public prediction contract."""

    return PredictionView(
        event_id=str(row["event_id"]),
        engine_id=int(row["engine_id"]),
        cycle=int(row.get("cycle") or row["latest_scored_cycle"]),
        event_timestamp=(row.get("event_timestamp") or row["prediction_event_timestamp"]),
        ingestion_timestamp=(
            row.get("ingestion_timestamp") or row["prediction_ingestion_timestamp"]
        ),
        generated_at=row["generated_at"],
        estimated_rul=(float(row["rul_cycles"]) if row["rul_cycles"] is not None else None),
        status=_prediction_status(row["status"]),
        data_quality_status=_quality_status(row["data_quality_status"]),
        quality_flags=list(row["quality_flags"] or []),
        model_release=str(row["model_release"]),
        feature_version=str(row["feature_version"]),
    )


def _alert_from_row(row: Any) -> AlertView:
    """Convert the current schema's alert columns into the public alert contract."""

    return AlertView(
        alert_id=int(row["alert_id"]),
        engine_id=int(row["engine_id"]),
        alert_type=_alert_type(row["alert_type"]),
        state=_alert_state(row["state"]),
        severity=_severity(row["severity"]),
        deduplication_key=str(row.get("deduplication_key") or f"alert:{row['alert_id']}"),
        title=str(row.get("title") or _alert_title(row["alert_type"])),
        message=str(row.get("message") or _alert_title(row["alert_type"])),
        source_event_id=row.get("source_event_id"),
        source_prediction_id=(
            int(row["source_prediction_id"])
            if row.get("source_prediction_id") is not None
            else None
        ),
        opened_at=row["opened_at"],
        updated_at=row["updated_at"],
        acknowledged_at=row.get("acknowledged_at"),
        acknowledged_by=row.get("acknowledged_by"),
        resolved_at=row["resolved_at"],
        version=int(row.get("version") or 1),
    )


def _derive_health_status(
    *,
    latest_prediction: PredictionView | None,
    latest_event_timestamp: datetime,
    as_of: datetime,
    stale_after: timedelta,
) -> FleetHealthStatus:
    if latest_prediction is None:
        return FleetHealthStatus.UNAVAILABLE
    if latest_prediction.status is PredictionStatus.WITHHELD:
        return FleetHealthStatus.WITHHELD
    if latest_event_timestamp < as_of - stale_after:
        return FleetHealthStatus.STALE
    if latest_prediction.status is PredictionStatus.DEGRADED:
        return FleetHealthStatus.DEGRADED
    return FleetHealthStatus.CURRENT


def _prediction_status(value: object) -> PredictionStatus:
    return PredictionStatus(str(value).lower())


def _quality_status(value: object) -> DataQualityStatus:
    return DataQualityStatus(str(value).lower())


def _alert_type(value: object) -> AlertType:
    normalized = str(value).lower()
    if normalized in {item.value for item in AlertType}:
        return AlertType(normalized)
    raise ValueError(f"unsupported alert type: {value}")


def _alert_state(value: object) -> AlertState:
    normalized = str(value).lower()
    if normalized in {item.value for item in AlertState}:
        return AlertState(normalized)
    raise ValueError(f"unsupported alert state: {value}")


def _severity(value: object) -> AlertSeverity:
    normalized = str(value).lower()
    if normalized == "error":
        normalized = AlertSeverity.CRITICAL.value
    return AlertSeverity(normalized)


def _alert_title(value: object) -> str:
    return str(value).replace("_", " ").strip().title()


def _require_engine_id(engine_id: int) -> None:
    if engine_id <= 0:
        raise ValueError("engine_id must be positive")


def _require_limit(limit: int) -> None:
    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")


def _require_aware_datetime(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of must include a timezone offset")


def _require_positive_duration(value: timedelta) -> None:
    if value <= timedelta(0):
        raise ValueError("stale_after must be positive")
