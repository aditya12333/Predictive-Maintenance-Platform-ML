"""Transactional alert evaluation, deduplication, lifecycle, and audit records."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from psycopg.types.json import Jsonb
from sqlalchemy import Connection, Engine, text

from predictive_maintenance.api.contracts import (
    AlertAction,
    AlertState,
    AlertTransitionRequest,
    AlertView,
)
from predictive_maintenance.inference.contracts import PredictionRecord, PredictionStatus
from predictive_maintenance.storage.operational import _alert_from_row


@dataclass(frozen=True)
class AlertPolicy:
    """Optional operational thresholds used to turn predictions into alerts."""

    warning_rul_cycles: float | None = None
    critical_rul_cycles: float | None = None
    data_quality_alerts_enabled: bool = True

    def __post_init__(self) -> None:
        if self.warning_rul_cycles is not None and self.warning_rul_cycles < 0:
            raise ValueError("warning_rul_cycles must be non-negative")
        if self.critical_rul_cycles is not None and self.critical_rul_cycles < 0:
            raise ValueError("critical_rul_cycles must be non-negative")
        if (
            self.warning_rul_cycles is not None
            and self.critical_rul_cycles is not None
            and self.critical_rul_cycles > self.warning_rul_cycles
        ):
            raise ValueError("critical_rul_cycles must be less than or equal to warning_rul_cycles")


def persist_prediction_alerts_transaction(
    connection: Connection,
    *,
    prediction: PredictionRecord,
    policy: AlertPolicy,
    now: datetime | None = None,
) -> int:
    """Create or refresh deduplicated alerts for one persisted prediction."""

    alert_time = now or datetime.now(UTC)
    if alert_time.tzinfo is None or alert_time.utcoffset() is None:
        raise ValueError("now must include a timezone offset")
    decisions: list[dict[str, Any]] = []
    if policy.data_quality_alerts_enabled:
        for flag in dict.fromkeys(prediction.quality_flags):
            base_flag = flag.split(":", 1)[0].strip() or "unknown"
            decisions.append(
                {
                    "alert_type": "DATA_QUALITY",
                    "severity": _quality_alert_severity(base_flag),
                    "deduplication_key": f"data-quality:{prediction.engine_id}:{base_flag}",
                    "title": f"Data quality: {base_flag.replace('_', ' ').title()}",
                    "message": flag,
                }
            )
    if prediction.status is not PredictionStatus.WITHHELD and prediction.estimated_rul is not None:
        severity: str | None = None
        if (
            policy.critical_rul_cycles is not None
            and prediction.estimated_rul <= policy.critical_rul_cycles
        ):
            severity = "CRITICAL"
        elif (
            policy.warning_rul_cycles is not None
            and prediction.estimated_rul <= policy.warning_rul_cycles
        ):
            severity = "WARNING"
        if severity is not None:
            decisions.append(
                {
                    "alert_type": "EQUIPMENT_RISK",
                    "severity": severity,
                    "deduplication_key": f"equipment-risk:{prediction.engine_id}",
                    "title": (
                        "Critical equipment risk"
                        if severity == "CRITICAL"
                        else "Equipment risk warning"
                    ),
                    "message": f"Estimated RUL is {prediction.estimated_rul:g} cycles.",
                }
            )
    if not decisions:
        return 0
    prediction_row = (
        connection.execute(
            text(
                """
                SELECT prediction_id
                FROM prediction
                WHERE event_id = :event_id AND model_release = :model_release
                ORDER BY generated_at DESC, prediction_id DESC
                LIMIT 1
                """
            ),
            {"event_id": prediction.event_id, "model_release": prediction.model_release},
        )
        .mappings()
        .one_or_none()
    )
    if prediction_row is None:
        raise ValueError(f"prediction for event {prediction.event_id} was not persisted")
    params_base = {
        "engine_id": prediction.engine_id,
        "source_event_id": prediction.event_id,
        "source_prediction_id": int(prediction_row["prediction_id"]),
        "opened_at": alert_time,
        "updated_at": alert_time,
    }
    for decision in decisions:
        connection.execute(
            text(
                """
                INSERT INTO alert (
                    engine_id, alert_type, state, severity, deduplication_key,
                    title, message, source_event_id, source_prediction_id,
                    opened_at, updated_at, version
                ) VALUES (
                    :engine_id, :alert_type, 'OPEN', :severity, :deduplication_key,
                    :title, :message, :source_event_id, :source_prediction_id,
                    :opened_at, :updated_at, 1
                )
                ON CONFLICT (deduplication_key) WHERE resolved_at IS NULL DO UPDATE SET
                    severity = EXCLUDED.severity,
                    title = EXCLUDED.title,
                    message = EXCLUDED.message,
                    source_event_id = EXCLUDED.source_event_id,
                    source_prediction_id = EXCLUDED.source_prediction_id,
                    updated_at = EXCLUDED.updated_at,
                    version = alert.version + 1
                """
            ),
            {**params_base, **decision},
        )
    return len(decisions)


def _quality_alert_severity(flag: str) -> str:
    if flag in {"invalid_data_quality", "incompatible_schema", "feature_build_error"}:
        return "CRITICAL"
    return "WARNING"


class AlertLifecycleError(RuntimeError):
    """Base error for an alert transition that cannot be applied."""


class AlertNotFoundError(AlertLifecycleError):
    """Raised when the requested alert does not exist."""


class AlertVersionConflictError(AlertLifecycleError):
    """Raised when a human update is based on an old alert version."""


class InvalidAlertTransitionError(AlertLifecycleError):
    """Raised when an alert is already closed or the action is not allowed."""


def transition_alert(
    engine: Engine,
    *,
    alert_id: int,
    request: AlertTransitionRequest,
    now: datetime | None = None,
) -> AlertView:
    """Apply one version-checked human action and write its audit event atomically."""

    if alert_id <= 0:
        raise ValueError("alert_id must be positive")
    transition_time = now or datetime.now(UTC)
    if transition_time.tzinfo is None or transition_time.utcoffset() is None:
        raise ValueError("now must include a timezone offset")

    with engine.begin() as connection:
        row = _lock_alert(connection, alert_id)
        current_state = AlertState(str(row["state"]).lower())
        current_version = int(row["version"])
        if request.expected_version != current_version:
            raise AlertVersionConflictError(
                f"alert {alert_id} is version {current_version}; "
                f"request expected {request.expected_version}"
            )
        new_state = _next_state(current_state, request.action)
        acknowledged_at = row["acknowledged_at"]
        acknowledged_by = row["acknowledged_by"]
        resolved_at = row["resolved_at"]
        if request.action is AlertAction.ACKNOWLEDGE:
            acknowledged_at = transition_time
            acknowledged_by = request.actor_id
        if request.action in {AlertAction.DISMISS, AlertAction.RESOLVE}:
            resolved_at = transition_time

        connection.execute(
            text(
                """
                UPDATE alert
                SET
                    state = :state,
                    updated_at = :updated_at,
                    acknowledged_at = :acknowledged_at,
                    acknowledged_by = :acknowledged_by,
                    resolved_at = :resolved_at,
                    version = version + 1
                WHERE alert_id = :alert_id AND version = :expected_version
                """
            ),
            {
                "alert_id": alert_id,
                "state": new_state.value.upper(),
                "updated_at": transition_time,
                "acknowledged_at": acknowledged_at,
                "acknowledged_by": acknowledged_by,
                "resolved_at": resolved_at,
                "expected_version": request.expected_version,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO audit_event (
                    event_type, actor_id, subject_type, subject_id, details
                ) VALUES (
                    :event_type, :actor_id, 'alert', :subject_id, :details
                )
                """
            ),
            {
                "event_type": f"alert.{request.action.value}",
                "actor_id": request.actor_id,
                "subject_id": str(alert_id),
                "details": Jsonb(
                    {
                        "old_state": current_state.value,
                        "new_state": new_state.value,
                        "old_version": current_version,
                        "new_version": current_version + 1,
                        "reason": request.reason,
                    }
                ),
            },
        )
        updated = _lock_alert(connection, alert_id, for_update=False)
        return _alert_from_row(updated)


def _lock_alert(connection: Connection, alert_id: int, *, for_update: bool = True) -> Any:
    lock_clause = " FOR UPDATE" if for_update else ""
    row = (
        connection.execute(
            text(
                f"""
                SELECT alert_id, engine_id, alert_type, state, severity,
                       deduplication_key, title, message, source_event_id, source_prediction_id,
                       opened_at, updated_at, acknowledged_at, acknowledged_by,
                       resolved_at, version
                FROM alert
                WHERE alert_id = :alert_id
                {lock_clause}
                """
            ),
            {"alert_id": alert_id},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise AlertNotFoundError(f"alert {alert_id} was not found")
    return row


def _next_state(current: AlertState, action: AlertAction) -> AlertState:
    allowed: dict[AlertState, dict[AlertAction, AlertState]] = {
        AlertState.OPEN: {
            AlertAction.ACKNOWLEDGE: AlertState.ACKNOWLEDGED,
            AlertAction.DISMISS: AlertState.DISMISSED,
            AlertAction.RESOLVE: AlertState.RESOLVED,
        },
        AlertState.ACKNOWLEDGED: {
            AlertAction.DISMISS: AlertState.DISMISSED,
            AlertAction.RESOLVE: AlertState.RESOLVED,
        },
        AlertState.RESOLVED: {},
        AlertState.DISMISSED: {},
    }
    try:
        return allowed[current][action]
    except KeyError as error:
        raise InvalidAlertTransitionError(
            f"cannot {action.value} an alert in {current.value} state"
        ) from error
