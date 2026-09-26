"""Stable response contracts for operational and dashboard APIs."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from predictive_maintenance.inference.contracts import (
    DataQualityStatus,
    PredictionStatus,
)


class FleetHealthStatus(StrEnum):
    """Operational interpretation of the latest equipment evidence."""

    CURRENT = "current"
    STALE = "stale"
    DEGRADED = "degraded"
    WITHHELD = "withheld"
    UNAVAILABLE = "unavailable"


class AlertType(StrEnum):
    """Distinct alert families shown to an operations user."""

    EQUIPMENT_RISK = "equipment_risk"
    DATA_QUALITY = "data_quality"


class AlertSeverity(StrEnum):
    """Shared severity vocabulary for equipment and data-quality alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(StrEnum):
    """Alert lifecycle state controlled by the processing and operations workflows."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class AlertAction(StrEnum):
    """Human action accepted by the alert transition endpoint."""

    ACKNOWLEDGE = "acknowledge"
    DISMISS = "dismiss"
    RESOLVE = "resolve"


class PredictionView(BaseModel):
    """One immutable prediction joined with the event that produced it."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    engine_id: int = Field(gt=0)
    cycle: int = Field(gt=0)
    event_timestamp: datetime
    ingestion_timestamp: datetime
    generated_at: datetime
    estimated_rul: float | None = Field(default=None, ge=0)
    status: PredictionStatus
    data_quality_status: DataQualityStatus
    quality_flags: list[str] = Field(default_factory=list)
    model_release: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prediction_state(self) -> "PredictionView":
        """Keep dashboard output consistent with the persisted prediction state."""

        if self.status is PredictionStatus.WITHHELD and self.estimated_rul is not None:
            raise ValueError("withheld predictions must not contain estimated_rul")
        if self.status is not PredictionStatus.WITHHELD and self.estimated_rul is None:
            raise ValueError("available or degraded predictions require estimated_rul")
        return self


class DataQualityIssueView(BaseModel):
    """An open or historical data-quality issue shown to an operator."""

    model_config = ConfigDict(extra="forbid")

    issue_id: int = Field(gt=0)
    event_id: str | None = Field(default=None, min_length=1)
    engine_id: int | None = Field(default=None, gt=0)
    issue_type: str = Field(min_length=1)
    severity: AlertSeverity
    message: str = Field(min_length=1)
    detected_at: datetime
    resolved_at: datetime | None = None


class AlertView(BaseModel):
    """Current alert state and the evidence that caused it."""

    model_config = ConfigDict(extra="forbid")

    alert_id: int = Field(gt=0)
    engine_id: int = Field(gt=0)
    alert_type: AlertType
    state: AlertState
    severity: AlertSeverity
    deduplication_key: str = Field(default="legacy", min_length=1)
    title: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source_event_id: str | None = Field(default=None, min_length=1)
    source_prediction_id: int | None = Field(default=None, gt=0)
    opened_at: datetime
    updated_at: datetime
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = Field(default=None, min_length=1)
    resolved_at: datetime | None = None
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_alert_state(self) -> "AlertView":
        """Require timestamps and actor details appropriate to the lifecycle state."""

        closed = self.state in {AlertState.RESOLVED, AlertState.DISMISSED}
        if closed and self.resolved_at is None:
            raise ValueError("resolved or dismissed alerts require resolved_at")
        if self.acknowledged_at is not None and self.acknowledged_by is None:
            raise ValueError("acknowledged alerts require acknowledged_by")
        return self


class FleetEngineSummary(BaseModel):
    """One row in the fleet overview."""

    model_config = ConfigDict(extra="forbid")

    engine_id: int = Field(gt=0)
    health_status: FleetHealthStatus
    latest_received_cycle: int | None = Field(default=None, gt=0)
    latest_scored_cycle: int | None = Field(default=None, gt=0)
    latest_event_timestamp: datetime | None = None
    latest_prediction: PredictionView | None = None
    open_alert_count: int = Field(default=0, ge=0)
    open_data_quality_issue_count: int = Field(default=0, ge=0)


class FleetHealthResponse(BaseModel):
    """Fleet-level state consumed by the dashboard overview."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    total_engines: int = Field(ge=0)
    current_engines: int = Field(ge=0)
    stale_engines: int = Field(ge=0)
    degraded_engines: int = Field(ge=0)
    withheld_engines: int = Field(ge=0)
    unavailable_engines: int = Field(ge=0)
    engines: list[FleetEngineSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "FleetHealthResponse":
        """Ensure the fleet summary counts describe the same fleet."""

        counts = (
            self.current_engines
            + self.stale_engines
            + self.degraded_engines
            + self.withheld_engines
            + self.unavailable_engines
        )
        if counts != self.total_engines:
            raise ValueError("fleet health status counts must equal total_engines")
        return self


class EquipmentDetailResponse(BaseModel):
    """Evidence and active operational state for one engine."""

    model_config = ConfigDict(extra="forbid")

    engine_id: int = Field(gt=0)
    health_status: FleetHealthStatus
    latest_received_cycle: int | None = Field(default=None, gt=0)
    latest_scored_cycle: int | None = Field(default=None, gt=0)
    latest_event_timestamp: datetime | None = None
    latest_prediction: PredictionView | None = None
    last_valid_prediction: PredictionView | None = None
    active_alerts: list[AlertView] = Field(default_factory=list)
    open_data_quality_issues: list[DataQualityIssueView] = Field(default_factory=list)


class PredictionHistoryResponse(BaseModel):
    """Immutable prediction history for one engine."""

    model_config = ConfigDict(extra="forbid")

    engine_id: int = Field(gt=0)
    items: list[PredictionView] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, min_length=1)


class AlertListResponse(BaseModel):
    """Paged alert query response."""

    model_config = ConfigDict(extra="forbid")

    items: list[AlertView] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, min_length=1)


class AlertTransitionRequest(BaseModel):
    """Optimistic-concurrency request for a human alert action."""

    model_config = ConfigDict(extra="forbid")

    action: AlertAction
    expected_version: int = Field(ge=1)
    actor_id: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=2_000)
