"""Tests for Phase 7 dashboard and operational response contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from predictive_maintenance.api.contracts import (
    AlertAction,
    AlertListResponse,
    AlertSeverity,
    AlertState,
    AlertTransitionRequest,
    AlertType,
    AlertView,
    FleetEngineSummary,
    FleetHealthResponse,
    FleetHealthStatus,
    PredictionHistoryResponse,
    PredictionView,
)
from predictive_maintenance.inference.contracts import DataQualityStatus, PredictionStatus


def prediction(*, status: PredictionStatus = PredictionStatus.AVAILABLE) -> PredictionView:
    return PredictionView(
        event_id="event-1",
        engine_id=1,
        cycle=10,
        event_timestamp=datetime(2026, 9, 24, 10, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 9, 24, 10, 1, tzinfo=UTC),
        generated_at=datetime(2026, 9, 24, 10, 2, tzinfo=UTC),
        estimated_rul=None if status is PredictionStatus.WITHHELD else 22.0,
        status=status,
        data_quality_status=(
            DataQualityStatus.INVALID
            if status is PredictionStatus.WITHHELD
            else DataQualityStatus.VALID
        ),
        model_release="rul-lightgbm-v1",
        feature_version="features-v1",
    )


def test_prediction_view_preserves_withheld_state() -> None:
    result = prediction(status=PredictionStatus.WITHHELD)

    assert result.estimated_rul is None
    assert result.data_quality_status is DataQualityStatus.INVALID


def test_prediction_view_rejects_rul_for_withheld_state() -> None:
    with pytest.raises(ValidationError, match="withheld predictions"):
        PredictionView(
            **{
                **prediction(status=PredictionStatus.WITHHELD).model_dump(),
                "estimated_rul": 4.0,
            }
        )


def test_fleet_health_requires_consistent_status_counts() -> None:
    engine = FleetEngineSummary(
        engine_id=1,
        health_status=FleetHealthStatus.CURRENT,
        latest_received_cycle=10,
        latest_scored_cycle=10,
        latest_prediction=prediction(),
    )
    response = FleetHealthResponse(
        generated_at=datetime(2026, 9, 24, tzinfo=UTC),
        total_engines=1,
        current_engines=1,
        stale_engines=0,
        degraded_engines=0,
        withheld_engines=0,
        unavailable_engines=0,
        engines=[engine],
    )

    assert response.engines[0].health_status is FleetHealthStatus.CURRENT

    with pytest.raises(ValidationError, match="must equal total_engines"):
        FleetHealthResponse(
            **{
                **response.model_dump(),
                "total_engines": 2,
            }
        )


def test_alert_transition_requires_expected_version_and_actor() -> None:
    request = AlertTransitionRequest(
        action=AlertAction.ACKNOWLEDGE,
        expected_version=3,
        actor_id="maintenance-user-1",
        reason="Inspection started.",
    )

    assert request.action is AlertAction.ACKNOWLEDGE
    assert request.expected_version == 3

    with pytest.raises(ValidationError):
        AlertTransitionRequest(
            action=AlertAction.RESOLVE,
            expected_version=0,
            actor_id="",
        )


def test_alert_view_requires_resolution_timestamp_for_closed_alert() -> None:
    with pytest.raises(ValidationError, match="resolved or dismissed"):
        AlertView(
            alert_id=1,
            engine_id=1,
            alert_type=AlertType.DATA_QUALITY,
            state=AlertState.RESOLVED,
            severity=AlertSeverity.WARNING,
            title="Cycle gap",
            message="Cycle 4 was not observed.",
            opened_at=datetime(2026, 9, 24, tzinfo=UTC),
            updated_at=datetime(2026, 9, 24, tzinfo=UTC),
        )


def test_history_and_alert_list_are_paged_contracts() -> None:
    history = PredictionHistoryResponse(engine_id=1, items=[prediction()], next_cursor="next")
    alerts = AlertListResponse(items=[], next_cursor=None)

    assert history.next_cursor == "next"
    assert alerts.items == []
