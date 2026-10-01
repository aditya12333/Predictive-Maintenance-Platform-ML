"""Tests for structured logs and request correlation context."""

import json
import logging
from datetime import UTC, datetime
from typing import cast

from fastapi.testclient import TestClient
from sqlalchemy import Engine

from predictive_maintenance.api.app import create_app
from predictive_maintenance.monitoring.logging import (
    JsonLogFormatter,
    get_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)
from predictive_maintenance.storage.database import EventPersistenceOutcome


class FakeSink:
    def persist(self, event):  # type: ignore[no-untyped-def]
        return EventPersistenceOutcome.INSERTED


def test_formatter_emits_json_context_fields() -> None:
    token = set_correlation_id("trace-123")
    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="event persisted",
            args=(),
            exc_info=None,
        )
        record.event_id = "event-1"  # type: ignore[attr-defined]
        output = json.loads(JsonLogFormatter().format(record))
    finally:
        reset_correlation_id(token)

    assert output["message"] == "event persisted"
    assert output["correlation_id"] == "trace-123"
    assert output["event_id"] == "event-1"


def test_api_returns_supplied_correlation_id() -> None:
    client = TestClient(
        create_app(engine=cast(Engine, object()), sink=FakeSink()),
    )
    response = client.get("/health", headers={"X-Correlation-ID": "recruiter-demo"})

    assert response.status_code == 200
    assert response.headers["x-correlation-id"] == "recruiter-demo"
    assert get_correlation_id() is None


def test_api_generates_correlation_id_for_telemetry() -> None:
    client = TestClient(
        create_app(engine=cast(Engine, object()), sink=FakeSink()),
    )
    response = client.post(
        "/v1/telemetry",
        json={
            "event_id": "observability-event-1",
            "engine_id": 1,
            "cycle": 1,
            "event_timestamp": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            "schema_version": "telemetry-v1",
            "source_id": "unit-test",
            "measurements": {"sensor_1": 1.0},
        },
    )

    assert response.status_code == 202
    assert len(response.headers["x-correlation-id"]) == 32
    assert get_correlation_id() is None


def test_metrics_endpoint_exposes_application_metrics() -> None:
    client = TestClient(
        create_app(engine=cast(Engine, object()), sink=FakeSink()),
    )
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "pm_api_requests_total" in response.text
    assert "pm_api_request_duration_seconds" in response.text
    assert "pm_prediction_outcomes_total" in response.text
