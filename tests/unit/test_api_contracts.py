"""Tests for the telemetry ingestion request contract."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from predictive_maintenance.api.app import TelemetryRequest


def valid_payload() -> dict[str, object]:
    return {
        "event_id": "engine-1-cycle-1",
        "engine_id": 1,
        "cycle": 1,
        "event_timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "schema_version": "telemetry-v1",
        "source_id": "unit-test",
        "measurements": {"sensor_1": 1.0},
    }


def test_request_accepts_canonical_telemetry_schema() -> None:
    request = TelemetryRequest(**valid_payload())

    assert request.schema_version == "telemetry-v1"


def test_request_rejects_incompatible_telemetry_schema() -> None:
    payload = valid_payload()
    payload["schema_version"] = "telemetry.v1"

    with pytest.raises(ValidationError, match="telemetry-v1"):
        TelemetryRequest(**payload)
