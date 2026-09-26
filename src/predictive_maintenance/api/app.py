"""HTTP ingestion API for telemetry events."""

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Engine, text

from predictive_maintenance.api.contracts import (
    AlertListResponse,
    AlertState,
    AlertTransitionRequest,
    AlertView,
    EquipmentDetailResponse,
    FleetHealthResponse,
    PredictionHistoryResponse,
)
from predictive_maintenance.core.settings import load_settings
from predictive_maintenance.data.contracts import TELEMETRY_SCHEMA_VERSION
from predictive_maintenance.storage.alerts import (
    AlertNotFoundError,
    AlertVersionConflictError,
    InvalidAlertTransitionError,
    transition_alert,
)
from predictive_maintenance.storage.database import (
    DurableEventSink,
    EventPersistenceOutcome,
    PostgresEventSink,
    TelemetryEventRecord,
    create_database_engine,
)
from predictive_maintenance.storage.operational import (
    load_alerts,
    load_equipment_detail,
    load_fleet_health,
    load_prediction_history,
)
from predictive_maintenance.streaming.producer import RedpandaEventSink


class TelemetryRequest(BaseModel):
    """Validated telemetry envelope accepted from a gateway."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=200)
    engine_id: int = Field(gt=0)
    cycle: int = Field(gt=0)
    event_timestamp: datetime
    schema_version: str = Field(min_length=1, max_length=50)
    source_id: str = Field(min_length=1, max_length=200)
    measurements: dict[str, float | None] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def require_supported_schema(cls, value: str) -> str:
        if value != TELEMETRY_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {TELEMETRY_SCHEMA_VERSION}")
        return value

    @field_validator("event_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_timestamp must include a timezone offset")
        return value


class TelemetryReceipt(BaseModel):
    """Stable response returned after durable local persistence."""

    event_id: str
    status: str
    persistence_outcome: EventPersistenceOutcome


def _payload_digest(event: TelemetryRequest) -> str:
    canonical = json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def create_app(
    engine: Engine | None = None,
    sink: DurableEventSink | None = None,
) -> FastAPI:
    """Create the API with an injectable durable sink for testing and deployment."""

    app = FastAPI(title="AeroReliability Telemetry API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET", "PATCH", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    settings = load_settings()
    database_engine = engine or create_database_engine(settings)
    event_sink = sink
    if event_sink is None:
        event_sink = (
            RedpandaEventSink(settings)
            if settings.event_sink == "redpanda"
            else PostgresEventSink(database_engine)
        )

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        """Liveness probe: the API process is running."""

        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    def ready() -> dict[str, str]:
        """Readiness probe: the API can reach its operational database."""

        try:
            with database_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database is unavailable") from exc
        return {"status": "ready"}

    @app.get(
        "/v1/fleet/health",
        response_model=FleetHealthResponse,
        tags=["operations"],
    )
    def fleet_health() -> FleetHealthResponse:
        """Return the current operational state of every observed engine."""

        return load_fleet_health(
            database_engine,
            as_of=datetime.now(UTC),
            stale_after=timedelta(seconds=settings.dashboard_stale_after_seconds),
        )

    @app.get(
        "/v1/equipment/{engine_id}",
        response_model=EquipmentDetailResponse,
        tags=["operations"],
    )
    def equipment_detail(
        engine_id: int = Path(gt=0),
    ) -> EquipmentDetailResponse:
        """Return current state and operational evidence for one engine."""

        try:
            return load_equipment_detail(
                database_engine,
                engine_id=engine_id,
                as_of=datetime.now(UTC),
                stale_after=timedelta(seconds=settings.dashboard_stale_after_seconds),
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get(
        "/v1/equipment/{engine_id}/predictions",
        response_model=PredictionHistoryResponse,
        tags=["operations"],
    )
    def equipment_predictions(
        engine_id: int = Path(gt=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> PredictionHistoryResponse:
        """Return immutable RUL prediction history for one engine."""

        return load_prediction_history(database_engine, engine_id=engine_id, limit=limit)

    @app.get(
        "/v1/alerts",
        response_model=AlertListResponse,
        tags=["operations"],
    )
    def alerts(
        engine_id: int | None = Query(default=None, gt=0),
        state: Annotated[AlertState | None, Query()] = None,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> AlertListResponse:
        """Return active or historical alerts with optional filters."""

        return load_alerts(
            database_engine,
            engine_id=engine_id,
            state=state,
            limit=limit,
        )

    @app.patch(
        "/v1/alerts/{alert_id}",
        response_model=AlertView,
        tags=["operations"],
    )
    def update_alert(
        request: AlertTransitionRequest,
        alert_id: int = Path(gt=0),
    ) -> AlertView:
        """Apply one version-checked human alert action."""

        try:
            return transition_alert(
                database_engine,
                alert_id=alert_id,
                request=request,
            )
        except AlertNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except AlertVersionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except InvalidAlertTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/v1/telemetry",
        response_model=TelemetryReceipt,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["telemetry"],
    )
    def ingest_telemetry(event: TelemetryRequest, request: Request) -> TelemetryReceipt:
        del request  # Reserved for trace/request correlation in the next slice.
        ingestion_timestamp = datetime.now(UTC)
        outcome = event_sink.persist(
            TelemetryEventRecord(
                event_id=event.event_id,
                engine_id=event.engine_id,
                cycle=event.cycle,
                event_timestamp=event.event_timestamp,
                ingestion_timestamp=ingestion_timestamp,
                schema_version=event.schema_version,
                source_id=event.source_id,
                payload_digest=_payload_digest(event),
                payload=event.model_dump(mode="json"),
            ),
        )
        return TelemetryReceipt(
            event_id=event.event_id,
            status="RECEIVED",
            persistence_outcome=outcome,
        )

    return app


app = create_app()
