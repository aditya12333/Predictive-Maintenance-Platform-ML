"""HTTP ingestion API for telemetry events."""

import json
from datetime import UTC, datetime
from hashlib import sha256

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Engine, text

from predictive_maintenance.core.settings import load_settings
from predictive_maintenance.data.contracts import TELEMETRY_SCHEMA_VERSION
from predictive_maintenance.storage.database import (
    DurableEventSink,
    EventPersistenceOutcome,
    PostgresEventSink,
    TelemetryEventRecord,
    create_database_engine,
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
