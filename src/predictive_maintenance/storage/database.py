"""Database engine construction and idempotent telemetry persistence."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from psycopg.types.json import Jsonb
from sqlalchemy import Connection, Engine, create_engine, text

from predictive_maintenance.core.settings import PlatformSettings


def create_database_engine(settings: PlatformSettings) -> Engine:
    """Create an engine without opening a connection until it is first used."""

    return create_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
    )


class EventPersistenceOutcome(StrEnum):
    """Result of an idempotent telemetry insert."""

    INSERTED = "inserted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class TelemetryPersistenceResult:
    """Telemetry outcome and quality context needed by inference."""

    outcome: EventPersistenceOutcome
    is_degraded: bool
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class TelemetryEventRecord:
    """Validated event metadata required by the operational store."""

    event_id: str
    engine_id: int
    cycle: int
    event_timestamp: datetime
    ingestion_timestamp: datetime
    schema_version: str
    source_id: str
    payload_digest: str
    payload: dict[str, Any] | None = None


class DurableEventSink(Protocol):
    """Contract used by the API to durably accept an event."""

    def persist(self, event: TelemetryEventRecord) -> EventPersistenceOutcome:
        """Persist an event and return its idempotency outcome."""


def persist_telemetry_event(engine: Engine, event: TelemetryEventRecord) -> EventPersistenceOutcome:
    """Insert an event, treating exact retries as safe and conflicts as explicit."""

    with engine.begin() as connection:
        return persist_telemetry_event_transaction(connection, event).outcome


def persist_telemetry_event_transaction(
    connection: Connection,
    event: TelemetryEventRecord,
    *,
    quality_flags: tuple[str, ...] = (),
) -> TelemetryPersistenceResult:
    """Persist telemetry using the caller's transaction and return inference quality."""

    insert = text(
        """
        INSERT INTO telemetry_event (
            event_id, engine_id, cycle, event_timestamp, ingestion_timestamp,
            schema_version, source_id, payload_digest, quality_status
        ) VALUES (
            :event_id, :engine_id, :cycle, :event_timestamp, :ingestion_timestamp,
            :schema_version, :source_id, :payload_digest, :quality_status
        )
        ON CONFLICT (event_id) DO NOTHING
        RETURNING event_id
        """
    )
    lookup = text(
        "SELECT payload_digest, quality_status FROM telemetry_event WHERE event_id = :event_id"
    )
    quality_issue_lookup = text(
        """
        SELECT issue_type
        FROM data_quality_issue
        WHERE event_id = :event_id
          AND resolved_at IS NULL
          AND issue_type IN ('cycle_gap', 'ordering_window_expired')
        ORDER BY issue_type
        """
    )
    latest_cycle = text("SELECT max(cycle) FROM telemetry_event WHERE engine_id = :engine_id")
    gap_issue = text(
        """
        INSERT INTO data_quality_issue (
            event_id, engine_id, issue_type, severity, message, fingerprint
        ) VALUES (
            :event_id, :engine_id, 'cycle_gap', 'WARNING', :message, :fingerprint
        )
        """
    )
    conflict_issue = text(
        """
        INSERT INTO data_quality_issue (
            event_id, engine_id, issue_type, severity, message, fingerprint
        ) VALUES (
            :event_id, :engine_id, 'conflicting_duplicate', 'ERROR', :message, :fingerprint
        )
        ON CONFLICT (event_id, issue_type, fingerprint) WHERE
            issue_type = 'conflicting_duplicate' AND resolved_at IS NULL
        DO NOTHING
        """
    )

    previous_cycle = connection.execute(latest_cycle, {"engine_id": event.engine_id}).scalar_one()
    has_gap = previous_cycle is not None and event.cycle > previous_cycle + 1
    effective_flags = list(dict.fromkeys(quality_flags))
    if has_gap and "cycle_gap" not in effective_flags:
        effective_flags.append("cycle_gap")
    is_degraded = bool(effective_flags)
    inserted_id = connection.execute(
        insert,
        {
            "event_id": event.event_id,
            "engine_id": event.engine_id,
            "cycle": event.cycle,
            "event_timestamp": event.event_timestamp,
            "ingestion_timestamp": event.ingestion_timestamp,
            "schema_version": event.schema_version,
            "source_id": event.source_id,
            "payload_digest": event.payload_digest,
            "quality_status": "DEGRADED" if is_degraded else "VALID",
        },
    ).scalar_one_or_none()

    if inserted_id is not None:
        if has_gap:
            if previous_cycle is None:  # pragma: no cover - guarded by has_gap
                raise RuntimeError("cycle-gap calculation lost its previous cycle")
            missing_start = previous_cycle + 1
            missing_end = event.cycle - 1
            connection.execute(
                gap_issue,
                {
                    "event_id": event.event_id,
                    "engine_id": event.engine_id,
                    "message": (
                        f"Cycles {missing_start} through {missing_end} were not observed "
                        "before this event"
                    ),
                    "fingerprint": f"{missing_start}:{missing_end}",
                },
            )
        return TelemetryPersistenceResult(
            outcome=EventPersistenceOutcome.INSERTED,
            is_degraded=is_degraded,
            quality_flags=tuple(effective_flags),
        )

    existing = connection.execute(lookup, {"event_id": event.event_id}).mappings().one()
    if existing["payload_digest"] == event.payload_digest:
        stored_is_degraded = str(existing["quality_status"]).upper() == "DEGRADED"
        stored_flags = [
            str(issue_type)
            for issue_type in connection.execute(
                quality_issue_lookup, {"event_id": event.event_id}
            ).scalars()
        ]
        for flag in effective_flags:
            if flag not in stored_flags:
                stored_flags.append(flag)
        return TelemetryPersistenceResult(
            outcome=EventPersistenceOutcome.DUPLICATE,
            is_degraded=stored_is_degraded or bool(stored_flags),
            quality_flags=tuple(stored_flags),
        )
    connection.execute(
        conflict_issue,
        {
            "event_id": event.event_id,
            "engine_id": event.engine_id,
            "message": (
                "Repeated event_id arrived with a different payload digest; "
                "original event preserved for investigation"
            ),
            "fingerprint": event.payload_digest,
        },
    )
    return TelemetryPersistenceResult(
        outcome=EventPersistenceOutcome.CONFLICT,
        is_degraded=True,
        quality_flags=("conflicting_duplicate",),
    )


def record_ordering_window_expired_issue(
    engine: Engine,
    event: TelemetryEventRecord,
    idle_seconds: float,
) -> None:
    """Record an idempotent warning for an event released after idle flushing."""

    statement = text(
        """
        INSERT INTO data_quality_issue (
            event_id, engine_id, issue_type, severity, message
        ) VALUES (
            :event_id,
            :engine_id,
            'ordering_window_expired',
            'WARNING',
            :message
        )
        ON CONFLICT (event_id, issue_type)
        WHERE issue_type = 'ordering_window_expired'
          AND resolved_at IS NULL
        DO NOTHING
        """
    )

    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "event_id": event.event_id,
                "engine_id": event.engine_id,
                "message": (
                    "Event was released after the event-time ordering window expired "
                    f"following {idle_seconds:.1f} seconds of engine inactivity"
                ),
            },
        )


def persist_pending_event(engine: Engine, event: TelemetryEventRecord) -> None:
    """Durably stage an event before event-time reordering."""

    statement = text(
        """
        INSERT INTO pending_event (
            event_id, engine_id, cycle, event_timestamp, payload_digest, payload
        ) VALUES (
            :event_id, :engine_id, :cycle, :event_timestamp, :payload_digest, :payload
        )
        ON CONFLICT (event_id) DO NOTHING
        """
    )
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "event_id": event.event_id,
                "engine_id": event.engine_id,
                "cycle": event.cycle,
                "event_timestamp": event.event_timestamp,
                "payload_digest": event.payload_digest,
                "payload": Jsonb(event.payload or {}),
            },
        )


def delete_pending_event(engine: Engine, event_id: str) -> None:
    """Remove a pending event after final processing has committed."""

    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM pending_event WHERE event_id = :event_id"),
            {"event_id": event_id},
        )


def load_pending_events(engine: Engine) -> list[dict[str, object]]:
    """Load durable pending payloads for worker restart recovery."""

    statement = text(
        """
        SELECT event_id, engine_id, cycle, received_at, payload
        FROM pending_event
        ORDER BY engine_id, cycle
        """
    )
    with engine.connect() as connection:
        return [dict(row._mapping) for row in connection.execute(statement)]


class PostgresEventSink:
    """PostgreSQL implementation of the durable event-sink contract."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def persist(self, event: TelemetryEventRecord) -> EventPersistenceOutcome:
        return persist_telemetry_event(self._engine, event)
