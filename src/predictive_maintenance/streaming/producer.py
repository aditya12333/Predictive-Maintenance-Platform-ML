"""Kafka-compatible telemetry publishing."""

import json
from typing import Any

from confluent_kafka import Producer

from predictive_maintenance.core.settings import PlatformSettings
from predictive_maintenance.storage.database import (
    EventPersistenceOutcome,
    TelemetryEventRecord,
)


class TelemetryProducer:
    """Publish telemetry to Redpanda or another Kafka-compatible broker."""

    def __init__(self, settings: PlatformSettings, topic: str | None = None) -> None:
        self._topic = topic or settings.kafka_topic
        self._dead_letter_topic = settings.kafka_dead_letter_topic
        self._producer = Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "acks": "all",
                "enable.idempotence": True,
            }
        )

    def publish(self, *, event_id: str, engine_id: int, payload: dict[str, Any]) -> None:
        """Publish one event and wait for broker acknowledgement."""

        delivery_error: list[Exception] = []

        def on_delivery(error: Any, _message: Any) -> None:
            if error is not None:
                delivery_error.append(RuntimeError(str(error)))

        self._producer.produce(
            self._topic,
            key=str(engine_id),
            value=json.dumps(payload, separators=(",", ":")),
            on_delivery=on_delivery,
        )
        undelivered = self._producer.flush()
        if undelivered:
            raise RuntimeError(f"{undelivered} telemetry message(s) were not delivered")
        if delivery_error:
            raise delivery_error[0]

    def publish_dead_letter(self, *, payload: str | None, metadata: dict[str, Any]) -> None:
        """Publish an invalid message and failure context to the DLQ topic."""

        dead_letter_payload = {"original_payload": payload, **metadata}
        delivery_error: list[Exception] = []

        def on_delivery(error: Any, _message: Any) -> None:
            if error is not None:
                delivery_error.append(RuntimeError(str(error)))

        self._producer.produce(
            self._dead_letter_topic,
            value=json.dumps(dead_letter_payload, separators=(",", ":")),
            on_delivery=on_delivery,
        )
        undelivered = self._producer.flush()
        if undelivered:
            raise RuntimeError(f"{undelivered} dead-letter message(s) were not delivered")
        if delivery_error:
            raise delivery_error[0]

    def close(self) -> None:
        """Flush pending messages before process shutdown."""

        self._producer.flush()


class RedpandaEventSink:
    """Durable-event-sink implementation backed by the Kafka API."""

    def __init__(self, settings: PlatformSettings) -> None:
        self._producer = TelemetryProducer(settings)

    def persist(self, event: TelemetryEventRecord) -> EventPersistenceOutcome:
        payload = event.payload or {
            "event_id": event.event_id,
            "engine_id": event.engine_id,
            "cycle": event.cycle,
            "event_timestamp": event.event_timestamp.isoformat(),
            "ingestion_timestamp": event.ingestion_timestamp.isoformat(),
            "schema_version": event.schema_version,
            "source_id": event.source_id,
            "payload_digest": event.payload_digest,
        }
        self._producer.publish(event_id=event.event_id, engine_id=event.engine_id, payload=payload)
        return EventPersistenceOutcome.INSERTED

    def close(self) -> None:
        """Flush pending broker messages before shutdown."""

        self._producer.close()
