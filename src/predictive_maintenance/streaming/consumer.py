"""Kafka-compatible telemetry consumption with manual acknowledgements."""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from typing import TypeVar

from confluent_kafka import Consumer
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from predictive_maintenance.api.app import TelemetryRequest, _payload_digest
from predictive_maintenance.core.settings import PlatformSettings
from predictive_maintenance.inference.processor import TransactionalInferenceProcessor
from predictive_maintenance.storage.database import (
    TelemetryEventRecord,
    delete_pending_event,
    load_pending_events,
    persist_pending_event,
    persist_telemetry_event,
    record_ordering_window_expired_issue,
)
from predictive_maintenance.streaming.producer import TelemetryProducer
from predictive_maintenance.streaming.reorder import EventTimeReorderBuffer


class ConsumerResult(StrEnum):
    NO_MESSAGE = "no_message"
    PROCESSED = "processed"
    BUFFERED = "buffered"
    DEAD_LETTERED = "dead_lettered"


OperationResult = TypeVar("OperationResult")


class TelemetryConsumer:
    """Consume telemetry and commit offsets only after database persistence."""

    def __init__(
        self,
        settings: PlatformSettings,
        engine: Engine,
        *,
        event_processor: TransactionalInferenceProcessor | None = None,
    ) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": settings.kafka_consumer_group,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            }
        )
        self._consumer.subscribe([settings.kafka_topic])
        self._engine = engine
        self._settings = settings
        self._producer = TelemetryProducer(settings)
        self._event_processor = event_processor
        if self._event_processor is None:
            if settings.inference_model_source == "local_manifest":
                if settings.approved_model_manifest_path is None:
                    raise ValueError("local manifest model source has no manifest path")
                self._event_processor = TransactionalInferenceProcessor.load_approved(
                    engine=engine,
                    manifest_path=settings.approved_model_manifest_path,
                )
            elif settings.inference_model_source == "mlflow_champion":
                if settings.mlflow_tracking_uri is None:
                    raise ValueError("MLflow champion model source has no tracking URI")
                self._event_processor = TransactionalInferenceProcessor.load_champion(
                    engine=engine,
                    tracking_uri=settings.mlflow_tracking_uri,
                    registered_model_name=settings.mlflow_registered_model_name,
                    cache_root=settings.model_cache_root,
                )

    def _process_with_retry(
        self,
        operation: Callable[[], OperationResult],
    ) -> OperationResult:
        """Retry transient database failures with bounded exponential backoff."""

        attempts = 0
        while True:
            try:
                return operation()
            except (ConnectionError, OperationalError, TimeoutError):
                if attempts >= self._settings.max_retry_attempts:
                    raise
                delay = self._settings.retry_backoff_seconds * (2**attempts)
                attempts += 1
                time.sleep(delay)

    def _process_released_event(
        self,
        event: TelemetryEventRecord,
        *,
        quality_flags: tuple[str, ...] = (),
    ) -> object:
        """Route an ordered event through atomic inference when it is configured."""

        if self._event_processor is not None:
            return self._event_processor.process(event, quality_flags=quality_flags)
        return persist_telemetry_event(self._engine, event)

    def recover_pending(self) -> int:
        """Finish events left pending by a previous worker attempt."""

        recovered = 0
        reorder = EventTimeReorderBuffer[TelemetryEventRecord](allowed_lateness=2)
        latest_received_by_engine: dict[int, datetime] = {}
        for row in load_pending_events(self._engine):
            received_at = row["received_at"]
            if not isinstance(received_at, datetime):
                raise ValueError("pending event received_at must be a datetime")
            payload = row["payload"]
            if not isinstance(payload, dict):
                raise ValueError("pending telemetry payload must be a JSON object")
            event = TelemetryRequest.model_validate(payload)
            latest_received_by_engine[event.engine_id] = max(
                latest_received_by_engine.get(event.engine_id, received_at),
                received_at,
            )
            record = TelemetryEventRecord(
                event_id=event.event_id,
                engine_id=event.engine_id,
                cycle=event.cycle,
                event_timestamp=event.event_timestamp,
                ingestion_timestamp=datetime.now(UTC),
                schema_version=event.schema_version,
                source_id=event.source_id,
                payload_digest=_payload_digest(event),
                payload=payload,
            )
            result = reorder.observe(event.engine_id, event.cycle, record)
            for released in result.released:
                released_event = released
                self._process_with_retry(partial(self._process_released_event, released_event))
                delete_pending_event(self._engine, released_event.event_id)
                recovered += 1

        now = datetime.now(UTC)
        for engine_id, latest_received_at in latest_received_by_engine.items():
            idle_seconds = (now - latest_received_at).total_seconds()
            if idle_seconds < self._settings.reorder_idle_flush_seconds:
                continue
            for released_event in reorder.flush(engine_id):
                self._process_with_retry(
                    partial(
                        self._process_released_event,
                        released_event,
                        quality_flags=("ordering_window_expired",),
                    )
                )
                record_ordering_window_expired_issue(
                    self._engine,
                    released_event,
                    idle_seconds,
                )
                delete_pending_event(self._engine, released_event.event_id)
                recovered += 1
        return recovered

    def consume_once(self, timeout: float = 5.0) -> ConsumerResult:
        """Process one event; return None when no message arrives before timeout."""

        recovered_before_poll = self.recover_pending()
        message = self._consumer.poll(timeout)
        if message is None:
            if recovered_before_poll:
                return ConsumerResult.PROCESSED
            return ConsumerResult.NO_MESSAGE
        if message.error() is not None:
            raise RuntimeError(str(message.error()))

        raw_value = message.value()
        try:
            if raw_value is None:
                raise ValueError("telemetry message has no value")
            payload = json.loads(raw_value)
            event = TelemetryRequest.model_validate(payload)
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            self._producer.publish_dead_letter(
                payload=raw_value.decode("utf-8", errors="replace") if raw_value else None,
                metadata={
                    "source_topic": message.topic(),
                    "partition": message.partition(),
                    "offset": message.offset(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            self._consumer.commit(message=message, asynchronous=False)
            return ConsumerResult.DEAD_LETTERED
        record = TelemetryEventRecord(
            event_id=event.event_id,
            engine_id=event.engine_id,
            cycle=event.cycle,
            event_timestamp=event.event_timestamp,
            ingestion_timestamp=datetime.now(UTC),
            schema_version=event.schema_version,
            source_id=event.source_id,
            payload_digest=_payload_digest(event),
            payload=payload,
        )
        self._process_with_retry(lambda: persist_pending_event(self._engine, record))
        self._consumer.commit(message=message, asynchronous=False)
        recovered_after_poll = self.recover_pending()
        if recovered_before_poll or recovered_after_poll:
            return ConsumerResult.PROCESSED

        return ConsumerResult.BUFFERED

    def close(self) -> None:
        """Leave the consumer group cleanly."""

        self._consumer.close()
        self._producer.close()

    def run(self, max_messages: int | None = None) -> int:
        """Consume continuously, optionally stopping after a message count."""

        processed = 0
        while max_messages is None or processed < max_messages:
            self.consume_once()
            processed += 1
        return processed
