"""Unit tests for consumer outcome reporting."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy import Engine

import predictive_maintenance.streaming.consumer as consumer_module
from predictive_maintenance.core.settings import PlatformSettings
from predictive_maintenance.streaming.consumer import ConsumerResult, TelemetryConsumer


class FakeMessage:
    def __init__(self, value: bytes | None, *, offset: int = 0) -> None:
        self._value = value
        self._offset = offset

    def value(self) -> bytes | None:
        return self._value

    def error(self) -> None:
        return None

    def topic(self) -> str:
        return "telemetry.test"

    def partition(self) -> int:
        return 0

    def offset(self) -> int:
        return self._offset


class FakeKafkaConsumer:
    def __init__(self, message: FakeMessage | None) -> None:
        self.message = message
        self.commits: list[FakeMessage] = []

    def poll(self, timeout: float) -> FakeMessage | None:
        return self.message

    def commit(self, *, message: FakeMessage, asynchronous: bool) -> None:
        self.commits.append(message)


class FakeProducer:
    def __init__(self) -> None:
        self.dead_letters: list[tuple[str | None, dict[str, object]]] = []

    def publish_dead_letter(
        self,
        *,
        payload: str | None,
        metadata: dict[str, object],
    ) -> None:
        self.dead_letters.append((payload, metadata))


def build_consumer(
    message: FakeMessage | None,
    recover_pending: list[int],
) -> tuple[TelemetryConsumer, FakeKafkaConsumer, FakeProducer]:
    consumer = object.__new__(TelemetryConsumer)
    kafka_consumer = FakeKafkaConsumer(message)
    producer = FakeProducer()
    consumer._consumer = kafka_consumer
    consumer._producer = producer
    consumer._engine = object()
    consumer._settings = SimpleNamespace(
        max_retry_attempts=0,
        retry_backoff_seconds=1.0,
    )
    consumer._event_processor = None
    consumer.recover_pending = Mock(side_effect=recover_pending)
    return consumer, kafka_consumer, producer


def valid_payload() -> bytes:
    return (
        b'{"event_id":"unit-event-1","engine_id":1,"cycle":1,'
        b'"event_timestamp":"2026-01-01T00:00:01Z",'
        b'"schema_version":"telemetry-v1","source_id":"unit-test",'
        b'"measurements":{"sensor_1":0.5}}'
    )


def test_consume_once_returns_no_message() -> None:
    consumer, _, _ = build_consumer(None, [0])

    assert consumer.consume_once(timeout=0.01) == ConsumerResult.NO_MESSAGE


def test_consume_once_dead_letters_invalid_message() -> None:
    message = FakeMessage(b'{"engine_id":-1}')
    consumer, kafka_consumer, producer = build_consumer(message, [0])

    result = consumer.consume_once(timeout=0.01)

    assert result == ConsumerResult.DEAD_LETTERED
    assert producer.dead_letters[0][0] == '{"engine_id":-1}'
    assert kafka_consumer.commits == [message]


@pytest.mark.parametrize(
    ("recovered", "expected"),
    [(0, ConsumerResult.BUFFERED), (1, ConsumerResult.PROCESSED)],
)
def test_consume_once_reports_processing_state(
    monkeypatch: pytest.MonkeyPatch,
    recovered: int,
    expected: ConsumerResult,
) -> None:
    message = FakeMessage(valid_payload())
    consumer, kafka_consumer, _ = build_consumer(message, [0, recovered])
    monkeypatch.setattr(consumer_module, "persist_pending_event", lambda engine, event: None)

    assert consumer.consume_once(timeout=0.01) == expected
    assert kafka_consumer.commits == [message]


def test_released_event_uses_configured_transactional_processor() -> None:
    consumer, _, _ = build_consumer(None, [0])
    event_processor = Mock()
    event_processor.process.return_value = object()
    consumer._event_processor = event_processor
    record = object()

    result = consumer._process_released_event(
        record,
        quality_flags=("ordering_window_expired",),
    )

    assert result is event_processor.process.return_value
    event_processor.process.assert_called_once_with(
        record,
        quality_flags=("ordering_window_expired",),
    )


def test_constructor_loads_explicitly_configured_approved_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kafka_consumer = Mock()
    producer = Mock()
    event_processor = Mock()
    loader = Mock(return_value=event_processor)
    engine = cast(Engine, object())
    manifest_path = Path("artifacts/approved/manifest.json")
    settings = PlatformSettings(
        inference_model_source="local_manifest",
        approved_model_manifest_path=manifest_path,
        _env_file=None,
    )
    monkeypatch.setattr(consumer_module, "Consumer", Mock(return_value=kafka_consumer))
    monkeypatch.setattr(consumer_module, "TelemetryProducer", Mock(return_value=producer))
    monkeypatch.setattr(
        consumer_module.TransactionalInferenceProcessor,
        "load_approved",
        loader,
    )

    consumer = TelemetryConsumer(settings, engine)

    assert consumer._event_processor is event_processor
    loader.assert_called_once_with(
        engine=engine,
        manifest_path=manifest_path,
        alert_policy=consumer_module.AlertPolicy(),
    )


def test_constructor_loads_mlflow_champion_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kafka_consumer = Mock()
    producer = Mock()
    event_processor = Mock()
    loader = Mock(return_value=event_processor)
    engine = cast(Engine, object())
    settings = PlatformSettings(
        inference_model_source="mlflow_champion",
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_registered_model_name="fd001-rul",
        model_cache_root=tmp_path / "model-cache",
        _env_file=None,
    )
    monkeypatch.setattr(consumer_module, "Consumer", Mock(return_value=kafka_consumer))
    monkeypatch.setattr(consumer_module, "TelemetryProducer", Mock(return_value=producer))
    monkeypatch.setattr(
        consumer_module.TransactionalInferenceProcessor,
        "load_champion",
        loader,
    )

    consumer = TelemetryConsumer(settings, engine)

    assert consumer._event_processor is event_processor
    loader.assert_called_once_with(
        engine=engine,
        tracking_uri="http://127.0.0.1:5000",
        registered_model_name="fd001-rul",
        cache_root=tmp_path / "model-cache",
        alert_policy=consumer_module.AlertPolicy(),
    )
