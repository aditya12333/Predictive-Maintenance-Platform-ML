# Phase 4: Streaming Reliability and Data Quality

## Purpose

This phase establishes the reliable telemetry-consumer foundation used before model
inference. The platform accepts Kafka-compatible events, validates them, preserves
invalid payloads in a dead-letter topic, buffers events for event-time ordering, and
records data-quality warnings when an engine becomes idle.

## Implemented behavior

### Consumer outcomes

`TelemetryConsumer.consume_once()` returns a dedicated `ConsumerResult` value:

- `NO_MESSAGE`: polling timed out without receiving an event.
- `DEAD_LETTERED`: validation failed; the original payload and validation details were
  published to the dead-letter topic and the source offset was committed.
- `BUFFERED`: the event was valid but remains in the event-time reorder buffer.
- `PROCESSED`: one or more pending events were released and persisted.

The consumer commits a Kafka offset only after the event has been durably staged in
PostgreSQL. Invalid messages are committed only after successful dead-letter
publication.

### Durable pending storage

Valid events are first written to PostgreSQL's `pending_event` table. The payload is
stored as JSONB using Psycopg's `Jsonb` adapter. This protects events from process
restarts while the event-time ordering window is still open.

### Event-time ordering

The reorder buffer uses an allowed lateness of two cycles. For example, when cycle 3
is the newest observed cycle, the watermark is `3 - 2 = 1`; cycle 1 can be released,
while cycles 2 and 3 remain pending.

### Idle flush

If no new event arrives for an engine during the configured idle timeout, remaining
buffered events are flushed. The timeout is configurable through:

```dotenv
PM_REORDER_IDLE_FLUSH_SECONDS=300
```

Local testing used a one-minute or one-second override. Idle-flushed events are
persisted and receive an idempotent `ordering_window_expired` warning with severity
`WARNING`.

## Database and migration changes

- `pending_event.received_at` is loaded so idle duration can be calculated.
- Migration `26ef8994f2d0` adds a unique partial index preventing duplicate unresolved
  `ordering_window_expired` warnings.
- Warning inserts use `ON CONFLICT ... DO NOTHING`, making retries safe.

## Verification completed

Manual Redpanda/PostgreSQL verification confirmed:

1. Empty topic returns `no_message`.
2. Invalid JSON/schema payload returns `dead_lettered` and appears in the DLQ.
3. Valid cycle 1 is stored in `pending_event` and returns `buffered`.
4. A later cycle advances the watermark and releases cycle 1 as `processed`.
5. Idle flushing removes remaining pending events and creates an
   `ordering_window_expired` warning.

Automated checks:

- Ruff passes.
- Mypy passes.
- 30 tests pass.
- PostgreSQL-dependent integration tests are skipped when PostgreSQL is unavailable.

## Known boundary

The idle warning currently explains ordering-window expiration. Model inference and
prediction-quality decisions are intentionally handled in the next phase.
