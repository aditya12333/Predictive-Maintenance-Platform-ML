"""Prometheus-compatible application metrics."""

from prometheus_client import Counter, Gauge, Histogram, generate_latest

API_REQUESTS = Counter(
    "pm_api_requests_total",
    "HTTP requests completed by the API",
    ("method", "status"),
)
API_REQUEST_LATENCY = Histogram(
    "pm_api_request_duration_seconds",
    "HTTP request duration in seconds",
)
TELEMETRY_RECEIPTS = Counter(
    "pm_telemetry_receipts_total",
    "Telemetry persistence outcomes returned by the API",
    ("outcome",),
)
STREAM_EVENTS = Counter(
    "pm_stream_events_total",
    "Telemetry consumer outcomes",
    ("result",),
)
STREAM_RETRIES = Counter(
    "pm_stream_retries_total",
    "Retry attempts made by the telemetry consumer",
)
STREAM_PROCESSING_LATENCY = Histogram(
    "pm_stream_processing_duration_seconds",
    "Time spent persisting and releasing one telemetry event",
)
DEAD_LETTER_MESSAGES = Counter(
    "pm_dead_letter_messages_total",
    "Messages published to the dead-letter topic",
    ("error_type",),
)
PERSISTENCE_OUTCOMES = Counter(
    "pm_persistence_outcomes_total",
    "Transactional telemetry and prediction persistence outcomes",
    ("kind", "outcome"),
)
PREDICTION_OUTCOMES = Counter(
    "pm_prediction_outcomes_total",
    "Prediction states produced by inference",
    ("status",),
)
CONSUMER_LAG = Gauge(
    "pm_consumer_lag_records",
    "Latest observed consumer lag in records",
)


def render_metrics() -> bytes:
    """Render the process and application metrics in Prometheus text format."""

    return generate_latest()
