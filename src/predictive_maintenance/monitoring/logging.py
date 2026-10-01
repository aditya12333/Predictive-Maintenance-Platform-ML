"""Structured logging and request correlation context."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

CORRELATION_ID_HEADER = "X-Correlation-ID"
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def new_correlation_id() -> str:
    """Create a compact identifier safe to return in an HTTP response header."""

    return uuid4().hex


def set_correlation_id(value: str | None = None) -> Token[str | None]:
    """Set the current context identifier and return its reset token."""

    return _correlation_id.set(value or new_correlation_id())


def reset_correlation_id(token: Token[str | None]) -> None:
    """Restore the correlation context that existed before a request or event."""

    _correlation_id.reset(token)


def get_correlation_id() -> str | None:
    """Return the identifier associated with the current execution context."""

    return _correlation_id.get()


class JsonLogFormatter(logging.Formatter):
    """Render log records as structured JSON for collection systems."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None) or get_correlation_id(),
        }
        for field in ("event_id", "engine_id", "cycle", "status_code", "duration_ms"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install one JSON stream handler on the root logger."""

    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(handler, "_predictive_maintenance", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    handler._predictive_maintenance = True  # type: ignore[attr-defined]
    root.addHandler(handler)
