"""Exercise the staging API and worker boundary with one valid telemetry event."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime


def request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected an object from {url}")
    return value


def main() -> int:
    api_url = os.getenv("PM_SMOKE_API_URL", "http://127.0.0.1:8000").rstrip("/")
    metrics_url = os.getenv("PM_SMOKE_METRICS_URL", "http://127.0.0.1:9101/metrics").rstrip("/")
    engine_id = int(os.getenv("PM_SMOKE_ENGINE_ID", str(990000 + int(time.time()) % 1000)))

    deadline = time.monotonic() + 30
    while True:
        try:
            for path in ("/health", "/ready"):
                response = request_json(f"{api_url}{path}")
                if response.get("status") not in {"ok", "ready"}:
                    raise RuntimeError(f"{path} returned unexpected response: {response}")
            break
        except (OSError, urllib.error.HTTPError):
            if time.monotonic() >= deadline:
                raise RuntimeError("API did not become ready within 30 seconds") from None
            time.sleep(1)

    base_timestamp = datetime.now(UTC)
    for cycle in (1, 3, 5):
        event_id = f"staging-smoke-{engine_id}-cycle-{cycle}"
        event = {
            "event_id": event_id,
            "engine_id": engine_id,
            "cycle": cycle,
            "event_timestamp": base_timestamp.isoformat().replace("+00:00", "Z"),
            "schema_version": "telemetry-v1",
            "source_id": "staging-smoke-test",
            "measurements": {f"sensor_{index}": 1.0 for index in range(1, 22)},
        }
        accepted = request_json(f"{api_url}/v1/telemetry", event)
        if accepted.get("status") != "RECEIVED":
            raise RuntimeError(f"telemetry was not accepted: {accepted}")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            detail = request_json(f"{api_url}/v1/equipment/{engine_id}")
            if detail.get("latest_received_cycle") == 3:
                break
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        time.sleep(1)
    else:
        raise RuntimeError("telemetry was not visible through the equipment endpoint")

    with urllib.request.urlopen(metrics_url, timeout=5) as response:
        metrics = response.read().decode()
    if "pm_stream_events_total" not in metrics:
        raise RuntimeError("worker metrics did not expose pm_stream_events_total")

    print(f"staging smoke test passed for engine {engine_id} (cycles 1, 3, and 5)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"staging smoke test failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
