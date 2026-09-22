"""Repeatable local serving benchmark for an integrity-verified model package."""

import json
import resource
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from predictive_maintenance.inference.artifacts import load_model_manifest


class ServingBenchmarkError(RuntimeError):
    """Raised when a serving benchmark cannot produce trustworthy measurements."""


class _Predictor(Protocol):
    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class ServingBenchmark:
    """Measured model-load and single-record prediction characteristics."""

    schema_version: str
    model_release: str
    measured_at: str
    repetitions: int
    warmup_repetitions: int
    model_size_bytes: int
    load_latency_ms: float
    prediction_mean_ms: float
    prediction_p50_ms: float
    prediction_p95_ms: float
    prediction_p99_ms: float
    prediction_max_ms: float
    process_peak_rss_bytes: int


def benchmark_candidate(
    *,
    manifest_path: Path,
    output_path: Path,
    repetitions: int = 1_000,
    warmup_repetitions: int = 100,
) -> ServingBenchmark:
    """Measure one-record prediction latency and atomically publish the report."""

    if repetitions <= 0:
        raise ServingBenchmarkError("repetitions must be greater than zero")
    if warmup_repetitions < 0:
        raise ServingBenchmarkError("warmup_repetitions must not be negative")
    if output_path.exists():
        raise ServingBenchmarkError(f"immutable benchmark report already exists: {output_path}")

    manifest = load_model_manifest(manifest_path)
    input_values = np.zeros((1, len(manifest.feature_names)), dtype=np.float64)
    try:
        load_started = time.perf_counter_ns()
        model = cast(_Predictor, joblib.load(manifest.model_path))
        load_latency_ms = _elapsed_ms(load_started)

        for _ in range(warmup_repetitions):
            _predict_once(model, input_values)

        latencies_ms = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            prediction_started = time.perf_counter_ns()
            _predict_once(model, input_values)
            latencies_ms[index] = _elapsed_ms(prediction_started)
    except Exception as error:
        raise ServingBenchmarkError("candidate serving benchmark failed") from error

    report = ServingBenchmark(
        schema_version="serving-benchmark-v1",
        model_release=manifest.model_release,
        measured_at=datetime.now(UTC).isoformat(),
        repetitions=repetitions,
        warmup_repetitions=warmup_repetitions,
        model_size_bytes=manifest.model_path.stat().st_size,
        load_latency_ms=load_latency_ms,
        prediction_mean_ms=float(np.mean(latencies_ms)),
        prediction_p50_ms=float(np.percentile(latencies_ms, 50)),
        prediction_p95_ms=float(np.percentile(latencies_ms, 95)),
        prediction_p99_ms=float(np.percentile(latencies_ms, 99)),
        prediction_max_ms=float(np.max(latencies_ms)),
        process_peak_rss_bytes=_process_peak_rss_bytes(),
    )
    _publish_report(report, output_path=output_path)
    return report


def _predict_once(model: _Predictor, input_values: NDArray[np.float64]) -> None:
    predictions = np.maximum(0.0, np.asarray(model.predict(input_values), dtype=np.float64))
    if predictions.shape != (1,) or not np.isfinite(predictions).all():
        raise ServingBenchmarkError("candidate produced an invalid benchmark prediction")


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000


def _process_peak_rss_bytes() -> int:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak_rss if sys.platform == "darwin" else peak_rss * 1_024


def _publish_report(report: ServingBenchmark, *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(tempfile.mkdtemp(prefix=".benchmark-", dir=output_path.parent))
    try:
        staged_path = staging_directory / output_path.name
        staged_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
        staged_path.rename(output_path)
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)
