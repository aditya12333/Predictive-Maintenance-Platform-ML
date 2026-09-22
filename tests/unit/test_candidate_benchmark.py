"""Tests for the repeatable candidate serving benchmark."""

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from predictive_maintenance.features.contracts import FEATURE_NAMES
from predictive_maintenance.inference.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ModelApprovalStatus,
    ModelArtifactManifest,
    RegressionMetrics,
    calculate_artifact_sha256,
)
from predictive_maintenance.training.benchmark import (
    ServingBenchmarkError,
    benchmark_candidate,
)


def _write_candidate(root: Path) -> Path:
    model = DummyRegressor(strategy="constant", constant=12.0)
    model.fit(np.zeros((2, len(FEATURE_NAMES))), np.asarray([10.0, 14.0]))
    model_path = root / "model.joblib"
    joblib.dump(model, model_path)
    evaluation_path = root / "evaluation.json"
    evaluation_path.write_text("{}\n", encoding="utf-8")
    metrics = RegressionMetrics(
        model_name="lightgbm",
        sample_count=2,
        mae_cycles=1.0,
        rmse_cycles=2.0,
        nasa_score=3.0,
    )
    manifest = ModelArtifactManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        model_release="rul-lightgbm-v1",
        model_name="lightgbm",
        source_training_run="comparison-v1",
        created_at=datetime(2026, 9, 22, tzinfo=UTC),
        feature_version="features-v1",
        feature_names=FEATURE_NAMES,
        training_dataset_version="nasa-cmapss-classic:v1:FD001",
        dataset_manifest_sha256="a" * 64,
        target_definition="max_cycle_for_engine - current_cycle",
        prediction_postprocessing="max(0, predicted_rul)",
        model_path=Path("model.joblib"),
        model_sha256=calculate_artifact_sha256(model_path),
        evaluation_report_path=Path("evaluation.json"),
        evaluation_report_sha256=calculate_artifact_sha256(evaluation_path),
        validation_metrics=metrics,
        official_test_metrics=metrics,
        status=ModelApprovalStatus.CANDIDATE,
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_benchmark_candidate_publishes_finite_measurements(tmp_path: Path) -> None:
    manifest_path = _write_candidate(tmp_path)
    output_path = tmp_path / "benchmark.json"

    report = benchmark_candidate(
        manifest_path=manifest_path,
        output_path=output_path,
        repetitions=10,
        warmup_repetitions=2,
    )

    assert report.repetitions == 10
    assert report.model_size_bytes > 0
    assert report.load_latency_ms >= 0
    assert report.prediction_p95_ms >= 0
    assert report.process_peak_rss_bytes > 0
    assert output_path.is_file()

    with pytest.raises(ServingBenchmarkError, match="already exists"):
        benchmark_candidate(
            manifest_path=manifest_path,
            output_path=output_path,
            repetitions=10,
        )
