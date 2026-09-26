"""Tests for shared runtime configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from predictive_maintenance.core.settings import Environment, PlatformSettings


def test_settings_use_safe_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PM_ENVIRONMENT", raising=False)
    monkeypatch.delenv("PM_LOG_LEVEL", raising=False)
    monkeypatch.delenv("PM_DATA_ROOT", raising=False)
    monkeypatch.delenv("PM_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("PM_MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("PM_MLFLOW_REGISTERED_MODEL_NAME", raising=False)
    monkeypatch.delenv("PM_INFERENCE_MODEL_SOURCE", raising=False)
    monkeypatch.delenv("PM_MODEL_CACHE_ROOT", raising=False)
    monkeypatch.delenv("PM_DASHBOARD_STALE_AFTER_SECONDS", raising=False)

    settings = PlatformSettings(_env_file=None)

    assert settings.environment is Environment.LOCAL
    assert settings.log_level == "INFO"
    assert settings.data_root == Path("data")
    assert settings.mlflow_tracking_uri is None
    assert settings.mlflow_experiment_name == "cmapss-fd001-rul"
    assert settings.mlflow_registered_model_name == "cmapss-fd001-rul"
    assert settings.inference_model_source == "none"
    assert settings.model_cache_root == Path("artifacts/model-cache")
    assert settings.dashboard_stale_after_seconds == 3600


def test_settings_load_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PM_ENVIRONMENT", "test")
    monkeypatch.setenv("PM_LOG_LEVEL", "debug")
    monkeypatch.setenv("PM_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "PM_APPROVED_MODEL_MANIFEST_PATH",
        str(tmp_path / "model" / "manifest.json"),
    )
    monkeypatch.setenv("PM_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
    monkeypatch.setenv("PM_MLFLOW_EXPERIMENT_NAME", " fd001-rul-test ")
    monkeypatch.setenv("PM_MLFLOW_REGISTERED_MODEL_NAME", " fd001-registry-test ")
    monkeypatch.setenv("PM_INFERENCE_MODEL_SOURCE", "local_manifest")
    monkeypatch.setenv("PM_MODEL_CACHE_ROOT", str(tmp_path / "cache"))

    settings = PlatformSettings(_env_file=None)

    assert settings.environment is Environment.TEST
    assert settings.log_level == "DEBUG"
    assert settings.data_root == tmp_path
    assert settings.approved_model_manifest_path == tmp_path / "model" / "manifest.json"
    assert settings.mlflow_tracking_uri == "http://127.0.0.1:5000"
    assert settings.mlflow_experiment_name == "fd001-rul-test"
    assert settings.mlflow_registered_model_name == "fd001-registry-test"
    assert settings.inference_model_source == "local_manifest"
    assert settings.model_cache_root == tmp_path / "cache"


def test_settings_reject_unknown_environment() -> None:
    with pytest.raises(ValidationError, match="environment"):
        PlatformSettings(environment="customer-laptop", _env_file=None)


def test_settings_reject_unknown_log_level() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        PlatformSettings(log_level="verbose", _env_file=None)


def test_settings_reject_non_http_mlflow_tracking_uri() -> None:
    with pytest.raises(ValidationError, match="tracking-server URL"):
        PlatformSettings(mlflow_tracking_uri="sqlite:///mlflow.db", _env_file=None)


def test_settings_reject_blank_mlflow_experiment_name() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        PlatformSettings(mlflow_experiment_name="   ", _env_file=None)


def test_settings_reject_blank_mlflow_registered_model_name() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        PlatformSettings(mlflow_registered_model_name="   ", _env_file=None)


def test_settings_require_manifest_for_local_model_source() -> None:
    with pytest.raises(ValidationError, match="PM_APPROVED_MODEL_MANIFEST_PATH"):
        PlatformSettings(inference_model_source="local_manifest", _env_file=None)


def test_settings_require_tracking_uri_for_champion_model_source() -> None:
    with pytest.raises(ValidationError, match="PM_MLFLOW_TRACKING_URI"):
        PlatformSettings(inference_model_source="mlflow_champion", _env_file=None)


def test_settings_reject_unused_local_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="may only be set"):
        PlatformSettings(
            approved_model_manifest_path=tmp_path / "manifest.json",
            _env_file=None,
        )


def test_settings_require_critical_threshold_not_above_warning() -> None:
    with pytest.raises(ValidationError, match="CRITICAL_RUL_CYCLES"):
        PlatformSettings(
            equipment_warning_rul_cycles=10,
            equipment_critical_rul_cycles=20,
            _env_file=None,
        )
