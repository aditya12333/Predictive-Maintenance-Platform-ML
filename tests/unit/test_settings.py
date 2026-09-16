"""Tests for shared runtime configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from predictive_maintenance.core.settings import Environment, PlatformSettings


def test_settings_use_safe_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PM_ENVIRONMENT", raising=False)
    monkeypatch.delenv("PM_LOG_LEVEL", raising=False)
    monkeypatch.delenv("PM_DATA_ROOT", raising=False)

    settings = PlatformSettings(_env_file=None)

    assert settings.environment is Environment.LOCAL
    assert settings.log_level == "INFO"
    assert settings.data_root == Path("data")


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

    settings = PlatformSettings(_env_file=None)

    assert settings.environment is Environment.TEST
    assert settings.log_level == "DEBUG"
    assert settings.data_root == tmp_path
    assert settings.approved_model_manifest_path == tmp_path / "model" / "manifest.json"


def test_settings_reject_unknown_environment() -> None:
    with pytest.raises(ValidationError, match="environment"):
        PlatformSettings(environment="customer-laptop", _env_file=None)


def test_settings_reject_unknown_log_level() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        PlatformSettings(log_level="verbose", _env_file=None)
