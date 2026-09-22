"""Validated runtime configuration shared by project entry points."""

from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Supported deployment environments."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
EventSink = Literal["postgres", "redpanda"]
InferenceModelSource = Literal["none", "local_manifest", "mlflow_champion"]


class PlatformSettings(BaseSettings):
    """Settings that are safe and meaningful for every project process.

    Service-specific settings, such as a database URL, will live in the service
    that requires them. This prevents an offline data command from requiring
    unrelated infrastructure credentials.
    """

    model_config = SettingsConfigDict(
        env_prefix="PM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    log_level: LogLevel = "INFO"
    data_root: Path = Path("data")

    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://pm_app:pm_local_only@localhost:5432/predictive_maintenance"
    )
    kafka_bootstrap_servers: str = "localhost:19092"
    kafka_topic: str = "telemetry.events"
    kafka_consumer_group: str = "telemetry-processor"
    event_sink: EventSink = "postgres"
    inference_model_source: InferenceModelSource = "none"
    approved_model_manifest_path: Path | None = None
    model_cache_root: Path = Path("artifacts/model-cache")
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = Field(default="cmapss-fd001-rul", min_length=1)
    mlflow_registered_model_name: str = Field(default="cmapss-fd001-rul", min_length=1)
    max_retry_attempts: int = Field(default=3, ge=0, description="Maximum number of retry attempts")
    retry_backoff_seconds: float = Field(
        default=1.0, gt=0, description="Delay before retrying a failed operation"
    )
    kafka_dead_letter_topic: str = Field(
        default="telemetry.events.dlq",
        min_length=1,
        description="Topic for messages that cannot be processed after retries",
    )
    reorder_idle_flush_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Idle time before pending events are released from the reorder buffer",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Accept conventional case-insensitive log-level values."""

        return value.upper() if isinstance(value, str) else value

    @field_validator("mlflow_tracking_uri")
    @classmethod
    def validate_mlflow_tracking_uri(cls, value: str | None) -> str | None:
        """Require an explicit HTTP endpoint when MLflow tracking is enabled."""

        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an HTTP or HTTPS tracking-server URL")
        return value.rstrip("/")

    @field_validator("mlflow_experiment_name", "mlflow_registered_model_name")
    @classmethod
    def validate_mlflow_name(cls, value: str) -> str:
        """Reject an MLflow resource name that contains only whitespace."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_inference_model_source(self) -> "PlatformSettings":
        """Require one complete and unambiguous inference model configuration."""

        if self.inference_model_source == "local_manifest":
            if self.approved_model_manifest_path is None:
                raise ValueError(
                    "PM_APPROVED_MODEL_MANIFEST_PATH is required when "
                    "PM_INFERENCE_MODEL_SOURCE=local_manifest"
                )
        elif self.approved_model_manifest_path is not None:
            raise ValueError(
                "PM_APPROVED_MODEL_MANIFEST_PATH may only be set when "
                "PM_INFERENCE_MODEL_SOURCE=local_manifest"
            )

        if (
            self.inference_model_source == "mlflow_champion"
            and self.mlflow_tracking_uri is None
        ):
            raise ValueError(
                "PM_MLFLOW_TRACKING_URI is required when "
                "PM_INFERENCE_MODEL_SOURCE=mlflow_champion"
            )
        return self


def load_settings() -> PlatformSettings:
    """Load and validate the platform settings for the current process."""

    return PlatformSettings()
