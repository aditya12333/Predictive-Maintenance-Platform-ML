"""Typed provenance and validation manifests for prepared datasets."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ImmutableModel(BaseModel):
    """Base model for immutable pipeline metadata."""

    model_config = ConfigDict(frozen=True)


class FileArtifact(ImmutableModel):
    """Integrity metadata for one file artifact."""

    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    crc32: str | None = None


class SourceManifest(ImmutableModel):
    """Provenance for one immutable source extraction."""

    dataset_id: str
    dataset_version: str
    source_url: str
    landing_page: str
    source_filename: str
    source_sha256: str = Field(min_length=64, max_length=64)
    inner_archive_sha256: str = Field(min_length=64, max_length=64)
    access_level: str
    license: str
    citation: str
    observed_at: datetime
    files: tuple[FileArtifact, ...]


class QualityIssue(ImmutableModel):
    """One validation failure or non-blocking data-quality warning."""

    rule_id: str
    severity: Literal["warning", "error"]
    scope: Literal["file", "record", "sequence", "relationship", "dataset"]
    message: str
    source_file: str | None = None
    line_number: int | None = Field(default=None, ge=1)
    engine_id: int | None = Field(default=None, ge=1)
    cycle: int | None = Field(default=None, ge=1)
    column_name: str | None = None
    observed_value: str | None = None


class RejectedRecord(ImmutableModel):
    """A raw source record that could not enter the canonical dataset."""

    source_file: str
    line_number: int = Field(ge=1)
    raw_record: str
    reason: str


class DatasetStatistics(ImmutableModel):
    """Basic structural statistics for a telemetry split."""

    split: Literal["train", "test"]
    row_count: int = Field(ge=0)
    engine_count: int = Field(ge=0)
    minimum_cycle: int = Field(ge=1)
    maximum_cycle: int = Field(ge=1)


class ValidationReport(ImmutableModel):
    """Complete validation outcome for one publication attempt."""

    dataset_id: str
    dataset_version: str
    subset_id: str
    schema_version: str
    status: Literal["PASSED", "PASSED_WITH_WARNINGS", "FAILED"]
    generated_at: datetime
    statistics: tuple[DatasetStatistics, ...]
    issues: tuple[QualityIssue, ...]


class DatasetManifest(ImmutableModel):
    """Integrity and lineage metadata for a published interim dataset."""

    dataset_id: str
    dataset_version: str
    subset_id: str
    telemetry_schema_version: str
    test_rul_schema_version: str
    source_manifest_sha256: str = Field(min_length=64, max_length=64)
    published_at: datetime
    validation_status: Literal["PASSED", "PASSED_WITH_WARNINGS"]
    files: tuple[FileArtifact, ...]


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(UTC)


def write_model_json(model: BaseModel, destination: Path) -> None:
    """Write a Pydantic model as stable, human-readable JSON."""

    destination.write_text(
        model.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
