"""Prepare and atomically publish the validated C-MAPSS FD001 dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

from pydantic import BaseModel

from predictive_maintenance.data.contracts import (
    DATASET_ID,
    DATASET_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    TEST_RUL_SCHEMA_VERSION,
)
from predictive_maintenance.data.download import NASA_ARCHIVE_SHA256, calculate_sha256
from predictive_maintenance.data.extract import (
    SOURCE_MANIFEST_NAME,
    extract_cmapss_archive,
)
from predictive_maintenance.data.manifest import (
    DatasetManifest,
    FileArtifact,
    RejectedRecord,
    utc_now,
    write_model_json,
)
from predictive_maintenance.data.validate import (
    parse_telemetry_file,
    parse_test_rul_file,
    validate_fd001,
)

DEFAULT_SOURCE_ARCHIVE = Path("data/raw/cmapss/nasa_turbofan_outer.zip")
DEFAULT_RAW_VERSION_DIR = Path("data/raw/cmapss/v1")
DEFAULT_INTERIM_DIR = Path("data/interim/cmapss/v1/fd001")
DEFAULT_QUARANTINE_DIR = Path("data/quarantine/cmapss/v1/fd001")
DATASET_MANIFEST_NAME = "dataset-manifest.json"
VALIDATION_REPORT_NAME = "validation-report.json"


class DatasetPreparationError(RuntimeError):
    """Raised when a trusted interim dataset cannot be published."""


def _artifact(file_path: Path, *, relative_to: Path) -> FileArtifact:
    return FileArtifact(
        relative_path=file_path.relative_to(relative_to).as_posix(),
        size_bytes=file_path.stat().st_size,
        sha256=calculate_sha256(file_path),
    )


def _write_json_lines(records: tuple[BaseModel, ...], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(record.model_dump_json())
            output.write("\n")


def _publish_quarantine_report(
    quarantine_dir: Path,
    *,
    validation_report: BaseModel,
    rejected_records: tuple[RejectedRecord, ...],
) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    write_model_json(validation_report, quarantine_dir / VALIDATION_REPORT_NAME)
    if rejected_records:
        _write_json_lines(rejected_records, quarantine_dir / "rejected-records.jsonl")


def _validate_existing_publication(
    destination_dir: Path,
    *,
    source_manifest_sha256: str,
) -> DatasetManifest:
    manifest_path = destination_dir / DATASET_MANIFEST_NAME
    if not manifest_path.is_file():
        raise DatasetPreparationError(
            f"Existing interim publication has no manifest: {destination_dir}"
        )
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.source_manifest_sha256 != source_manifest_sha256:
        raise DatasetPreparationError(
            f"Existing interim publication belongs to a different source: {destination_dir}"
        )
    for artifact in manifest.files:
        artifact_path = destination_dir / artifact.relative_path
        if not artifact_path.is_file():
            raise DatasetPreparationError(f"Published artifact is missing: {artifact_path}")
        if artifact_path.stat().st_size != artifact.size_bytes:
            raise DatasetPreparationError(f"Published artifact size changed: {artifact_path}")
        if calculate_sha256(artifact_path) != artifact.sha256:
            raise DatasetPreparationError(
                f"Published artifact checksum changed: {artifact_path}"
            )
    return manifest


def prepare_fd001(
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    raw_version_dir: Path = DEFAULT_RAW_VERSION_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    quarantine_dir: Path = DEFAULT_QUARANTINE_DIR,
    *,
    expected_source_sha256: str = NASA_ARCHIVE_SHA256,
    allow_warnings: bool = False,
) -> DatasetManifest:
    """Extract, validate, and atomically publish canonical FD001 Parquet files."""

    extract_cmapss_archive(
        source_archive,
        raw_version_dir,
        expected_source_sha256=expected_source_sha256,
    )
    source_manifest_path = raw_version_dir / SOURCE_MANIFEST_NAME
    source_manifest_sha256 = calculate_sha256(source_manifest_path)

    if interim_dir.exists():
        return _validate_existing_publication(
            interim_dir,
            source_manifest_sha256=source_manifest_sha256,
        )

    extracted_dir = raw_version_dir / "extracted"
    train_result = parse_telemetry_file(
        extracted_dir / "train_FD001.txt",
        subset_id="FD001",
        split="train",
    )
    test_result = parse_telemetry_file(
        extracted_dir / "test_FD001.txt",
        subset_id="FD001",
        split="test",
    )
    rul_result = parse_test_rul_file(
        extracted_dir / "RUL_FD001.txt",
        subset_id="FD001",
        test_frame=test_result.frame,
    )
    report = validate_fd001(train_result, test_result, rul_result)
    rejected_records = (
        *train_result.rejected_records,
        *test_result.rejected_records,
        *rul_result.rejected_records,
    )

    publication_blocked = report.status == "FAILED" or (
        report.status == "PASSED_WITH_WARNINGS" and not allow_warnings
    )
    if publication_blocked:
        _publish_quarantine_report(
            quarantine_dir,
            validation_report=report,
            rejected_records=rejected_records,
        )
        raise DatasetPreparationError(
            f"FD001 publication blocked with validation status {report.status}. "
            f"Review {quarantine_dir / VALIDATION_REPORT_NAME}."
        )

    interim_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=interim_dir.parent,
        prefix=f".{interim_dir.name}-",
    ) as temporary_directory:
        staged_dir = Path(temporary_directory) / interim_dir.name
        staged_dir.mkdir()

        train_path = staged_dir / "train.parquet"
        test_path = staged_dir / "test.parquet"
        test_rul_path = staged_dir / "test-rul.parquet"
        report_path = staged_dir / VALIDATION_REPORT_NAME

        train_result.frame.write_parquet(train_path, compression="zstd", statistics=True)
        test_result.frame.write_parquet(test_path, compression="zstd", statistics=True)
        rul_result.frame.write_parquet(test_rul_path, compression="zstd", statistics=True)
        write_model_json(report, report_path)

        published_paths = (train_path, test_path, test_rul_path, report_path)
        artifacts = tuple(
            _artifact(path, relative_to=staged_dir) for path in published_paths
        )
        manifest = DatasetManifest(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            subset_id="FD001",
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            test_rul_schema_version=TEST_RUL_SCHEMA_VERSION,
            source_manifest_sha256=source_manifest_sha256,
            published_at=utc_now(),
            validation_status=cast(
                Literal["PASSED", "PASSED_WITH_WARNINGS"],
                report.status,
            ),
            files=artifacts,
        )
        write_model_json(manifest, staged_dir / DATASET_MANIFEST_NAME)
        staged_dir.replace(interim_dir)

    return manifest


def main() -> None:
    """Prepare the default local FD001 dataset."""

    manifest = prepare_fd001()
    manifest_digest = calculate_sha256(DEFAULT_INTERIM_DIR / DATASET_MANIFEST_NAME)
    print(f"Published FD001 dataset: {DEFAULT_INTERIM_DIR}")
    print(f"Validation status: {manifest.validation_status}")
    print(f"Manifest digest: {manifest_digest}")


if __name__ == "__main__":
    main()
