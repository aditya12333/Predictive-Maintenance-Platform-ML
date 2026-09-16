"""Safe, idempotent extraction of the nested NASA C-MAPSS archive."""

import hashlib
import stat
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile, ZipInfo

from predictive_maintenance.data.contracts import (
    DATASET_ID,
    DATASET_VERSION,
    EXPECTED_INNER_MEMBERS,
)
from predictive_maintenance.data.download import NASA_DATASET_URL, calculate_sha256
from predictive_maintenance.data.manifest import (
    FileArtifact,
    SourceManifest,
    utc_now,
    write_model_json,
)

LANDING_PAGE = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
INNER_ARCHIVE_BASENAME = "CMAPSSData.zip"
SOURCE_MANIFEST_NAME = "source-manifest.json"
MAX_ARCHIVE_MEMBERS = 100
MAX_TOTAL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


class DatasetExtractionError(RuntimeError):
    """Raised when source extraction cannot be completed safely."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_unsafe_member(info: ZipInfo) -> bool:
    member_path = PurePosixPath(info.filename)
    unix_mode = info.external_attr >> 16
    return (
        member_path.is_absolute()
        or ".." in member_path.parts
        or stat.S_ISLNK(unix_mode)
    )


def _validate_archive_members(infos: list[ZipInfo], *, archive_name: str) -> None:
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise DatasetExtractionError(
            f"Archive {archive_name} contains too many members: {len(infos)}"
        )

    total_size = 0
    for info in infos:
        if _is_unsafe_member(info):
            raise DatasetExtractionError(
                f"Archive {archive_name} contains an unsafe member: {info.filename}"
            )
        if info.is_dir():
            continue
        if info.file_size > MAX_MEMBER_BYTES:
            raise DatasetExtractionError(
                f"Archive member exceeds the size limit: {info.filename}"
            )
        if info.file_size and not info.compress_size:
            raise DatasetExtractionError(
                f"Archive member has an invalid compressed size: {info.filename}"
            )
        if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            raise DatasetExtractionError(
                f"Archive member exceeds the compression-ratio limit: {info.filename}"
            )
        total_size += info.file_size

    if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise DatasetExtractionError(
            f"Archive {archive_name} exceeds the total extraction-size limit"
        )


def _validate_existing_extraction(
    destination_dir: Path,
    *,
    expected_source_sha256: str,
) -> SourceManifest:
    manifest_path = destination_dir / SOURCE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise DatasetExtractionError(
            f"Existing extraction has no source manifest: {destination_dir}"
        )

    manifest = SourceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.source_sha256 != expected_source_sha256:
        raise DatasetExtractionError(
            f"Existing extraction belongs to a different source: {destination_dir}"
        )

    for artifact in manifest.files:
        file_path = destination_dir / artifact.relative_path
        if not file_path.is_file():
            raise DatasetExtractionError(f"Extracted artifact is missing: {file_path}")
        if file_path.stat().st_size != artifact.size_bytes:
            raise DatasetExtractionError(f"Extracted artifact size changed: {file_path}")
        if calculate_sha256(file_path) != artifact.sha256:
            raise DatasetExtractionError(f"Extracted artifact checksum changed: {file_path}")

    return manifest


def extract_cmapss_archive(
    source_archive: Path,
    destination_dir: Path,
    *,
    expected_source_sha256: str,
) -> SourceManifest:
    """Verify and atomically extract the nested C-MAPSS source archive."""

    if not source_archive.is_file():
        raise DatasetExtractionError(f"Source archive does not exist: {source_archive}")

    source_sha256 = calculate_sha256(source_archive)
    if source_sha256 != expected_source_sha256:
        raise DatasetExtractionError(
            f"Source checksum mismatch. Expected {expected_source_sha256}, "
            f"received {source_sha256}."
        )

    if destination_dir.exists():
        return _validate_existing_extraction(
            destination_dir,
            expected_source_sha256=expected_source_sha256,
        )

    destination_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(source_archive) as outer_archive:
            outer_infos = outer_archive.infolist()
            _validate_archive_members(outer_infos, archive_name=source_archive.name)
            bad_member = outer_archive.testzip()
            if bad_member is not None:
                raise DatasetExtractionError(
                    f"Outer archive CRC validation failed for {bad_member}"
                )

            inner_candidates = [
                info
                for info in outer_infos
                if not info.is_dir()
                and PurePosixPath(info.filename).name == INNER_ARCHIVE_BASENAME
            ]
            if len(inner_candidates) != 1:
                raise DatasetExtractionError(
                    f"Expected exactly one {INNER_ARCHIVE_BASENAME}, "
                    f"found {len(inner_candidates)}"
                )

            inner_content = outer_archive.read(inner_candidates[0])

        with ZipFile(BytesIO(inner_content)) as inner_archive:
            inner_infos = inner_archive.infolist()
            _validate_archive_members(inner_infos, archive_name=INNER_ARCHIVE_BASENAME)
            bad_member = inner_archive.testzip()
            if bad_member is not None:
                raise DatasetExtractionError(
                    f"Inner archive CRC validation failed for {bad_member}"
                )

            regular_infos = [info for info in inner_infos if not info.is_dir()]
            member_names = tuple(sorted(info.filename for info in regular_infos))
            if member_names != tuple(sorted(EXPECTED_INNER_MEMBERS)):
                missing = sorted(set(EXPECTED_INNER_MEMBERS) - set(member_names))
                unexpected = sorted(set(member_names) - set(EXPECTED_INNER_MEMBERS))
                raise DatasetExtractionError(
                    f"Inner archive manifest differs from the contract. "
                    f"Missing={missing}, unexpected={unexpected}"
                )

            with TemporaryDirectory(
                dir=destination_dir.parent,
                prefix=f".{destination_dir.name}-",
            ) as temporary_directory:
                staged_dir = Path(temporary_directory) / destination_dir.name
                extracted_dir = staged_dir / "extracted"
                extracted_dir.mkdir(parents=True)
                artifacts: list[FileArtifact] = []

                for info in sorted(regular_infos, key=lambda item: item.filename):
                    content = inner_archive.read(info)
                    relative_path = Path("extracted") / info.filename
                    output_path = staged_dir / relative_path
                    output_path.write_bytes(content)
                    artifacts.append(
                        FileArtifact(
                            relative_path=relative_path.as_posix(),
                            size_bytes=len(content),
                            sha256=_sha256_bytes(content),
                            crc32=f"{info.CRC:08x}",
                        )
                    )

                manifest = SourceManifest(
                    dataset_id=DATASET_ID,
                    dataset_version=DATASET_VERSION,
                    source_url=NASA_DATASET_URL,
                    landing_page=LANDING_PAGE,
                    source_filename=source_archive.name,
                    source_sha256=source_sha256,
                    inner_archive_sha256=_sha256_bytes(inner_content),
                    access_level="public",
                    license="not specified",
                    citation=(
                        "A. Saxena and K. Goebel (2008). Turbofan Engine "
                        "Degradation Simulation Data Set, NASA Prognostics Data Repository."
                    ),
                    observed_at=utc_now(),
                    files=tuple(artifacts),
                )
                write_model_json(manifest, staged_dir / SOURCE_MANIFEST_NAME)
                staged_dir.replace(destination_dir)

    except BadZipFile as error:
        raise DatasetExtractionError(
            f"Invalid ZIP archive encountered while extracting {source_archive}"
        ) from error

    return manifest
