"""Resolve an approved MLflow champion into an integrity-checked local runtime."""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

from predictive_maintenance.inference.artifacts import (
    ArtifactLoadError,
    ModelApprovalStatus,
    ModelArtifactManifest,
    load_model_manifest,
)
from predictive_maintenance.inference.sklearn_predictor import SklearnRULPredictor
from predictive_maintenance.training.registry import (
    APPROVAL_STATUS_TAG,
    immutable_version_tags,
)


class ChampionResolutionError(RuntimeError):
    """Raised when the current champion cannot be trusted or loaded."""


@dataclass(frozen=True)
class ResolvedChampion:
    """A registry champion whose lifecycle evidence and local package are verified."""

    registered_model_name: str
    model_version: str
    model_release: str
    source_run_id: str
    cache_directory: Path
    manifest: ModelArtifactManifest
    predictor: SklearnRULPredictor


def resolve_champion(
    *,
    tracking_uri: str,
    registered_model_name: str,
    cache_root: Path,
) -> ResolvedChampion:
    """Resolve the live alias, verify its audit chain, and load its cached package."""

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version_by_alias(registered_model_name, "champion")
        model_version = str(version.version)
        tags = dict(version.tags or {})
        source_run_id = version.run_id
        if not source_run_id:
            raise ChampionResolutionError("champion version has no registry source run")
        if tags.get(APPROVAL_STATUS_TAG) != ModelApprovalStatus.APPROVED.value:
            raise ChampionResolutionError("champion version is not approved")

        _verify_decision_chain(
            client,
            registered_model_name=registered_model_name,
            model_version=model_version,
            tags=tags,
        )
        source_run = client.get_run(source_run_id)
        if source_run.info.status != "FINISHED":
            raise ChampionResolutionError("champion registry source run is not finished")

        cache_directory = _cache_directory(
            cache_root,
            registered_model_name=registered_model_name,
            model_version=model_version,
        )
        if not cache_directory.exists():
            _download_package(
                client,
                source_run_id=source_run_id,
                cache_directory=cache_directory,
            )
        manifest = _verify_cached_package(cache_directory, tags=tags)
        predictor = SklearnRULPredictor.from_verified_manifest(manifest)
        return ResolvedChampion(
            registered_model_name=registered_model_name,
            model_version=model_version,
            model_release=manifest.model_release,
            source_run_id=source_run_id,
            cache_directory=cache_directory,
            manifest=manifest,
            predictor=predictor,
        )
    except ChampionResolutionError:
        raise
    except Exception as error:
        raise ChampionResolutionError(
            f"could not resolve champion from MLflow at {tracking_uri}"
        ) from error


def _verify_decision_chain(
    client: MlflowClient,
    *,
    registered_model_name: str,
    model_version: str,
    tags: dict[str, str],
) -> None:
    promotion_run_id = tags.get("promotion_decision_run_id")
    promotion_sha256 = tags.get("promotion_record_sha256")
    approval_run_id = tags.get("approval_decision_run_id")
    approval_sha256 = tags.get("approval_record_sha256")
    if not promotion_run_id or not promotion_sha256:
        raise ChampionResolutionError("champion is missing promotion decision linkage")
    if not approval_run_id or not approval_sha256:
        raise ChampionResolutionError("champion is missing approval decision linkage")

    promotion = _download_verified_json(
        client,
        run_id=promotion_run_id,
        artifact_path="decision/promotion-decision.json",
        expected_sha256=promotion_sha256,
        decision_name="promotion",
    )
    expected_promotion_fields = {
        "decision": "promoted",
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "new_champion_version": model_version,
        "approval_decision_run_id": approval_run_id,
        "approval_record_sha256": approval_sha256,
    }
    mismatched = [
        field
        for field, expected in expected_promotion_fields.items()
        if promotion.get(field) != expected
    ]
    if mismatched:
        raise ChampionResolutionError(
            f"promotion decision conflicts on: {', '.join(sorted(mismatched))}"
        )

    approval = _download_verified_json(
        client,
        run_id=approval_run_id,
        artifact_path="decision/approval-decision.json",
        expected_sha256=approval_sha256,
        decision_name="approval",
    )
    expected_approval_fields = {
        "decision": "approved",
        "registered_model_name": registered_model_name,
        "model_version": model_version,
    }
    mismatched = [
        field
        for field, expected in expected_approval_fields.items()
        if approval.get(field) != expected
    ]
    if mismatched:
        raise ChampionResolutionError(
            f"approval decision conflicts on: {', '.join(sorted(mismatched))}"
        )


def _download_verified_json(
    client: MlflowClient,
    *,
    run_id: str,
    artifact_path: str,
    expected_sha256: str,
    decision_name: str,
) -> dict[str, Any]:
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ChampionResolutionError(f"{decision_name} decision run is not finished")
    with tempfile.TemporaryDirectory(prefix=f"verify-{decision_name}-") as temp_directory:
        downloaded_path = Path(
            client.download_artifacts(run_id, artifact_path, temp_directory)
        )
        decision_bytes = downloaded_path.read_bytes()
    if hashlib.sha256(decision_bytes).hexdigest() != expected_sha256:
        raise ChampionResolutionError(f"{decision_name} decision checksum does not match")
    try:
        decision = json.loads(decision_bytes)
    except json.JSONDecodeError as error:
        raise ChampionResolutionError(f"{decision_name} decision is not valid JSON") from error
    if not isinstance(decision, dict):
        raise ChampionResolutionError(f"{decision_name} decision must be a JSON object")
    return decision


def _cache_directory(
    cache_root: Path,
    *,
    registered_model_name: str,
    model_version: str,
) -> Path:
    model_key = hashlib.sha256(registered_model_name.encode()).hexdigest()[:16]
    version_key = hashlib.sha256(model_version.encode()).hexdigest()[:16]
    return cache_root.resolve() / model_key / version_key


def _download_package(
    client: MlflowClient,
    *,
    source_run_id: str,
    cache_directory: Path,
) -> None:
    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="champion-download-",
        dir=cache_directory.parent,
    ) as temp_directory:
        downloaded_directory = Path(
            client.download_artifacts(
                source_run_id,
                "candidate-package",
                temp_directory,
            )
        )
        if not downloaded_directory.is_dir():
            raise ChampionResolutionError("downloaded champion package is not a directory")
        if cache_directory.exists():
            return
        downloaded_directory.replace(cache_directory)


def _verify_cached_package(
    cache_directory: Path,
    *,
    tags: dict[str, str],
) -> ModelArtifactManifest:
    if not cache_directory.is_dir():
        raise ChampionResolutionError("champion cache path is not a directory")
    manifest_path = cache_directory / "manifest.json"
    try:
        manifest = load_model_manifest(manifest_path)
    except ArtifactLoadError as error:
        raise ChampionResolutionError("cached champion package failed integrity checks") from error
    if manifest.status is not ModelApprovalStatus.CANDIDATE:
        raise ChampionResolutionError(
            "registry source package must remain an immutable candidate"
        )
    mismatched = [
        key
        for key, expected in immutable_version_tags(manifest).items()
        if tags.get(key) != expected
    ]
    if mismatched:
        raise ChampionResolutionError(
            f"cached champion conflicts with registry on: {', '.join(sorted(mismatched))}"
        )
    return manifest
