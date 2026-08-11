"""Small, testable W&B boundary for dataset and experiment provenance."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flight_delay.data.manifest import read_manifest


class TrackingError(RuntimeError):
    """Raised when experiment tracking cannot satisfy its provenance contract."""


@dataclass(frozen=True)
class GitProvenance:
    """Repository revision recorded with every artifact and run."""

    sha: str
    dirty: bool


@dataclass(frozen=True)
class ArtifactReference:
    """Stable identifiers returned after an artifact upload completes."""

    qualified_name: str
    digest: str
    run_id: str
    run_url: str
    artifact_url: str


def git_provenance(repository: Path = Path(".")) -> GitProvenance:
    """Read the current Git SHA and dirty flag without mutating the repository."""

    environment_sha = os.getenv("FLIGHT_DELAY_GIT_SHA")
    environment_dirty = os.getenv("FLIGHT_DELAY_GIT_DIRTY")
    if environment_sha is not None or environment_dirty is not None:
        if not environment_sha or not re.fullmatch(r"[0-9a-f]{40}", environment_sha):
            raise TrackingError("FLIGHT_DELAY_GIT_SHA must be a 40-character lowercase Git SHA")
        if environment_dirty not in {"true", "false"}:
            raise TrackingError("FLIGHT_DELAY_GIT_DIRTY must be true or false")
        return GitProvenance(sha=environment_sha, dirty=environment_dirty == "true")
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise TrackingError(f"cannot read Git provenance: {error}") from error
    return GitProvenance(sha=sha, dirty=bool(status.strip()))


def dataset_artifact_metadata(
    source_manifest_path: Path,
    processed_manifest_path: Path,
    provenance: GitProvenance,
) -> dict[str, Any]:
    """Build complete, timestamp-free W&B metadata from stable manifests."""

    source = read_manifest(source_manifest_path)
    processed = read_manifest(processed_manifest_path)
    preprocessing = processed["preprocessing"]
    return {
        "month_range": source["month_range"],
        "split_boundaries": processed["split_boundaries"],
        "split_counts": processed["split_counts"],
        "random_seed": preprocessing["random_seed"],
        "monthly_sample_cap": preprocessing["monthly_sample_cap"],
        "source_manifest_digest": source["manifest_digest"],
        "processed_manifest_digest": processed["manifest_digest"],
        "git_sha": provenance.sha,
        "git_dirty": provenance.dirty,
    }


def _artifact_identity(artifact: Any, entity: str, project: str) -> tuple[str, str, str]:
    name = str(getattr(artifact, "name", ""))
    qualified = str(getattr(artifact, "qualified_name", ""))
    if not qualified and name:
        qualified = name if name.count("/") == 2 else f"{entity}/{project}/{name}"
    digest = str(getattr(artifact, "digest", ""))
    url = str(getattr(artifact, "url", ""))
    if not qualified or not digest:
        raise TrackingError("W&B did not return a qualified artifact version and digest")
    return qualified, digest, url


def publish_dataset_artifact(
    *,
    source_manifest_path: Path,
    processed_manifest_path: Path,
    processed_directory: Path,
    entity: str,
    project: str,
    mode: str,
    artifact_name: str = "flight-delay-bts-sampled",
    repository: Path = Path("."),
    wandb_module: Any | None = None,
) -> ArtifactReference:
    """Upload the three splits and stable manifests as one versioned dataset artifact."""

    if mode not in {"online", "offline", "disabled"}:
        raise TrackingError(f"unsupported W&B mode: {mode}")
    if mode == "online" and not entity:
        raise TrackingError("WANDB_ENTITY is required for online artifact publication")
    if wandb_module is None:
        if mode == "online" and not os.getenv("WANDB_API_KEY"):
            raise TrackingError("WANDB_API_KEY is required for online artifact publication")
        try:
            import wandb as wandb_module
        except ImportError as error:
            raise TrackingError("wandb is not installed") from error

    provenance = git_provenance(repository)
    metadata = dataset_artifact_metadata(source_manifest_path, processed_manifest_path, provenance)
    settings_factory = getattr(wandb_module, "Settings", None)
    settings = settings_factory(code_dir=".") if settings_factory else None
    run = wandb_module.init(
        entity=entity or None,
        project=project,
        job_type="log_dataset",
        mode=mode,
        config=metadata,
        settings=settings,
    )
    if run is None:
        raise TrackingError("W&B failed to initialize a run")
    try:
        artifact = wandb_module.Artifact(
            artifact_name,
            type="dataset",
            description=(
                "Official BTS Reporting Carrier sample with chronological train, validation, "
                "and sealed test splits."
            ),
            metadata=metadata,
        )
        for split_name in ("train", "validation", "test"):
            artifact.add_file(
                str(processed_directory / f"{split_name}.parquet"),
                name=f"data/processed/{split_name}.parquet",
            )
        artifact.add_file(str(source_manifest_path), name="data/manifests/source_manifest.json")
        artifact.add_file(
            str(processed_manifest_path), name="data/manifests/processed_manifest.json"
        )
        run.log_code(
            root=str(repository),
            include_fn=lambda path: path.endswith(".py"),
            exclude_fn=lambda path: any(
                part in path for part in ("data/", "wandb/", ".venv/", "evidence/", "__pycache__/")
            ),
        )
        uploaded = run.log_artifact(artifact)
        completed = uploaded.wait()
        qualified, digest, artifact_url = _artifact_identity(completed, entity, project)
        return ArtifactReference(
            qualified_name=qualified,
            digest=digest,
            run_id=str(run.id),
            run_url=str(run.url),
            artifact_url=artifact_url,
        )
    finally:
        run.finish()
