from __future__ import annotations

from pathlib import Path
from typing import Any

from flight_delay.data.manifest import write_manifest
from flight_delay.modeling.tracking import GitProvenance, git_provenance, publish_dataset_artifact


class FakeCompletedArtifact:
    qualified_name = "entity/project/flight-delay-bts-sampled:v0"
    digest = "artifact-digest"
    url = "https://wandb.example/artifacts/v0"


class FakeLoggedArtifact:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> FakeCompletedArtifact:
        self.waited = True
        return FakeCompletedArtifact()


class FakeArtifact:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))


class FakeRun:
    id = "run-123"
    url = "https://wandb.example/runs/run-123"

    def __init__(self) -> None:
        self.logged = FakeLoggedArtifact()
        self.artifact: FakeArtifact | None = None
        self.finished = False
        self.code_logged = False

    def log_code(self, **kwargs: Any) -> None:
        self.code_logged = True

    def log_artifact(self, artifact: FakeArtifact) -> FakeLoggedArtifact:
        self.artifact = artifact
        return self.logged

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs: dict[str, Any] = {}

    @staticmethod
    def Settings(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run


def _manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_path = tmp_path / "source.json"
    processed_path = tmp_path / "processed.json"
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    source = write_manifest(
        source_path,
        {"schema_version": 1, "month_range": {"start": "2025-01", "end": "2026-05"}},
    )
    write_manifest(
        processed_path,
        {
            "schema_version": 1,
            "source_manifest_digest": source["manifest_digest"],
            "preprocessing": {"random_seed": 42, "monthly_sample_cap": 75000},
            "split_boundaries": {"train": {"start": "2025-01-01"}},
            "split_counts": {"train": {"row_count": 2, "target_prevalence": 0.5}},
        },
    )
    for split in ("train", "validation", "test"):
        (processed_directory / f"{split}.parquet").write_bytes(b"PAR1")
    return source_path, processed_path, processed_directory


def test_publish_dataset_uses_fake_run_and_waits(monkeypatch: Any, tmp_path: Path) -> None:
    source, processed, directory = _manifests(tmp_path)
    fake = FakeWandb()
    monkeypatch.setattr(
        "flight_delay.modeling.tracking.git_provenance",
        lambda repository: GitProvenance("abc123", False),
    )

    reference = publish_dataset_artifact(
        source_manifest_path=source,
        processed_manifest_path=processed,
        processed_directory=directory,
        entity="entity",
        project="project",
        mode="disabled",
        wandb_module=fake,
    )

    assert reference.qualified_name.endswith(":v0")
    assert reference.digest == "artifact-digest"
    assert fake.run.logged.waited
    assert fake.run.finished
    assert fake.run.code_logged
    assert fake.run.artifact is not None
    assert [name for _, name in fake.run.artifact.files] == [
        "data/processed/train.parquet",
        "data/processed/validation.parquet",
        "data/processed/test.parquet",
        "data/manifests/source_manifest.json",
        "data/manifests/processed_manifest.json",
    ]
    assert fake.init_kwargs["config"]["git_sha"] == "abc123"


def test_git_provenance_accepts_validated_container_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("FLIGHT_DELAY_GIT_SHA", "a" * 40)
    monkeypatch.setenv("FLIGHT_DELAY_GIT_DIRTY", "false")

    assert git_provenance() == GitProvenance("a" * 40, False)
