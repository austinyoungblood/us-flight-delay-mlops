from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from flight_delay.data.manifest import write_manifest
from flight_delay.modeling.tracking import GitProvenance
from flight_delay.modeling.training import load_training_frames, run_training_experiment


def test_training_loader_never_reads_sealed_test(monkeypatch: Any, tmp_path: Path) -> None:
    observed: list[Path] = []

    def fake_read(path: Path) -> pd.DataFrame:
        observed.append(path)
        return pd.DataFrame({"target": [0, 1]})

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    load_training_frames(tmp_path)

    assert [path.name for path in observed] == ["train.parquet", "validation.parquet"]
    assert all(path.name != "test.parquet" for path in observed)


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_date": pd.date_range("2025-01-01", periods=rows),
            "Reporting_Airline": ["UA" if index % 2 else "WN" for index in range(rows)],
            "Origin": ["DEN"] * rows,
            "Dest": ["LAX" if index % 3 else "SFO" for index in range(rows)],
            "Month": [1] * rows,
            "DayofMonth": [(index % 28) + 1 for index in range(rows)],
            "DayOfWeek": [(index % 7) + 1 for index in range(rows)],
            "CRSDepTime": [700 + index for index in range(rows)],
            "CRSArrTime": [900 + index for index in range(rows)],
            "CRSElapsedTime": [120.0 + index for index in range(rows)],
            "Distance": [800.0 + index for index in range(rows)],
            "scheduled_departure_hour": [7] * rows,
            "scheduled_arrival_hour": [9] * rows,
            "scheduled_departure_minute_bucket": [0] * rows,
            "scheduled_arrival_minute_bucket": [0] * rows,
            "is_weekend": [index % 2 for index in range(rows)],
            "target": [index % 2 for index in range(rows)],
        }
    )


class FakeConfig(dict[str, Any]):
    def update(self, other: Any, *, allow_val_change: bool = False) -> None:
        super().update(other)


class FakeDataset:
    qualified_name = "entity/project/flight-delay-bts-sampled:v0"
    digest = "dataset-digest"
    url = "https://wandb.example/dataset/v0"

    def __init__(self, root: Path) -> None:
        self.root = root

    def download(self, *, root: str) -> str:
        return str(self.root)


class FakeModelVersion:
    qualified_name = "entity/project/flight-delay-model:v0"
    digest = "model-digest"
    url = "https://wandb.example/model/v0"


class FakeLoggedModel:
    def wait(self) -> FakeModelVersion:
        return FakeModelVersion()


class FakeModelArtifact:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.directory = ""

    def add_dir(self, directory: str, *, name: str) -> None:
        self.directory = directory


class FakeTrainingRun:
    id = "training-run"
    url = "https://wandb.example/run/training-run"

    def __init__(self, artifact_root: Path) -> None:
        self.dataset = FakeDataset(artifact_root)
        self.config = FakeConfig()
        self.finished = False
        self.logged: list[dict[str, Any]] = []

    def log_code(self, **kwargs: Any) -> None:
        pass

    def use_artifact(self, name: str, *, type: str) -> FakeDataset:
        return self.dataset

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)

    def log_artifact(self, artifact: FakeModelArtifact) -> FakeLoggedModel:
        return FakeLoggedModel()

    def finish(self) -> None:
        self.finished = True


class FakeTrainingWandb:
    __version__ = "0.28.1"
    Artifact = FakeModelArtifact

    def __init__(self, artifact_root: Path) -> None:
        self.run = FakeTrainingRun(artifact_root)

    @staticmethod
    def Settings(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    @staticmethod
    def Image(figure: Any) -> Any:
        return figure

    def init(self, **kwargs: Any) -> FakeTrainingRun:
        self.run.config.update(kwargs["config"])
        return self.run


def _artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    processed = root / "data/processed"
    manifests = root / "data/manifests"
    processed.mkdir(parents=True)
    manifests.mkdir(parents=True)
    frame = _frame(30)
    frame.iloc[:20].to_parquet(processed / "train.parquet", index=False)
    frame.iloc[20:].to_parquet(processed / "validation.parquet", index=False)
    source = write_manifest(
        manifests / "source_manifest.json",
        {"schema_version": 1, "month_range": {"start": "2025-01", "end": "2026-05"}},
    )
    write_manifest(
        manifests / "processed_manifest.json",
        {
            "schema_version": 1,
            "source_manifest_digest": source["manifest_digest"],
            "preprocessing": {"random_seed": 42, "monthly_sample_cap": 75000},
            "split_boundaries": {"train": {}, "validation": {}, "test": {}},
            "split_counts": {
                "train": {"row_count": 20, "target_prevalence": 0.5},
                "validation": {"row_count": 10, "target_prevalence": 0.5},
                "test": {"row_count": 0, "target_prevalence": 0.0},
            },
        },
    )
    return root


@pytest.mark.parametrize("candidate_id", ["dummy", "candidate_a"])
def test_training_orchestration_uses_artifact_and_logs_model(
    monkeypatch: Any, tmp_path: Path, candidate_id: str
) -> None:
    fake = FakeTrainingWandb(_artifact_root(tmp_path))
    monkeypatch.setattr(
        "flight_delay.modeling.training.git_provenance",
        lambda repository: GitProvenance("abc123", False),
    )
    experiment = {
        "experiment_name": candidate_id,
        "candidate_id": candidate_id,
        "random_seed": 42,
        "threshold": 0.5,
        "dataset_artifact": "flight-delay-bts-sampled:latest",
        "model": {},
        "latency_sample_count": 3,
    }

    result = run_training_experiment(
        experiment=experiment,
        entity="entity",
        project="project",
        mode="disabled",
        repository=tmp_path,
        bundle_root=tmp_path / "bundles",
        wandb_module=fake,
    )

    assert result.candidate_id == candidate_id
    assert result.dataset_artifact.endswith(":v0")
    assert result.model_artifact.endswith(":v0")
    assert result.latency["sample_count"] == 3
    assert fake.run.config["final_test_evaluated"] is False
    assert fake.run.finished
