"""Validation-only training orchestration with explicit W&B dataset lineage."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from flight_delay.data.manifest import read_manifest
from flight_delay.data.prepare import CANDIDATE_A_FEATURES
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.artifacts import build_training_baseline, write_model_bundle
from flight_delay.modeling.baselines import build_estimator
from flight_delay.modeling.evaluation import evaluate_binary, measure_single_row_latency
from flight_delay.modeling.tracking import TrackingError, git_provenance


@dataclass(frozen=True)
class TrainingResult:
    candidate_id: str
    run_id: str
    run_url: str
    dataset_artifact: str
    dataset_digest: str
    model_artifact: str
    model_digest: str
    model_artifact_url: str
    model_byte_size: int
    model_load_ms: float
    latency: dict[str, float | int]
    validation_metrics: dict[str, float | int]


def load_training_frames(artifact_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train and validation only; the sealed test path is deliberately absent."""

    root = artifact_directory / "data" / "processed"
    return pd.read_parquet(root / "train.parquet"), pd.read_parquet(root / "validation.parquet")


def _qualified_artifact(artifact: Any, entity: str, project: str) -> tuple[str, str, str]:
    name = str(getattr(artifact, "qualified_name", "") or getattr(artifact, "name", ""))
    if name.count("/") != 2:
        name = f"{entity}/{project}/{name}"
    digest = str(getattr(artifact, "digest", ""))
    url = str(getattr(artifact, "url", ""))
    if not name or not digest:
        raise TrackingError("W&B artifact identity is incomplete")
    return name, digest, url


def run_training_experiment(
    *,
    experiment: dict[str, Any],
    entity: str,
    project: str,
    mode: str,
    repository: Path = Path("."),
    bundle_root: Path = Path("artifacts/model_bundle"),
    wandb_module: Any | None = None,
) -> TrainingResult:
    """Train one allowed candidate and log validation-only evidence and a model artifact."""

    if mode == "online" and (not entity or not os.getenv("WANDB_API_KEY")):
        raise TrackingError("WANDB_ENTITY and WANDB_API_KEY are required for online training")
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as error:
            raise TrackingError("wandb is not installed") from error
    candidate_id = str(experiment["candidate_id"])
    feature_schema = list(CANDIDATE_A_FEATURES)
    validate_model_features(feature_schema)
    provenance = git_provenance(repository)
    run_config = {
        "experiment_name": experiment["experiment_name"],
        "candidate_id": candidate_id,
        "git_sha": provenance.sha,
        "git_dirty": provenance.dirty,
        "random_seed": int(experiment["random_seed"]),
        "threshold": float(experiment["threshold"]),
        "hyperparameters": experiment["model"],
        "safe_feature_schema": feature_schema,
        "leakage_check": "passed",
        "validation_only": True,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "wandb": getattr(wandb_module, "__version__", "test-double"),
        },
    }
    settings_factory = getattr(wandb_module, "Settings", None)
    settings = settings_factory(code_dir=".") if settings_factory else None
    run = wandb_module.init(
        entity=entity or None,
        project=project,
        name=str(experiment["experiment_name"]),
        job_type="train",
        mode=mode,
        config=run_config,
        settings=settings,
    )
    if run is None:
        raise TrackingError("W&B failed to initialize a training run")
    figures: dict[str, Any] = {}
    try:
        run.log_code(
            root=str(repository),
            include_fn=lambda path: path.endswith(".py"),
            exclude_fn=lambda path: any(
                part in path for part in ("data/", "wandb/", ".venv/", "evidence/", "__pycache__/")
            ),
        )
        dataset = run.use_artifact(str(experiment["dataset_artifact"]), type="dataset")
        dataset_name, dataset_digest, _ = _qualified_artifact(dataset, entity, project)
        artifact_root = Path(dataset.download(root=f"artifacts/datasets/{run.id}"))
        train, validation = load_training_frames(artifact_root)
        processed_manifest = read_manifest(artifact_root / "data/manifests/processed_manifest.json")
        source_manifest = read_manifest(artifact_root / "data/manifests/source_manifest.json")
        features_train = train.loc[:, feature_schema]
        features_validation = validation.loc[:, feature_schema]
        validate_model_features(features_train.columns)
        validate_model_features(features_validation.columns)
        estimator = build_estimator(candidate_id, experiment["model"])
        estimator.fit(features_train, train["target"])
        probabilities = estimator.predict_proba(features_validation)[:, 1]
        evaluation = evaluate_binary(
            validation["target"], probabilities, threshold=float(experiment["threshold"])
        )
        figures = evaluation.figures
        baseline = build_training_baseline(
            train, dataset_artifact=dataset_name, dataset_digest=dataset_digest
        )
        metadata = {
            **run_config,
            "dataset_artifact": dataset_name,
            "dataset_digest": dataset_digest,
            "source_manifest_digest": source_manifest["manifest_digest"],
            "processed_manifest_digest": processed_manifest["manifest_digest"],
            "split_boundaries": processed_manifest["split_boundaries"],
            "split_counts": processed_manifest["split_counts"],
            "final_test_evaluated": False,
            "registry_promoted": False,
        }
        run.config.update(metadata, allow_val_change=True)
        bundle = write_model_bundle(
            directory=bundle_root / candidate_id,
            model=estimator,
            feature_schema=feature_schema,
            threshold=float(experiment["threshold"]),
            training_baseline=baseline,
            metrics=evaluation.metrics,
            metadata=metadata,
        )
        latency = measure_single_row_latency(
            bundle.loaded_model,
            features_validation,
            sample_count=int(experiment.get("latency_sample_count", 200)),
        )
        run.log({f"validation/{key}": value for key, value in evaluation.metrics.items()})
        run.log({f"latency/{key}": value for key, value in latency.items()})
        run.log({"model/byte_size": bundle.byte_size, "model/load_ms": bundle.model_load_ms})
        for name, figure in figures.items():
            run.log({f"validation_plot/{name}": wandb_module.Image(figure)})
        model_artifact = wandb_module.Artifact(
            "flight-delay-model", type="model", metadata=metadata
        )
        model_artifact.add_dir(str(bundle.directory), name="model_bundle")
        completed_model = run.log_artifact(model_artifact).wait()
        model_name, model_digest, model_url = _qualified_artifact(completed_model, entity, project)
        return TrainingResult(
            candidate_id=candidate_id,
            run_id=str(run.id),
            run_url=str(run.url),
            dataset_artifact=dataset_name,
            dataset_digest=dataset_digest,
            model_artifact=model_name,
            model_digest=model_digest,
            model_artifact_url=model_url,
            model_byte_size=bundle.byte_size,
            model_load_ms=bundle.model_load_ms,
            latency=latency,
            validation_metrics=evaluation.metrics,
        )
    finally:
        for figure in figures.values():
            plt.close(figure)
        run.finish()


def training_result_dict(result: TrainingResult) -> dict[str, Any]:
    """Return a JSON-ready training result."""

    return asdict(result)
