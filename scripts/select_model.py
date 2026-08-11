"""Run bounded Brief 03 tuning, calibration, validation gates, and winner selection."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

import wandb
from flight_delay.data.manifest import canonical_json_bytes
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.artifacts import build_training_baseline
from flight_delay.modeling.calibration import (
    fit_sigmoid_calibrator,
    partition_development_data,
    reliability_table,
)
from flight_delay.modeling.candidates import build_candidate
from flight_delay.modeling.evaluation import evaluate_binary, measure_single_row_latency
from flight_delay.modeling.selection import choose_winner, select_threshold, validation_gates
from flight_delay.modeling.tracking import TrackingError, git_provenance
from flight_delay.modeling.training import load_training_frames


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _model_bytes(model: Any) -> int:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.tell()


def _json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _partition_metadata(partitions: Any) -> dict[str, dict[str, float | int]]:
    return {
        name: {"row_count": len(frame), "target_prevalence": float(frame["target"].mean())}
        for name, frame in (
            ("base_fit", partitions.base_fit),
            ("tuning", partitions.tuning),
            ("refit", partitions.refit),
            ("calibration", partitions.calibration),
            ("validation", partitions.validation),
        )
    }


def _init_run(
    *,
    entity: str,
    project: str,
    mode: str,
    name: str,
    tags: list[str],
    config: dict[str, Any],
) -> Any:
    return wandb.init(
        entity=entity,
        project=project,
        mode=mode,
        name=name,
        job_type="brief03-model-selection",
        tags=tags,
        config=config,
        settings=wandb.Settings(code_dir="."),
    )


def _download_dataset(run: Any, artifact_name: str, expected_digest: str) -> tuple[Path, Any]:
    artifact = run.use_artifact(artifact_name, type="dataset")
    if artifact.digest != expected_digest:
        raise TrackingError(
            f"dataset digest mismatch: expected {expected_digest}, got {artifact.digest}"
        )
    root = Path(artifact.download(root="artifacts/brief03/dataset-v0"))
    return root, artifact


def _log_code(run: Any) -> None:
    run.log_code(
        root=".",
        include_fn=lambda path: path.endswith(".py"),
        exclude_fn=lambda path: any(
            part in path for part in ("data/", "wandb/", ".venv/", "evidence/", "__pycache__/")
        ),
    )


def run_tuning(
    *,
    entity: str,
    project: str,
    mode: str,
    experiment: dict[str, Any],
    git: Any,
) -> tuple[dict[str, Any], Path, Any]:
    """Run exactly six Candidate B base configurations on Jan-Aug / September."""

    variants = list(product(experiment["tuning"]["alpha"], experiment["tuning"]["class_weight"]))
    if len(variants) > 6:
        raise ValueError("Candidate B tuning matrix exceeds six configurations")
    results: list[dict[str, Any]] = []
    dataset_root: Path | None = None
    partitions = None
    for index, (alpha, class_weight) in enumerate(variants, start=1):
        parameters = {
            **experiment["model"],
            "alpha": float(alpha),
            "class_weight": class_weight,
        }
        config = {
            "candidate_id": "candidate_b",
            "variant": index,
            "parameters": parameters,
            "dataset_artifact": experiment["dataset_artifact"],
            "dataset_digest": experiment["dataset_digest"],
            "git_sha": git.sha,
            "git_dirty": git.dirty,
            "selection_split": "2025-09",
            "threshold_tuned": False,
        }
        run = _init_run(
            entity=entity,
            project=project,
            mode=mode,
            name=f"brief03-candidate-b-tuning-{index}",
            tags=["brief03-tuning"],
            config=config,
        )
        try:
            _log_code(run)
            dataset_root, artifact = _download_dataset(
                run, experiment["dataset_artifact"], experiment["dataset_digest"]
            )
            train, validation = load_training_frames(dataset_root)
            partitions = partition_development_data(train, validation)
            model, schema = build_candidate("candidate_b", parameters)
            validate_model_features(schema)
            model.fit(partitions.base_fit.loc[:, schema], partitions.base_fit["target"])
            probabilities = model.predict_proba(partitions.tuning.loc[:, schema])[:, 1]
            latency = measure_single_row_latency(
                model, partitions.tuning.loc[:, schema], sample_count=50
            )
            evidence = {
                "variant": index,
                "parameters": parameters,
                "average_precision": float(
                    average_precision_score(partitions.tuning["target"], probabilities)
                ),
                "roc_auc": float(roc_auc_score(partitions.tuning["target"], probabilities)),
                "latency_p95_ms": latency["p95_ms"],
                "model_bytes": _model_bytes(model),
                "run_id": run.id,
                "run_url": run.url,
                "dataset_artifact": artifact.qualified_name,
                "dataset_digest": artifact.digest,
            }
            run.config.update(
                {"partitions": _partition_metadata(partitions), "feature_schema": list(schema)},
                allow_val_change=True,
            )
            run.log(
                {
                    f"tuning/{key}": value
                    for key, value in evidence.items()
                    if isinstance(value, int | float)
                }
            )
            results.append(evidence)
        finally:
            run.finish()
    if dataset_root is None or partitions is None:
        raise RuntimeError("tuning did not initialize dataset partitions")
    winner = max(
        results,
        key=lambda row: (
            row["average_precision"],
            row["roc_auc"],
            -row["latency_p95_ms"],
            -row["model_bytes"],
        ),
    )
    return {"variants": results, "selected": winner}, dataset_root, partitions


def run_final_candidate(
    *,
    entity: str,
    project: str,
    mode: str,
    experiment: dict[str, Any],
    parameters: dict[str, Any],
    partitions: Any,
    git: Any,
) -> dict[str, Any]:
    candidate_id = experiment["candidate_id"]
    tag = "brief03-control" if candidate_id == "candidate_a_calibrated" else "brief03-candidate"
    run = _init_run(
        entity=entity,
        project=project,
        mode=mode,
        name=experiment["experiment_name"],
        tags=[tag],
        config={
            "candidate_id": candidate_id,
            "parameters": parameters,
            "dataset_artifact": experiment["dataset_artifact"],
            "dataset_digest": experiment["dataset_digest"],
            "git_sha": git.sha,
            "git_dirty": git.dirty,
            "calibration_method": "sigmoid",
            "validation_only": True,
            "final_test_evaluated": False,
        },
    )
    figures: dict[str, Any] = {}
    try:
        _log_code(run)
        _, artifact = _download_dataset(
            run, experiment["dataset_artifact"], experiment["dataset_digest"]
        )
        base, schema = build_candidate(candidate_id, parameters)
        validate_model_features(schema)
        base.fit(partitions.refit.loc[:, schema], partitions.refit["target"])
        calibrated = fit_sigmoid_calibrator(
            base,
            partitions.calibration.loc[:, schema],
            partitions.calibration["target"],
        )
        validate_model_features(schema)
        validation_features = partitions.validation.loc[:, schema]
        probabilities = calibrated.predict_proba(validation_features)[:, 1]
        threshold = select_threshold(partitions.validation["target"], probabilities)
        evaluation = evaluate_binary(
            partitions.validation["target"], probabilities, threshold=threshold.threshold
        )
        figures = evaluation.figures
        table, ece = reliability_table(partitions.validation["target"], probabilities)
        metrics = {
            **evaluation.metrics,
            "prevalence": float(partitions.validation["target"].mean()),
            "expected_calibration_error": ece,
            "lineage_verified": artifact.digest == experiment["dataset_digest"],
            "leakage_check_passed": True,
        }
        latency = measure_single_row_latency(
            calibrated,
            validation_features,
            sample_count=int(experiment.get("latency_sample_count", 200)),
        )
        bundle_bytes = _model_bytes(calibrated)
        gates = validation_gates(
            metrics,
            ece=ece,
            latency_p95_ms=float(latency["p95_ms"]),
            bundle_bytes=bundle_bytes,
        )
        bundle = Path("artifacts/brief03/candidates") / candidate_id
        bundle.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, bundle / "model.joblib")
        _json(bundle / "feature_schema.json", {"features": list(schema)})
        threshold_evidence = {
            "threshold": threshold.threshold,
            "precision": threshold.precision,
            "recall": threshold.recall,
            "f1": threshold.f1,
            "objective": "maximum F1 subject to recall >= 0.60",
            "tie_breaks": [
                "higher_f1",
                "higher_recall",
                "higher_precision",
                "closest_to_0.5",
            ],
            "evaluated_threshold_count": len(threshold.table),
        }
        _json(bundle / "threshold.json", threshold_evidence)
        _json(
            bundle / "calibration.json",
            {"method": "sigmoid", "ece": ece, "reliability_table": table},
        )
        _json(bundle / "metrics.json", {"split": "validation", "metrics": metrics, "gates": gates})
        _json(
            bundle / "training_baseline.json",
            build_training_baseline(
                partitions.refit,
                dataset_artifact=artifact.qualified_name,
                dataset_digest=artifact.digest,
            ),
        )
        metadata = {
            "candidate_id": candidate_id,
            "parameters": parameters,
            "dataset_artifact": artifact.qualified_name,
            "dataset_digest": artifact.digest,
            "git_sha": git.sha,
            "git_dirty": git.dirty,
            "partitions": _partition_metadata(partitions),
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "wandb": wandb.__version__,
            },
            "validation_only": True,
            "final_test_evaluated": False,
            "registry_promoted": False,
            "threshold": threshold_evidence,
            "validation_metrics": metrics,
            "validation_gates": gates,
        }
        _json(bundle / "metadata.json", metadata)
        (bundle / "MODEL_CARD.md").write_text(
            f"# {candidate_id}\n\nCalibrated on October 2025 and evaluated on validation only. "
            "Final test and Registry promotion have not occurred.\n",
            encoding="utf-8",
        )
        bundle_bytes = sum(path.stat().st_size for path in bundle.iterdir() if path.is_file())
        gates = validation_gates(
            metrics,
            ece=ece,
            latency_p95_ms=float(latency["p95_ms"]),
            bundle_bytes=bundle_bytes,
        )
        metadata["validation_gates"] = gates
        metadata["bundle_bytes"] = bundle_bytes
        _json(bundle / "metrics.json", {"split": "validation", "metrics": metrics, "gates": gates})
        _json(bundle / "metadata.json", metadata)
        artifact_out = wandb.Artifact(
            f"flight-delay-{candidate_id.replace('_', '-')}", type="model", metadata=metadata
        )
        artifact_out.add_dir(str(bundle), name="model_bundle")
        logged = run.log_artifact(artifact_out).wait()
        run.config.update(
            {
                "partitions": _partition_metadata(partitions),
                "feature_schema": list(schema),
                "threshold": threshold.threshold,
                "validation_gates": gates,
            },
            allow_val_change=True,
        )
        run.log(
            {
                f"validation/{key}": value
                for key, value in metrics.items()
                if isinstance(value, int | float | bool)
            }
        )
        run.log({f"latency/{key}": value for key, value in latency.items()})
        for name, figure in figures.items():
            run.log({f"validation_plot/{name}": wandb.Image(figure)})
        threshold_figure, axis = plt.subplots(figsize=(7, 5))
        threshold_frame = pd.DataFrame(threshold.table)
        axis.plot(threshold_frame["threshold"], threshold_frame["precision"], label="precision")
        axis.plot(threshold_frame["threshold"], threshold_frame["recall"], label="recall")
        axis.plot(threshold_frame["threshold"], threshold_frame["f1"], label="f1")
        axis.axvline(threshold.threshold, color="black", linestyle="--")
        axis.legend()
        threshold_figure.tight_layout()
        run.log({"validation_plot/threshold_curve": wandb.Image(threshold_figure)})
        plt.close(threshold_figure)
        return {
            "candidate_id": candidate_id,
            "run_id": run.id,
            "run_url": run.url,
            "model_artifact": logged.qualified_name,
            "model_digest": logged.digest,
            "model_url": logged.url,
            "feature_schema": list(schema),
            "parameters": parameters,
            "threshold": threshold_evidence,
            "metrics": metrics,
            "calibration": {"ece": ece, "reliability_table": table},
            "latency": latency,
            "bundle_bytes": bundle_bytes,
            "gates": gates,
            "bundle_directory": str(bundle),
        }
    finally:
        for figure in figures.values():
            plt.close(figure)
        run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Brief 03 validation-only model selection.")
    parser.add_argument(
        "--candidate-a", type=Path, default=Path("configs/experiments/candidate_a_calibrated.yaml")
    )
    parser.add_argument(
        "--candidate-b", type=Path, default=Path("configs/experiments/candidate_b.yaml")
    )
    parser.add_argument("--wandb-mode", choices=("online",), default="online")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        entity = os.environ["WANDB_ENTITY"]
        project = os.getenv("WANDB_PROJECT", "us-flight-delay-mlops")
        git = git_provenance()
        if git.dirty:
            raise TrackingError("model selection requires a clean worktree")
        candidate_a = _load_yaml(args.candidate_a)
        candidate_b = _load_yaml(args.candidate_b)
        tuning, _, partitions = run_tuning(
            entity=entity,
            project=project,
            mode=args.wandb_mode,
            experiment=candidate_b,
            git=git,
        )
        control = run_final_candidate(
            entity=entity,
            project=project,
            mode=args.wandb_mode,
            experiment=candidate_a,
            parameters=candidate_a["model"],
            partitions=partitions,
            git=git,
        )
        selected_parameters = tuning["selected"]["parameters"]
        candidate = run_final_candidate(
            entity=entity,
            project=project,
            mode=args.wandb_mode,
            experiment=candidate_b,
            parameters=selected_parameters,
            partitions=partitions,
            git=git,
        )
        candidates = {control["candidate_id"]: control, candidate["candidate_id"]: candidate}
        try:
            winner = choose_winner(candidates)
        except ValueError as error:
            winner = None
            stop_reason = str(error)
        else:
            stop_reason = None
            winner_run = wandb.Api().run(f"{entity}/{project}/{candidates[winner]['run_id']}")
            winner_run.tags = tuple(sorted(set(winner_run.tags) | {"brief03-release-candidate"}))
            winner_run.update()
        result = {
            "dataset_artifact": candidate_b["dataset_artifact"],
            "dataset_digest": candidate_b["dataset_digest"],
            "git_sha": git.sha,
            "git_dirty": git.dirty,
            "partitions": _partition_metadata(partitions),
            "tuning": tuning,
            "candidates": candidates,
            "winner": winner,
            "selection_status": "selected" if winner else "blocked",
            "stop_reason": stop_reason,
            "final_test_evaluated": False,
            "registry_aliases_created": False,
        }
        output = Path("artifacts/brief03/validation_selection_report.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        _json(output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if winner else 2
    except (KeyError, OSError, TypeError, ValueError, TrackingError) as error:
        print(f"model selection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
