"""Run rolling-origin search and November calibrated-finalist selection."""

from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import warnings
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import wandb
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from flight_delay.data.download import sha256_file
from flight_delay.data.manifest import canonical_json_bytes, read_manifest
from flight_delay.features.leakage import validate_model_features
from flight_delay.modeling.calibration import calibration_audit, fit_calibrator
from flight_delay.modeling.evaluation import evaluate_binary, measure_single_row_latency
from flight_delay.modeling.remediation import (
    authorized_calibration_ids,
    build_remediation_model,
    partition_remediation_data,
    prior_scores,
    rank_base_results,
    rolling_origin_folds,
    validate_remediation_matrix,
)
from flight_delay.modeling.selection import (
    choose_remediation_winner,
    november_gates,
    select_threshold_remediation,
)
from flight_delay.modeling.tracking import TrackingError, git_provenance


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("remediation configuration must be a mapping")
    validate_remediation_matrix(payload["configurations"])
    methods = tuple(payload["calibration_methods"])
    if methods != ("sigmoid", "isotonic"):
        raise ValueError("calibration methods must be exactly sigmoid and isotonic")
    return payload


def _init_run(
    *, entity: str, project: str, name: str, tags: list[str], config: dict[str, Any]
) -> Any:
    return wandb.init(
        entity=entity,
        project=project,
        mode="online",
        name=name,
        job_type="brief04-remediation",
        tags=tags,
        config=config,
        settings=wandb.Settings(code_dir="."),
    )


def _log_code(run: Any) -> None:
    run.log_code(
        root=".",
        include_fn=lambda path: path.endswith(".py"),
        exclude_fn=lambda path: any(
            part in path for part in ("data/", "wandb/", ".venv/", "evidence/", "__pycache__/")
        ),
    )


def _download_dataset(run: Any, config: dict[str, Any]) -> tuple[Path, Any]:
    artifact = run.use_artifact(config["dataset_artifact"], type="dataset")
    if artifact.digest != config["dataset_digest"]:
        raise TrackingError(
            f"dataset digest mismatch: expected {config['dataset_digest']}, got {artifact.digest}"
        )
    root = Path(artifact.download(root="artifacts/brief04/dataset-v0"))
    return root, artifact


def _train_frame(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "data/processed/train.parquet")


def _november_frame(root: Path) -> pd.DataFrame:
    return pd.read_parquet(
        root / "data/processed/validation.parquet",
        filters=[
            ("flight_date", ">=", pd.Timestamp("2025-11-01")),
            ("flight_date", "<", pd.Timestamp("2025-12-01")),
        ],
    )


def _lineage(root: Path, artifact: Any, config: dict[str, Any], git: Any) -> dict[str, Any]:
    source = read_manifest(root / "data/manifests/source_manifest.json")
    processed = read_manifest(root / "data/manifests/processed_manifest.json")
    return {
        "dataset_artifact": artifact.qualified_name,
        "dataset_digest": artifact.digest,
        "source_manifest_digest": source["manifest_digest"],
        "processed_manifest_digest": processed["manifest_digest"],
        "train_sha256": sha256_file(root / "data/processed/train.parquet"),
        "validation_sha256": sha256_file(root / "data/processed/validation.parquet"),
        "expected_dataset_digest": config["dataset_digest"],
        "git_sha": git.sha,
        "git_dirty": git.dirty,
    }


def _fit_without_convergence_warning(model: Any, features: pd.DataFrame, target: pd.Series) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, target)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(str(convergence[0].message))


def _model_bytes(model: Any) -> int:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.tell()


def _partition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(frame["flight_date"])
    return {
        "row_count": len(frame),
        "prevalence": float(frame["target"].mean()),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
    }


def run_base_search(
    *, entity: str, project: str, config: dict[str, Any], git: Any
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Run exactly R0-R5, with four folds inside each W&B run."""

    results: list[dict[str, Any]] = []
    for config_id, parameters in config["configurations"].items():
        run = _init_run(
            entity=entity,
            project=project,
            name=f"brief04-base-{config_id.lower()}",
            tags=["brief04-rolling-origin"],
            config={
                "configuration_id": config_id,
                "parameters": parameters,
                "dataset_artifact": config["dataset_artifact"],
                "dataset_digest": config["dataset_digest"],
                "git_sha": git.sha,
                "git_dirty": git.dirty,
                "threshold_tuned": False,
                "december_evaluated": False,
                "final_test_evaluated": False,
            },
        )
        result: dict[str, Any] = {
            "configuration_id": config_id,
            "parameters": parameters,
            "run_id": run.id,
            "run_url": run.url,
            "status": "failed",
            "folds": [],
        }
        try:
            _log_code(run)
            root, artifact = _download_dataset(run, config)
            lineage = _lineage(root, artifact, config, git)
            train = _train_frame(root)
            folds = rolling_origin_folds(train)
            for fold in folds:
                model, schema = build_remediation_model(config_id, parameters)
                validate_model_features(schema)
                _fit_without_convergence_warning(model, fold.fit.loc[:, schema], fold.fit["target"])
                validate_model_features(schema)
                probabilities = model.predict_proba(fold.evaluation.loc[:, schema])[:, 1]
                latency = measure_single_row_latency(
                    model, fold.evaluation.loc[:, schema], sample_count=50
                )
                fold_result = {
                    "fold": fold.fold,
                    "fit": _partition_summary(fold.fit),
                    "evaluation": _partition_summary(fold.evaluation),
                    "average_precision": float(
                        average_precision_score(fold.evaluation["target"], probabilities)
                    ),
                    "roc_auc": float(roc_auc_score(fold.evaluation["target"], probabilities)),
                    "log_loss": float(log_loss(fold.evaluation["target"], probabilities)),
                    "latency_p95_ms": float(latency["p95_ms"]),
                    "model_bytes": _model_bytes(model),
                }
                result["folds"].append(fold_result)
            result.update(
                {
                    "status": "completed",
                    "feature_schema": list(schema),
                    "mean_average_precision": mean(
                        row["average_precision"] for row in result["folds"]
                    ),
                    "mean_roc_auc": mean(row["roc_auc"] for row in result["folds"]),
                    "std_average_precision": pstdev(
                        row["average_precision"] for row in result["folds"]
                    ),
                    "mean_log_loss": mean(row["log_loss"] for row in result["folds"]),
                    "mean_latency_p95_ms": mean(row["latency_p95_ms"] for row in result["folds"]),
                    "mean_model_bytes": mean(row["model_bytes"] for row in result["folds"]),
                    "lineage": lineage,
                }
            )
            run.config.update(
                {
                    "feature_schema": list(schema),
                    "leakage_check_passed": True,
                    "folds": [
                        {
                            "fold": fold.fold,
                            "fit": _partition_summary(fold.fit),
                            "evaluation": _partition_summary(fold.evaluation),
                        }
                        for fold in folds
                    ],
                },
                allow_val_change=True,
            )
            run.log(
                {
                    f"aggregate/{key}": value
                    for key, value in result.items()
                    if key.startswith(("mean_", "std_"))
                }
            )
            run.log(
                {
                    "rolling_origin/folds": wandb.Table(
                        dataframe=pd.DataFrame(
                            [
                                {
                                    "fold": row["fold"],
                                    "fit_rows": row["fit"]["row_count"],
                                    "evaluation_rows": row["evaluation"]["row_count"],
                                    "average_precision": row["average_precision"],
                                    "roc_auc": row["roc_auc"],
                                    "log_loss": row["log_loss"],
                                    "latency_p95_ms": row["latency_p95_ms"],
                                    "model_bytes": row["model_bytes"],
                                }
                                for row in result["folds"]
                            ]
                        )
                    )
                }
            )
        except RuntimeError as error:
            result["failure_reason"] = f"convergence: {error}"
            run.summary["status"] = "failed_convergence"
        finally:
            results.append(result)
            run.finish()
    ranked = rank_base_results(results)
    if len(ranked) < 2:
        raise RuntimeError("fewer than two base configurations completed")
    authorized = authorized_calibration_ids(ranked)
    report = {
        "git_sha": git.sha,
        "dataset_artifact": config["dataset_artifact"],
        "dataset_digest": config["dataset_digest"],
        "results": results,
        "ranking": [row["configuration_id"] for row in ranked],
        "authorized_calibration_ids": list(authorized),
    }
    _json(Path("artifacts/brief04/base_search_report.json"), report)
    return results, authorized


def _plain_calibration(audit: Any) -> dict[str, Any]:
    return {
        "mean_probability_gap": audit.mean_probability_gap,
        "equal_width_ece_10": audit.equal_width_ece_10,
        "equal_frequency_ece_15": audit.equal_frequency_ece_15,
        "equal_frequency_mce_15": audit.equal_frequency_mce_15,
        "equal_width_table_10": list(audit.equal_width_table_10),
        "equal_frequency_table_15": list(audit.equal_frequency_table_15),
    }


def _isotonic_steps(model: Any) -> int | None:
    try:
        calibrator = model.calibrated_classifiers_[0].calibrators[0]
        return len(calibrator.X_thresholds_)
    except (AttributeError, IndexError, TypeError):
        return None


def run_finalists(
    *,
    entity: str,
    project: str,
    config: dict[str, Any],
    authorized: tuple[str, ...],
    git: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Fit no more than six calibrated finalists and apply November gates."""

    if len(authorized) > 3:
        raise ValueError("more than three base configurations were authorized")
    finalists: list[dict[str, Any]] = []
    for config_id in authorized:
        for method in config["calibration_methods"]:
            finalist_id = f"{config_id}-{method}"
            parameters = config["configurations"][config_id]
            run = _init_run(
                entity=entity,
                project=project,
                name=f"brief04-finalist-{finalist_id.lower()}",
                tags=["brief04-calibrated-finalist"],
                config={
                    "finalist_id": finalist_id,
                    "configuration_id": config_id,
                    "parameters": parameters,
                    "calibration_method": method,
                    "dataset_artifact": config["dataset_artifact"],
                    "dataset_digest": config["dataset_digest"],
                    "git_sha": git.sha,
                    "git_dirty": git.dirty,
                    "calibration_period": "2025-11-01/2025-11-15",
                    "selection_period": "2025-11-16/2025-11-30",
                    "december_evaluated": False,
                    "final_test_evaluated": False,
                },
            )
            figures: dict[str, Any] = {}
            try:
                _log_code(run)
                root, artifact = _download_dataset(run, config)
                lineage = _lineage(root, artifact, config, git)
                train = _train_frame(root)
                november = _november_frame(root)
                partitions = partition_remediation_data(train, november)
                base, schema = build_remediation_model(config_id, parameters)
                validate_model_features(schema)
                _fit_without_convergence_warning(
                    base, partitions.final_fit.loc[:, schema], partitions.final_fit["target"]
                )
                selection_features = partitions.selection.loc[:, schema]
                validate_model_features(selection_features.columns)
                pre_probabilities = base.predict_proba(selection_features)[:, 1]
                pre_audit = calibration_audit(partitions.selection["target"], pre_probabilities)
                pre_metrics = {
                    "average_precision": float(
                        average_precision_score(partitions.selection["target"], pre_probabilities)
                    ),
                    "roc_auc": float(
                        roc_auc_score(partitions.selection["target"], pre_probabilities)
                    ),
                    "brier_score": float(
                        brier_score_loss(partitions.selection["target"], pre_probabilities)
                    ),
                    "log_loss": float(log_loss(partitions.selection["target"], pre_probabilities)),
                    "mean_probability_gap": pre_audit.mean_probability_gap,
                    "equal_frequency_ece_15": pre_audit.equal_frequency_ece_15,
                }
                calibrated = fit_calibrator(
                    base,
                    partitions.calibration.loc[:, schema],
                    partitions.calibration["target"],
                    method=method,
                )
                probabilities = calibrated.predict_proba(selection_features)[:, 1]
                threshold = select_threshold_remediation(
                    partitions.selection["target"], probabilities
                )
                evaluation = evaluate_binary(
                    partitions.selection["target"], probabilities, threshold=threshold.threshold
                )
                figures = evaluation.figures
                audit = calibration_audit(partitions.selection["target"], probabilities)
                metrics = {
                    **evaluation.metrics,
                    "prevalence": float(partitions.selection["target"].mean()),
                    "mean_probability_gap": audit.mean_probability_gap,
                    "lineage_verified": lineage["dataset_digest"] == config["dataset_digest"],
                    "schema_check_passed": True,
                    "leakage_check_passed": True,
                    "convergence_check_passed": True,
                }
                buffer = io.BytesIO()
                joblib.dump(calibrated, buffer)
                restored = joblib.load(io.BytesIO(buffer.getvalue()))
                restored_probabilities = restored.predict_proba(selection_features.iloc[:100])[:, 1]
                metrics["serialization_check_passed"] = bool(
                    np.allclose(probabilities[:100], restored_probabilities, rtol=0, atol=1e-12)
                )
                candidate_dir = Path("artifacts/brief04/finalists") / finalist_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(calibrated, candidate_dir / "model.joblib")
                threshold_evidence = {
                    "threshold": threshold.threshold,
                    "precision": threshold.precision,
                    "recall": threshold.recall,
                    "f1": threshold.f1,
                    "objective": "maximum F1 subject to recall >= 0.60",
                    "tie_breaks": [
                        "higher_f1",
                        "higher_precision",
                        "closest_to_0.5",
                        "lower_threshold",
                    ],
                    "evaluated_threshold_count": len(threshold.table),
                }
                calibration = {
                    **_plain_calibration(audit),
                    "method": method,
                    "probability_tie_count": int(
                        len(probabilities) - len(np.unique(probabilities))
                    ),
                    "isotonic_step_count": _isotonic_steps(calibrated),
                }
                period_prior = prior_scores(partitions.selection["target"])
                _json(candidate_dir / "feature_schema.json", {"features": list(schema)})
                _json(candidate_dir / "threshold.json", threshold_evidence)
                _json(candidate_dir / "calibration.json", calibration)
                _json(candidate_dir / "november_selection_metrics.json", metrics)
                _json(
                    candidate_dir / "metadata.json",
                    {
                        "finalist_id": finalist_id,
                        "configuration_id": config_id,
                        "parameters": parameters,
                        "lineage": lineage,
                        "partitions": {
                            "final_fit": _partition_summary(partitions.final_fit),
                            "calibration": _partition_summary(partitions.calibration),
                            "selection": _partition_summary(partitions.selection),
                        },
                        "versions": {
                            "python": platform.python_version(),
                            "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__,
                            "wandb": wandb.__version__,
                        },
                        "december_evaluated": False,
                        "final_test_evaluated": False,
                    },
                )
                core_bundle_bytes = sum(
                    path.stat().st_size for path in candidate_dir.iterdir() if path.is_file()
                )
                bundle_bytes = core_bundle_bytes + int(config["route_asset_estimate_bytes"])
                latency = measure_single_row_latency(
                    calibrated,
                    selection_features,
                    sample_count=int(config["latency_sample_count"]),
                )
                gates = november_gates(
                    metrics,
                    prior=period_prior,
                    ece=audit.equal_frequency_ece_15,
                    latency_p95_ms=float(latency["p95_ms"]),
                    bundle_bytes=bundle_bytes,
                )
                finalist = {
                    "finalist_id": finalist_id,
                    "configuration_id": config_id,
                    "calibration_method": method,
                    "parameters": parameters,
                    "feature_schema": list(schema),
                    "run_id": run.id,
                    "run_url": run.url,
                    "lineage": lineage,
                    "partitions": {
                        "final_fit": _partition_summary(partitions.final_fit),
                        "calibration": _partition_summary(partitions.calibration),
                        "selection": _partition_summary(partitions.selection),
                    },
                    "pre_calibration_metrics": pre_metrics,
                    "metrics": metrics,
                    "prior": period_prior,
                    "calibration": calibration,
                    "threshold": threshold_evidence,
                    "latency": latency,
                    "core_bundle_bytes": core_bundle_bytes,
                    "route_asset_estimate_bytes": int(config["route_asset_estimate_bytes"]),
                    "bundle_estimate_bytes": bundle_bytes,
                    "gates": gates,
                    "candidate_directory": str(candidate_dir),
                }
                finalists.append(finalist)
                run.config.update(
                    {
                        "feature_schema": list(schema),
                        "partitions": finalist["partitions"],
                        "threshold": threshold.threshold,
                        "november_gates": gates,
                    },
                    allow_val_change=True,
                )
                run.log({f"pre_calibration/{key}": value for key, value in pre_metrics.items()})
                run.log(
                    {
                        f"november/{key}": value
                        for key, value in metrics.items()
                        if isinstance(value, int | float | bool)
                    }
                )
                run.log(
                    {
                        f"calibration/{key}": value
                        for key, value in calibration.items()
                        if isinstance(value, int | float)
                    }
                )
                run.log({f"latency/{key}": value for key, value in latency.items()})
                run.log(
                    {
                        "calibration/equal_width_table_10": wandb.Table(
                            dataframe=pd.DataFrame(calibration["equal_width_table_10"])
                        ),
                        "calibration/equal_frequency_table_15": wandb.Table(
                            dataframe=pd.DataFrame(calibration["equal_frequency_table_15"])
                        ),
                    }
                )
                for name, figure in figures.items():
                    run.log({f"november_plot/{name}": wandb.Image(figure)})
                threshold_figure, axis = plt.subplots(figsize=(7, 5))
                threshold_frame = pd.DataFrame(threshold.table)
                for metric in ("precision", "recall", "f1"):
                    axis.plot(threshold_frame["threshold"], threshold_frame[metric], label=metric)
                axis.axvline(threshold.threshold, color="black", linestyle="--")
                axis.legend()
                threshold_figure.tight_layout()
                run.log({"november_plot/threshold_curve": wandb.Image(threshold_figure)})
                plt.close(threshold_figure)
            finally:
                for figure in figures.values():
                    plt.close(figure)
                run.finish()
    if len(finalists) > 6:
        raise ValueError("calibration matrix exceeded six finalists")
    try:
        winner = choose_remediation_winner(finalists)
    except ValueError:
        winner = None
    if winner is not None:
        winner_run = wandb.Api().run(f"{entity}/{project}/{winner['run_id']}")
        winner_run.tags = tuple(sorted(set(winner_run.tags) | {"brief04-november-candidate"}))
        winner_run.update()
    report = {
        "git_sha": git.sha,
        "dataset_artifact": config["dataset_artifact"],
        "dataset_digest": config["dataset_digest"],
        "authorized_calibration_ids": list(authorized),
        "finalists": finalists,
        "winner": winner["finalist_id"] if winner else None,
        "selection_status": "selected" if winner else "blocked",
        "stop_reason": None
        if winner
        else "no calibrated finalist passes every mandatory November gate",
        "december_evaluated": False,
        "final_test_evaluated": False,
    }
    _json(Path("artifacts/brief04/november_selection_report.json"), report)
    return finalists, winner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded remediation selection.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/remediation.yaml"))
    parser.add_argument("--wandb-mode", choices=("online",), default="online")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        entity = os.environ["WANDB_ENTITY"]
        project = os.getenv("WANDB_PROJECT", "us-flight-delay-mlops")
        git = git_provenance()
        if git.dirty:
            raise TrackingError("model remediation requires a clean worktree")
        config = _load_config(args.config)
        _, authorized = run_base_search(entity=entity, project=project, config=config, git=git)
        _, winner = run_finalists(
            entity=entity,
            project=project,
            config=config,
            authorized=authorized,
            git=git,
        )
        return 0 if winner else 2
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, TrackingError) as error:
        print(f"Model remediation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
