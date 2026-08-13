"""Aggregate training baselines and complete model-bundle serialization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_delay.data.manifest import canonical_json_bytes

MODEL_BUNDLE_FILES: frozenset[str] = frozenset(
    {
        "model.joblib",
        "feature_schema.json",
        "threshold.json",
        "training_baseline.json",
        "metrics.json",
        "metadata.json",
        "MODEL_CARD.md",
    }
)


@dataclass(frozen=True)
class ModelBundleResult:
    directory: Path
    byte_size: int
    model_load_ms: float
    loaded_model: Any


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError(f"numeric baseline {series.name} has no finite values")
    edges = np.unique(values.quantile(np.linspace(0, 1, 11)).to_numpy())
    if len(edges) == 1:
        edges = np.array([edges[0], edges[0] + 1.0])
    counts = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop").value_counts(
        sort=False, normalize=True
    )
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
        "bin_edges": [float(value) for value in edges],
        "bin_proportions": [float(value) for value in counts.to_numpy()],
    }


def _categorical_summary(series: pd.Series, *, limit: int = 20) -> dict[str, float]:
    normalized = series.astype("string").fillna("__MISSING__").astype(str)
    frequencies = normalized.value_counts(normalize=True)
    top = frequencies.iloc[:limit]
    result = {str(category): float(value) for category, value in top.items()}
    result["__OTHER__"] = float(frequencies.iloc[limit:].sum())
    return result


def build_training_baseline(
    train: pd.DataFrame, *, dataset_artifact: str, dataset_digest: str
) -> dict[str, Any]:
    """Create aggregate-only monitoring reference metadata from training rows."""

    numeric_columns = {
        "distance": "Distance",
        "scheduled_elapsed_time": "CRSElapsedTime",
        "scheduled_departure_hour": "scheduled_departure_hour",
    }
    categorical_columns = {
        "carrier": "Reporting_Airline",
        "origin": "Origin",
        "destination": "Dest",
        "month": "Month",
    }
    return {
        "row_count": len(train),
        "target_prevalence": float(train["target"].mean()),
        "dataset_artifact": dataset_artifact,
        "dataset_digest": dataset_digest,
        "numeric": {
            name: _numeric_summary(train[column]) for name, column in numeric_columns.items()
        },
        "categorical": {
            name: _categorical_summary(train[column])
            for name, column in categorical_columns.items()
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def write_model_bundle(
    *,
    directory: Path,
    model: Any,
    feature_schema: list[str],
    threshold: float,
    training_baseline: dict[str, Any],
    metrics: dict[str, float | int],
    metadata: dict[str, Any],
) -> ModelBundleResult:
    """Write and reload the exact seven-file baseline model bundle."""

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    _write_json(directory / "feature_schema.json", {"features": feature_schema})
    _write_json(directory / "threshold.json", {"threshold": threshold})
    _write_json(directory / "training_baseline.json", training_baseline)
    _write_json(directory / "metrics.json", {"split": "validation", "metrics": metrics})
    _write_json(directory / "metadata.json", metadata)
    card = (
        f"# {metadata['candidate_id']} model card\n\n"
        "This bundle was trained on the declared training split and evaluated on validation "
        "only. Final-test evaluation has not occurred. W&B Registry promotion has not occurred.\n"
    )
    (directory / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != MODEL_BUNDLE_FILES:
        raise ValueError(f"model bundle files differ from contract: {sorted(observed)}")
    load_start = time.perf_counter_ns()
    loaded = joblib.load(directory / "model.joblib")
    load_ms = (time.perf_counter_ns() - load_start) / 1_000_000
    byte_size = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    return ModelBundleResult(directory, byte_size, load_ms, loaded)
