"""BLOCKER 1: the R3 control must reconstruct from the canonical v1/v2 dataset, never from v3.

R3 is a control check on the frozen incumbent, so it has to run on the exact data that historically
produced the frozen R3 metrics. The v3 population is a different year range at a different sampling
density; scoring the incumbent on it would compare against numbers it was never measured on.

No real R3 fit and no real v3 fit occur here: both the loader and the reconstructor are injected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from flight_delay.modeling.v1_data import V1DataGuardError, require_allowed_v1_path
from flight_delay.modeling.v3.execution import reconstruct_r3_control
from tests.conftest import make_v3_frame

ROOT = Path(__file__).resolve().parents[2]

CANONICAL_TRAIN = "data/processed/train.parquet"
CANONICAL_VALIDATION = "data/processed/validation.parquet"


class ControlData:
    """Stand-in for the canonical v1 DevelopmentData."""

    def __init__(self) -> None:
        self.train = pd.DataFrame({"marker": ["canonical-v1-train"], "flight_date": ["2025-01-01"]})
        self.november = pd.DataFrame(
            {"marker": ["canonical-v1-november"], "flight_date": ["2025-11-01"]}
        )
        self.manifest = {"manifest_digest": "c" * 64}
        self.protocol_sha256 = "a" * 64


@pytest.fixture
def recorder() -> dict[str, Any]:
    return {"loader_calls": [], "reconstructor_frames": []}


def make_loader(recorder: dict[str, Any], control: ControlData):
    def loader(repository_root: Path) -> ControlData:
        recorder["loader_calls"].append(repository_root)
        return control

    return loader


def make_reconstructor(recorder: dict[str, Any], *, reproduced: bool = True):
    def reconstructor(train: pd.DataFrame, november: pd.DataFrame) -> dict[str, Any]:
        recorder["reconstructor_frames"].append((train, november))
        if not reproduced:
            raise RuntimeError("R3 reconstruction gate blocked all challengers")
        return {"reproduction": {"all_metrics_reproduced": True}, "metrics": {"precision": 0.28}}

    return reconstructor


def test_the_canonical_v1_control_loader_is_invoked(recorder: dict[str, Any]) -> None:
    control = ControlData()
    reconstruct_r3_control(
        ROOT,
        loader=make_loader(recorder, control),
        reconstructor=make_reconstructor(recorder),
    )
    assert len(recorder["loader_calls"]) == 1
    assert recorder["loader_calls"][0] == ROOT


def test_the_canonical_v1_train_and_november_control_is_reconstructed(
    recorder: dict[str, Any],
) -> None:
    control = ControlData()
    reconstruct_r3_control(
        ROOT,
        loader=make_loader(recorder, control),
        reconstructor=make_reconstructor(recorder),
    )
    train, november = recorder["reconstructor_frames"][0]
    assert train["marker"].iloc[0] == "canonical-v1-train"
    assert november["marker"].iloc[0] == "canonical-v1-november"
    assert train is control.train
    assert november is control.november


def test_v3_raw_history_is_never_passed_into_r3_reconstruction(recorder: dict[str, Any]) -> None:
    """The defect being fixed: v3 frames must never reach the incumbent control."""

    v3_history = make_v3_frame(start="2024-01-01", end="2024-03-31")
    v3_november = make_v3_frame(start="2025-11-01", end="2025-11-30")
    control = ControlData()
    reconstruct_r3_control(
        ROOT,
        loader=make_loader(recorder, control),
        reconstructor=make_reconstructor(recorder),
    )
    train, november = recorder["reconstructor_frames"][0]
    for v3_frame in (v3_history, v3_november):
        assert train is not v3_frame
        assert november is not v3_frame
    # A v3 frame is uncapped and spans 2024; the control frames are neither.
    assert len(train) != len(v3_history)
    assert pd.to_datetime(train["flight_date"]).min() >= pd.Timestamp("2025-01-01")


def test_lineage_separates_the_control_and_challenger_datasets(recorder: dict[str, Any]) -> None:
    _result, lineage = reconstruct_r3_control(
        ROOT,
        loader=make_loader(recorder, ControlData()),
        reconstructor=make_reconstructor(recorder),
    )
    assert lineage["r3_control_dataset_manifest_digest"] == "c" * 64
    assert lineage["r3_control_sources"] == [CANONICAL_TRAIN, CANONICAL_VALIDATION]
    assert lineage["r3_control_used_v3_population"] is False
    assert lineage["historical_test_accessed"] is False
    assert lineage["r3_control_train_rows"] == 1
    assert lineage["r3_control_november_rows"] == 1


def test_the_two_dataset_digests_are_recorded_separately() -> None:
    """The control digest is the v0/v1/v2 manifest; the challenger digest is the v3 manifest."""

    import json

    control_digest = json.loads(
        (ROOT / "data/manifests/processed_manifest.json").read_text(encoding="utf-8")
    )["manifest_digest"]
    v3_digest = json.loads(
        (ROOT / "data/manifests/v3_processed_manifest.json").read_text(encoding="utf-8")
    )["manifest_digest"]
    assert control_digest != v3_digest


def test_a_failed_r3_control_still_blocks_challenger_trust(recorder: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="blocked all challengers"):
        reconstruct_r3_control(
            ROOT,
            loader=make_loader(recorder, ControlData()),
            reconstructor=make_reconstructor(recorder, reproduced=False),
        )


def test_the_control_loader_refuses_the_sealed_test_split() -> None:
    """The canonical v1 guard the control path relies on rejects test.parquet outright."""

    with pytest.raises(V1DataGuardError, match="test.parquet access is prohibited"):
        require_allowed_v1_path(ROOT, ROOT / "data/processed/test.parquet")


def test_the_control_loader_never_reaches_december() -> None:
    """The v1 control window stops at 2025-12-01, so December cannot enter the control."""

    import inspect

    from flight_delay.modeling.v1_data import load_development_data

    source = inspect.getsource(load_development_data)
    assert 'end_exclusive="2025-12-01"' in source
    assert "2026-01-01" not in source


def test_applied_development_calls_the_control_not_the_v3_population() -> None:
    """The applied path must pass the repository root, never a prepared v3 frame."""

    import inspect

    from flight_delay.modeling.v3.execution import run_development_apply

    source = inspect.getsource(run_development_apply)
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    assert "reconstruct_r3_control(root)" in code
    # The v3 challenger frames must not appear in executable code, only in commentary.
    assert "prepared.raw_history" not in code
    assert "prepared.raw_november" not in code
