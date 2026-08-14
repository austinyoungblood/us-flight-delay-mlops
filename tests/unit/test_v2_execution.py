from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from flight_delay.modeling.v2.execution import (
    DECISION_PATH,
    DEVELOPMENT_MARKER,
    QUALIFICATION_MARKER,
    QUALIFICATION_RESULT,
    STATE_PATH,
    WINNER_LOCK,
    WINNER_MODEL,
    V2ExecutionError,
    _atomic_json,
    create_marker,
    preflight,
    require_merged_applied_state,
    run_december_apply,
    run_development_apply,
    update_marker,
    validate_december_handoff,
    validate_dependency_isolation,
)
from flight_delay.modeling.v2.features import build_historical_state, transform_with_state
from flight_delay.modeling.v2.protocol import sha256_file
from flight_delay.modeling.v2.tracking import NullTracker

ROOT = Path(__file__).resolve().parents[2]


class FrozenSyntheticModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.where(pd.to_numeric(features["DayofMonth"]).isin((9, 25)), 0.98, 0.02)
        return np.column_stack((1.0 - positive, positive))


def test_preflight_is_offline_dry_and_preserves_production() -> None:
    development = preflight(ROOT, stage="development")
    qualification = preflight(ROOT, stage="qualification")
    assert development["mode"] == "dry-run/preflight"
    assert development["parquet_opened"] is False
    assert development["december_opened"] is False
    assert development["historical_test_accessed"] is False
    assert development["network_contacted"] is False
    assert development["aws_contacted"] is False
    assert development["registry_mutated"] is False
    assert development["production_v0"]["unchanged"] is True
    assert development["lightgbm_candidate_count"] == 16
    assert development["catboost_candidate_count"] == 12
    assert qualification["requires_frozen_november_winner"] is True
    assert qualification["refitting_permitted"] is False
    assert qualification["historical_state_update_permitted"] is False


def test_preflight_refuses_forbidden_runtime_imports_in_isolated_process() -> None:
    code = f"""
import importlib.abc
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'boto3','botocore','wandb','lightgbm','catboost'}}:
            raise RuntimeError('forbidden import: ' + fullname)
        return None
import sys
sys.meta_path.insert(0, Block())
from flight_delay.modeling.v2.execution import preflight
report = preflight(Path({str(ROOT)!r}), stage='development')
assert report['parquet_opened'] is False
assert report['december_opened'] is False
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dependency_isolation_rejects_runtime_image_contamination(tmp_path: Path) -> None:
    (tmp_path / "services/api").mkdir(parents=True)
    (tmp_path / "services/user_ui").mkdir(parents=True)
    (tmp_path / "services/monitor_ui").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n[project.optional-dependencies]\n'
        'v2=["lightgbm==4.7.0","catboost==1.2.10"]\n'
    )
    (tmp_path / "requirements-v2.lock").write_text("lightgbm==4.7.0\ncatboost==1.2.10\n")
    for name in ("api", "user_ui", "monitor_ui"):
        (tmp_path / f"services/{name}/Dockerfile").write_text("RUN pip install .\n")
    assert validate_dependency_isolation(tmp_path)["runtime_images_install_base_only"] is True
    (tmp_path / "services/api/Dockerfile").write_text("RUN pip install '.[v2]'\n")
    with pytest.raises(V2ExecutionError, match="modeling-only"):
        validate_dependency_isolation(tmp_path)


def test_merged_state_and_marker_guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def clean_git(_root: Path, *arguments: str) -> str:
        responses = {
            ("status", "--porcelain"): "",
            ("branch", "--show-current"): "main",
            ("merge-base", "--is-ancestor", "226ddd2c279cc4dd087ce3d2daab64c7aad1682c", "HEAD"): "",
            ("rev-parse", "HEAD"): "implementation",
        }
        return responses[arguments]

    monkeypatch.setattr("flight_delay.modeling.v2.execution._git", clean_git)
    assert require_merged_applied_state(tmp_path) == "implementation"
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution._git",
        lambda _root, *arguments: "dirty" if arguments == ("status", "--porcelain") else "",
    )
    with pytest.raises(V2ExecutionError, match="clean Git"):
        require_merged_applied_state(tmp_path)

    marker = tmp_path / "marker.json"
    create_marker(marker, {"status": "started"})
    update_marker(marker, {"status": "complete"})
    assert json.loads(marker.read_text())["status"] == "complete"
    with pytest.raises(V2ExecutionError, match="already exists"):
        create_marker(marker, {})


def test_applied_development_state_machine_with_synthetic_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_v2_frame: pd.DataFrame,
) -> None:
    january = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.eq(1)
    ]
    state = build_historical_state(january, as_of="2025-01-31")
    prepared = SimpleNamespace(
        november_state=state,
        lineage={"november_state_sha256": state.sha256},
        raw_train=pd.DataFrame(),
        raw_november=pd.DataFrame(),
        search=None,
    )
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.require_merged_applied_state",
        lambda _root: "implementation",
    )
    monkeypatch.setattr("flight_delay.modeling.v2.execution.preflight", lambda *_a, **_k: {})
    monkeypatch.setattr("flight_delay.modeling.v2.execution.require_versions", lambda: {})
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.load_and_validate_v2_protocol",
        lambda *_a, **_k: ({"protocol_id": "v2"}, {}, "protocol"),
    )
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.prepare_development_data",
        lambda _root: prepared,
    )
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution._reconstruct_r3",
        lambda *_a: {"reproduction": {"all_metrics_reproduced": True}},
    )
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.run_screening_and_cpu_confirmation",
        lambda **_k: {
            "screening": [],
            "cpu_confirmation": [],
            "screening_cpu_differences": [],
            "advanced_to_refit": [],
        },
    )
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.run_refit_and_november",
        lambda **_k: {
            "decision": "governed_stop",
            "winner": None,
            "finalists": [],
            "production_remains": "v0",
            "stopped_before_december": True,
        },
    )
    monkeypatch.setattr("flight_delay.modeling.v2.execution._online_tracker", lambda: NullTracker())
    result = run_development_apply(tmp_path, tracking="online")
    assert result["decision"] == "governed_stop"
    assert (tmp_path / DECISION_PATH).is_file()
    assert (tmp_path / STATE_PATH).read_bytes() == state.to_bytes()
    marker = json.loads((tmp_path / DEVELOPMENT_MARKER).read_text())
    assert marker["status"] == "complete"
    assert marker["decision"] == "governed_stop"
    assert not (tmp_path / WINNER_MODEL).exists()


def test_december_handoff_and_evaluation_use_frozen_synthetic_artifacts_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_v2_frame: pd.DataFrame,
    v2_protocol: dict[str, Any],
) -> None:
    history = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.le(10)
    ]
    state = build_historical_state(history, as_of="2025-10-31")
    november = synthetic_v2_frame.loc[
        pd.to_datetime(synthetic_v2_frame["flight_date"]).dt.month.eq(11)
    ]
    selection = november.loc[pd.to_datetime(november["flight_date"]).dt.day.ge(16)]
    features = transform_with_state(selection, state)
    target = selection["target"]
    (tmp_path / STATE_PATH).parent.mkdir(parents=True)
    (tmp_path / STATE_PATH).write_bytes(state.to_bytes())
    model = FrozenSyntheticModel()
    joblib.dump(model, tmp_path / WINNER_MODEL)
    _atomic_json(
        tmp_path / DEVELOPMENT_MARKER,
        {"status": "complete", "decision": "winner"},
        refuse_existing=True,
    )
    _atomic_json(
        tmp_path / WINNER_LOCK,
        {
            "december_evaluated": False,
            "historical_state_as_of": "2025-10-31",
            "historical_state_sha256": state.sha256,
            "model_sha256": sha256_file(tmp_path / WINNER_MODEL),
            "implementation_git_sha": "implementation",
            "finalist_id": "CB01-none",
            "threshold": 0.5,
        },
        refuse_existing=True,
    )
    winner, restored = validate_december_handoff(tmp_path)
    assert winner["finalist_id"] == "CB01-none"
    assert restored.sha256 == state.sha256

    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.require_merged_applied_state",
        lambda _root: "implementation",
    )
    monkeypatch.setattr("flight_delay.modeling.v2.execution.preflight", lambda *_a, **_k: {})
    monkeypatch.setattr("flight_delay.modeling.v2.execution.require_versions", lambda: {})
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.load_and_validate_v2_protocol",
        lambda *_a, **_k: (v2_protocol, {}, "protocol"),
    )
    monkeypatch.setattr(
        "flight_delay.modeling.v2.execution.load_december_features",
        lambda *_a, **_k: (features, target, selection["flight_date"]),
    )
    monkeypatch.setattr("flight_delay.modeling.v2.execution._online_tracker", lambda: NullTracker())
    result = run_december_apply(tmp_path, tracking="online")
    assert result["passed"] is True
    assert result["same_frozen_november_model"] is True
    assert result["same_frozen_october_31_state"] is True
    assert result["production_remains"] == "v0"
    assert (tmp_path / QUALIFICATION_RESULT).is_file()
    assert json.loads((tmp_path / QUALIFICATION_MARKER).read_text())["status"] == "complete"
