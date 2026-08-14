from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_marks_v1_as_precommitted_untrained_and_keeps_v0_production() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "v1 model experiment protocol" in readme
    assert "V1 has **not** been trained" in readme
    assert "production:v0" in readme


def test_human_protocol_records_primary_alternative_and_holdout_boundaries() -> None:
    protocol = (ROOT / "docs/v1-model-experiment-protocol.md").read_text(encoding="utf-8")
    for required in (
        "catboost==1.2.10",
        "Prophet was considered but intentionally not selected",
        "No v1 model has been trained",
        "January 1-May 31, 2026 historical final test",
        "first complete DOT/BTS Reporting",
        "Carrier On-Time Performance month",
        "production:v0",
    ):
        assert required in protocol


def test_protocol_lock_records_pre_result_state_and_immutable_incumbent() -> None:
    lock = json.loads((ROOT / "experiments/v1/protocol_lock.json").read_text(encoding="utf-8"))
    assert lock["training_started"] is False
    assert lock["wandb_runs_created"] is False
    assert lock["fresh_final_accessed"] is False
    assert lock["incumbent_registry_version"] == "v0"
    assert lock["historical_test_consumed"] is True


def test_implementation_document_preserves_unexecuted_and_v0_boundaries() -> None:
    implementation = (ROOT / "docs/v1-model-experiment-implementation.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "PROTOCOL MERGED",
        "IMPLEMENTATION UNDER REVIEW",
        "REAL v1 TRAINING NOT YET RUN",
        "PRODUCTION STILL v0",
        'find_spec("catboost") is None',
        "R3-base rolling evidence",
        "governed R3-sigmoid reconstruction",
        "explicit November predicate",
        "explicit December predicate",
    ):
        assert required in implementation
