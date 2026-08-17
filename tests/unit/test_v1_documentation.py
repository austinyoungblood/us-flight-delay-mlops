from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_links_governed_results_and_keeps_v0_production() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/v1-model-experiment-result.md" in readme
    assert "docs/v2-model-experiment-result.md" in readme
    assert "all 12 November finalists had no eligible threshold" in readme
    assert "production:v0" in readme


def test_human_protocol_records_primary_alternative_and_holdout_boundaries() -> None:
    protocol = (ROOT / "docs/v1-model-experiment-protocol.md").read_text(encoding="utf-8")
    for required in (
        "catboost==1.2.10",
        "Prophet was considered but intentionally not selected",
        "At protocol lock time, no v1 model had been trained",
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


def test_implementation_document_records_stop_and_v0_boundaries() -> None:
    implementation = (ROOT / "docs/v1-model-experiment-implementation.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "PROTOCOL MERGED",
        "DEVELOPMENT EXECUTION COMPLETE",
        "DECEMBER NOT OPENED",
        "status=no_eligible_threshold",
        "PRODUCTION STILL v0",
        'find_spec("catboost") is None',
        "R3-base rolling evidence",
        "governed R3-sigmoid reconstruction",
        "explicit November predicate",
        "explicit December predicate",
    ):
        assert required in implementation
