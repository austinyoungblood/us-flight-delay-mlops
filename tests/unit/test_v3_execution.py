"""V3 preflight, dependency isolation, runtime estimation, and durable markers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flight_delay.modeling.v3.execution import (
    DEVELOPMENT_MARKER,
    PROTOCOL_COMMIT_SHA,
    QUALIFICATION_MARKER,
    V3ExecutionError,
    create_marker,
    estimate_applied_runtime,
    preflight,
    run_december_apply,
    run_development_apply,
    update_marker,
    validate_dependency_isolation,
    validate_production_v0,
)
from flight_delay.modeling.v3.protocol import (
    CANDIDATE_IDENTITY_IDS,
    load_and_validate_v3_protocol,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def protocol() -> dict:
    payload, _lock, _sha = load_and_validate_v3_protocol(
        ROOT / "configs/v3_experiment_protocol.yaml",
        lock_path=ROOT / "experiments/v3/protocol_lock.json",
        repository_root=ROOT,
    )
    return payload


def test_development_preflight_opens_no_data_and_touches_no_production() -> None:
    report = preflight(ROOT, stage="development")
    assert report["mode"] == "dry-run/preflight"
    assert report["parquet_opened"] is False
    assert report["december_opened"] is False
    assert report["historical_test_accessed"] is False
    assert report["january_may_2026_accessed"] is False
    assert report["network_contacted"] is False
    assert report["production_v0_mutated"] is False
    assert report["registry_mutated"] is False
    assert report["aws_contacted"] is False
    assert report["production_v0"]["unchanged"] is True


def test_preflight_reports_the_frozen_campaign_shape() -> None:
    report = preflight(ROOT, stage="development")
    assert report["candidate_identity_count"] == 8
    assert report["candidate_identities"] == list(CANDIDATE_IDENTITY_IDS)
    assert report["weight_policies"] == ["UNIFORM", "EXPONENTIAL_120D"]
    assert report["total_feature_count"] == 48
    assert report["native_categorical_count"] == 8
    assert report["finalist_count"] == 15
    assert report["november_state_as_of"] == "2025-10-31"
    assert report["stops_before_december"] is True


def test_qualification_preflight_forbids_every_mutation() -> None:
    report = preflight(ROOT, stage="qualification")
    assert report["refitting_permitted"] is False
    assert report["recalibration_permitted"] is False
    assert report["threshold_change_permitted"] is False
    assert report["historical_state_update_permitted"] is False
    assert report["candidate_switching_permitted"] is False
    assert report["requires_frozen_november_winner"] is True


def test_v3_adds_no_modeling_dependency_and_runtime_images_stay_clean() -> None:
    isolation = validate_dependency_isolation(ROOT)
    assert isolation["v3_added_modeling_dependency"] is False
    assert isolation["runtime_images_install_base_only"] is True
    assert isolation["modeling_extra"] == ["lightgbm==4.7.0", "catboost==1.2.10"]


def test_runtime_images_never_mention_a_modeling_package() -> None:
    for name in ("api", "user_ui", "monitor_ui"):
        source = (ROOT / f"services/{name}/Dockerfile").read_text(encoding="utf-8").casefold()
        for token in ("catboost", "lightgbm", "requirements-v2", "requirements-v3", ".[v3]"):
            assert token not in source


def test_production_v0_is_still_pinned(protocol: dict) -> None:
    observed = validate_production_v0(ROOT, protocol)
    assert observed["unchanged"] is True
    assert observed["release"]["registry_version"] == "v0"
    assert observed["deployment"]["registry_version"] == "v0"


def _manifest(month_rows: int = 600000) -> dict:
    months = [f"2024-{index:02d}" for index in range(1, 13)]
    months += [f"2025-{index:02d}" for index in range(1, 12)]
    return {
        "monthly_counts": [{"month": month, "model_eligible_rows": month_rows} for month in months]
    }


def test_runtime_estimate_applies_the_search_cap(protocol: dict) -> None:
    estimate = estimate_applied_runtime(protocol, _manifest())
    assert estimate["search_rows_per_month_cap"] == 50000
    # FOLD_1 fits 18 capped months (2024-02 through 2025-07).
    assert estimate["fold_rows"][0]["search_fit_rows"] == 18 * 50000
    assert estimate["fold_rows"][3]["search_fit_rows"] == 21 * 50000
    assert estimate["search_fit_rows_per_candidate"] == sum(
        row["search_fit_rows"] for row in estimate["fold_rows"]
    )


def test_runtime_estimate_uses_uncapped_rows_for_the_authoritative_refit(protocol: dict) -> None:
    estimate = estimate_applied_runtime(protocol, _manifest())
    # The refit spans 2024-02 through 2025-10: 21 uncapped months.
    assert estimate["full_refit_rows"] == 21 * 600000
    assert estimate["advisory_only"] is True


def test_runtime_estimate_covers_every_governed_stage(protocol: dict) -> None:
    estimate = estimate_applied_runtime(protocol, _manifest())
    stages = {row["stage"]: row for row in estimate["stages"]}
    assert set(stages) == {
        "screening_lightgbm_cpu",
        "screening_catboost_gpu",
        "cpu_confirmation_lightgbm",
        "cpu_confirmation_catboost",
        "full_refit_lightgbm_cpu",
        "full_refit_catboost_cpu",
    }
    assert stages["screening_lightgbm_cpu"]["identities"] == 4
    assert stages["cpu_confirmation_catboost"]["identities"] == 2
    assert stages["full_refit_catboost_cpu"]["identities"] == 1
    assert estimate["estimated_total_seconds"] > 0
    # Hours are reported rounded to two decimals for readability.
    assert estimate["estimated_total_hours"] == pytest.approx(
        estimate["estimated_total_seconds"] / 3600.0, abs=0.005
    )


def test_runtime_estimate_scales_with_rows(protocol: dict) -> None:
    small = estimate_applied_runtime(protocol, _manifest(month_rows=100000))
    large = estimate_applied_runtime(protocol, _manifest(month_rows=900000))
    assert large["estimated_total_seconds"] > small["estimated_total_seconds"]


def test_runtime_estimate_requires_row_counts(protocol: dict) -> None:
    with pytest.raises(V3ExecutionError, match="monthly row counts"):
        estimate_applied_runtime(protocol, {"monthly_counts": []})


def test_applied_runs_require_online_tracking(tmp_path: Path) -> None:
    with pytest.raises(V3ExecutionError, match="online governed tracking"):
        run_development_apply(tmp_path, tracking="disabled")
    with pytest.raises(V3ExecutionError, match="online governed tracking"):
        run_december_apply(tmp_path, tracking="disabled")


def test_applied_execution_is_locked_to_the_frozen_protocol_commit() -> None:
    assert len(PROTOCOL_COMMIT_SHA) == 40
    assert PROTOCOL_COMMIT_SHA.isalnum()


def test_markers_are_write_once_then_updatable(tmp_path: Path) -> None:
    marker = tmp_path / DEVELOPMENT_MARKER
    create_marker(marker, {"status": "started"})
    with pytest.raises(V3ExecutionError, match="already exists"):
        create_marker(marker, {"status": "started"})
    update_marker(marker, {"status": "complete", "decision": "governed_stop"})
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["decision"] == "governed_stop"


def test_marker_paths_are_v3_scoped_and_never_touch_v1_or_v2() -> None:
    for path in (DEVELOPMENT_MARKER, QUALIFICATION_MARKER):
        assert str(path).startswith("artifacts/v3/")


def test_unreadable_marker_is_refused(tmp_path: Path) -> None:
    broken = tmp_path / "marker.json"
    broken.write_text("not json", encoding="utf-8")
    with pytest.raises(V3ExecutionError, match="cannot read governed state"):
        update_marker(broken, {"status": "complete"})
