"""Uncapped v3 dataset preparation and its December authorization boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from flight_delay.data.download import YearMonth
from flight_delay.data.prepare_v3 import (
    DECEMBER_AUTHORIZATION,
    DECEMBER_SPLIT,
    HISTORY_SPLIT,
    NOVEMBER_SPLIT,
    SPLIT_BOUNDARIES,
    V3PreparationError,
    prepare_v3_dataset,
    split_for_month,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        (2024, 1, HISTORY_SPLIT),
        (2024, 12, HISTORY_SPLIT),
        (2025, 1, HISTORY_SPLIT),
        (2025, 10, HISTORY_SPLIT),
        (2025, 11, NOVEMBER_SPLIT),
    ],
)
def test_months_route_to_their_split(year: int, month: int, expected: str) -> None:
    assert split_for_month(YearMonth(year, month), include_december=False) == expected


def test_december_is_skipped_unless_authorized() -> None:
    december = YearMonth(2025, 12)
    assert split_for_month(december, include_december=False) is None
    assert split_for_month(december, include_december=True) == DECEMBER_SPLIT


def test_2026_is_never_routable() -> None:
    for month in (1, 3, 5):
        with pytest.raises(V3PreparationError, match="prohibited"):
            split_for_month(YearMonth(2026, month), include_december=True)


def test_split_boundaries_are_contiguous_and_month_aligned() -> None:
    assert SPLIT_BOUNDARIES[HISTORY_SPLIT] == ("2024-01-01", "2025-11-01")
    assert SPLIT_BOUNDARIES[NOVEMBER_SPLIT] == ("2025-11-01", "2025-12-01")
    assert SPLIT_BOUNDARIES[DECEMBER_SPLIT] == ("2025-12-01", "2026-01-01")


def test_a_wrong_december_authorization_is_refused(tmp_path: Path) -> None:
    with pytest.raises(V3PreparationError, match="exact qualification authorization"):
        prepare_v3_dataset(
            tmp_path,
            source_manifest_path=REPOSITORY_ROOT / "data/manifests/v3_source_manifest.json",
            december_authorization="please",
        )


def test_the_authorization_constant_is_explicit() -> None:
    assert DECEMBER_AUTHORIZATION == "december-2025-qualification-authorized"


def test_preparation_refuses_a_manifest_containing_2026(tmp_path: Path) -> None:
    import json

    from flight_delay.data.manifest import write_manifest

    source = json.loads(
        (REPOSITORY_ROOT / "data/manifests/v3_source_manifest.json").read_text(encoding="utf-8")
    )
    source["files"].append({**source["files"][0], "year": 2026, "month": 1})
    manifest_path = tmp_path / "tainted_source_manifest.json"
    write_manifest(manifest_path, {k: v for k, v in source.items() if k != "manifest_digest"})
    with pytest.raises(V3PreparationError, match="2026"):
        prepare_v3_dataset(tmp_path, source_manifest_path=manifest_path)


def test_committed_v3_source_manifest_has_no_2026_archive() -> None:
    import json

    manifest = json.loads(
        (REPOSITORY_ROOT / "data/manifests/v3_source_manifest.json").read_text(encoding="utf-8")
    )
    assert all(int(record["year"]) < 2026 for record in manifest["files"])
    assert len(manifest["files"]) == 24
