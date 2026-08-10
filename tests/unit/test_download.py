from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from flight_delay.data.download import (
    ARCHIVE_PREFIX,
    ChecksumConflictError,
    DownloadError,
    SourceFileRecord,
    YearMonth,
    archive_filename,
    archive_url,
    download_archive,
    inclusive_month_range,
    inspect_zip,
    source_manifest_payload,
)
from flight_delay.data.manifest import (
    canonical_json_bytes,
    validate_manifest,
    with_manifest_digest,
)


class FakeResponse:
    def __init__(self, body: bytes, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code
        self.closed = False

    def iter_content(self, chunk_size: int):  # type: ignore[no-untyped-def]
        for offset in range(0, len(self.body), max(1, chunk_size // 2)):
            yield self.body[offset : offset + max(1, chunk_size // 2)]

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, tuple[float, float]]] = []

    def get(self, url: str, *, stream: bool, timeout: tuple[float, float]):  # type: ignore[no-untyped-def]
        assert stream is True
        self.calls.append((url, timeout))
        return self.responses.pop(0)


def zip_bytes(*members: tuple[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return buffer.getvalue()


def test_inclusive_range_constructs_exact_17_reporting_carrier_urls() -> None:
    months = inclusive_month_range(YearMonth(2025, 1), YearMonth(2026, 5))
    urls = [archive_url(month) for month in months]
    assert len(months) == 17
    assert urls[0].endswith(f"{ARCHIVE_PREFIX}_2025_1.zip")
    assert urls[-1].endswith(f"{ARCHIVE_PREFIX}_2026_5.zip")
    assert all("Reporting_Carrier" in url for url in urls)
    assert all("Marketing_Carrier" not in url for url in urls)


def test_invalid_month_ranges_are_rejected() -> None:
    with pytest.raises(DownloadError, match="after end"):
        inclusive_month_range(YearMonth(2026, 1), YearMonth(2025, 1))
    with pytest.raises(DownloadError, match="YYYY-MM"):
        YearMonth.parse("2025-1")
    with pytest.raises(DownloadError, match="invalid BTS"):
        YearMonth(2025, 13)


def test_streamed_checksum_and_atomic_rename(tmp_path: Path) -> None:
    body = zip_bytes(("data.csv", b"a,b\n1,2\n"), ("README.txt", b"source notes"))
    response = FakeResponse(body)
    result = download_archive(
        YearMonth(2025, 1),
        tmp_path,
        client=FakeClient(response),
        chunk_size=4,
        sleep=lambda _: None,
    )
    final_path = tmp_path / archive_filename(YearMonth(2025, 1))
    assert result.status == "downloaded"
    assert result.record is not None
    assert result.record.sha256 == hashlib.sha256(body).hexdigest()
    assert final_path.read_bytes() == body
    assert not final_path.with_name(f"{final_path.name}.part").exists()
    assert response.closed is True


@pytest.mark.parametrize("body", [b"<!doctype html><title>Error</title>", b"PK\x03\x04broken"])
def test_html_and_corrupt_zip_are_rejected(tmp_path: Path, body: bytes) -> None:
    with pytest.raises(DownloadError):
        download_archive(
            YearMonth(2025, 1),
            tmp_path,
            client=FakeClient(FakeResponse(body)),
            sleep=lambda _: None,
        )
    assert not any(tmp_path.iterdir())


def test_zip_member_selection_excludes_documentation(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixture.zip"
    archive_path.write_bytes(
        zip_bytes(("nested/data.csv", b"x\n1\n"), ("documentation.csv", b"notes\n"))
    )
    inspection = inspect_zip(archive_path)
    assert inspection.selected_csv_member == "nested/data.csv"
    assert inspection.zip_members == ("documentation.csv", "nested/data.csv")


def test_idempotent_skip_and_checksum_conflict(tmp_path: Path) -> None:
    month = YearMonth(2025, 1)
    body = zip_bytes(("data.csv", b"a\n1\n"))
    first = download_archive(month, tmp_path, client=FakeClient(FakeResponse(body)))
    assert first.record is not None
    skipped = download_archive(
        month,
        tmp_path,
        client=FakeClient(),
        expected_record=first.record.to_manifest_dict(),
    )
    assert skipped.status == "skipped"
    (tmp_path / archive_filename(month)).write_bytes(b"changed")
    with pytest.raises(ChecksumConflictError, match="conflicts"):
        download_archive(
            month,
            tmp_path,
            client=FakeClient(),
            expected_record=first.record.to_manifest_dict(),
        )


def test_source_manifest_digest_is_canonical_and_stable() -> None:
    record = SourceFileRecord(
        year=2025,
        month=1,
        url=archive_url(YearMonth(2025, 1)),
        archive_filename=archive_filename(YearMonth(2025, 1)),
        byte_size=100,
        sha256="a" * 64,
        selected_csv_member="data.csv",
        zip_members=("README.txt", "data.csv"),
    )
    payload = source_manifest_payload(
        [record],
        start=YearMonth(2025, 1),
        end=YearMonth(2025, 1),
        expected_archive_count=1,
    )
    first = with_manifest_digest(payload)
    reordered = {key: payload[key] for key in reversed(payload)}
    second = with_manifest_digest(reordered)
    assert first["manifest_digest"] == second["manifest_digest"]
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert validate_manifest(first) == first["manifest_digest"]
