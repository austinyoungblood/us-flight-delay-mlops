"""Sequential, validated downloader for official BTS Reporting Carrier archives."""

from __future__ import annotations

import hashlib
import os
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import requests

from flight_delay.data.manifest import read_manifest, write_manifest
from flight_delay.data.preprocessing import DataQualityError

BTS_SOURCE_DIRECTORY_URL = "https://transtats.bts.gov/PREZIP/"
BTS_DATASET_NAME = "DOT/BTS Reporting Carrier On-Time Performance"
ARCHIVE_PREFIX = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
ZIP_MAGIC_PREFIXES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


class DownloadError(DataQualityError):
    """Raised when an archive cannot be downloaded or validated safely."""


class ChecksumConflictError(DownloadError):
    """Raised when an existing archive conflicts with stable provenance."""


class HTTPResponse(Protocol):
    """Minimum streaming response surface used by the downloader."""

    status_code: int

    def iter_content(self, chunk_size: int) -> Iterable[bytes]: ...

    def close(self) -> None: ...


class HTTPClient(Protocol):
    """Injectable HTTP client surface for hermetic tests."""

    def get(
        self,
        url: str,
        *,
        stream: bool,
        timeout: tuple[float, float],
    ) -> HTTPResponse: ...


@dataclass(frozen=True, order=True)
class YearMonth:
    """Validated calendar month used to construct official archive URLs."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if self.year < 1987 or not 1 <= self.month <= 12:
            raise DownloadError(f"invalid BTS year/month: {self.year}-{self.month}")

    @classmethod
    def parse(cls, value: str) -> YearMonth:
        """Parse an ISO-like ``YYYY-MM`` month."""

        parts = value.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            raise DownloadError(f"month must use YYYY-MM format: {value!r}")
        try:
            return cls(year=int(parts[0]), month=int(parts[1]))
        except ValueError as error:
            raise DownloadError(f"month must use YYYY-MM format: {value!r}") from error

    def iso(self) -> str:
        """Return the canonical ``YYYY-MM`` representation."""

        return f"{self.year:04d}-{self.month:02d}"


@dataclass(frozen=True)
class ZipInspection:
    """Validated stable metadata from an archive central directory."""

    selected_csv_member: str
    zip_members: tuple[str, ...]


@dataclass(frozen=True)
class SourceFileRecord:
    """Stable provenance for one downloaded monthly archive."""

    year: int
    month: int
    url: str
    archive_filename: str
    byte_size: int
    sha256: str
    selected_csv_member: str
    zip_members: tuple[str, ...]

    def to_manifest_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible record."""

        payload = asdict(self)
        payload["zip_members"] = list(self.zip_members)
        return payload


@dataclass(frozen=True)
class DownloadFileResult:
    """Typed result for one requested archive."""

    month: YearMonth
    status: Literal["downloaded", "skipped", "failed"]
    record: SourceFileRecord | None
    bytes_transferred: int
    error: str | None = None


@dataclass(frozen=True)
class DownloadSummary:
    """Aggregate results for a sequential archive batch."""

    results: tuple[DownloadFileResult, ...]

    @property
    def downloaded(self) -> int:
        return sum(result.status == "downloaded" for result in self.results)

    @property
    def skipped(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.results)

    @property
    def total_bytes(self) -> int:
        return sum(result.record.byte_size for result in self.results if result.record)


class DownloadBatchError(DownloadError):
    """Raised after a batch finishes with one or more failed months."""

    def __init__(self, summary: DownloadSummary) -> None:
        self.summary = summary
        failures = [
            f"{result.month.iso()}: {result.error}"
            for result in summary.results
            if result.status == "failed"
        ]
        super().__init__("BTS download batch failed; " + "; ".join(failures))


def inclusive_month_range(start: YearMonth, end: YearMonth) -> tuple[YearMonth, ...]:
    """Return every calendar month in the inclusive ordered range."""

    if start > end:
        raise DownloadError(f"start month {start.iso()} is after end month {end.iso()}")
    months: list[YearMonth] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(YearMonth(year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return tuple(months)


def archive_filename(month: YearMonth) -> str:
    """Return the official Reporting Carrier archive filename for a month."""

    return f"{ARCHIVE_PREFIX}_{month.year}_{month.month}.zip"


def archive_url(month: YearMonth, *, base_url: str = BTS_SOURCE_DIRECTORY_URL) -> str:
    """Construct an official archive URL without scraping a directory listing."""

    if "Marketing_Carrier" in base_url:
        raise DownloadError("Marketing Carrier archive family is forbidden")
    return f"{base_url.rstrip('/')}/{archive_filename(month)}"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file SHA-256 using bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _usable_csv_members(names: Iterable[str]) -> list[str]:
    excluded_tokens = ("readme", "documentation", "read_me", "manifest")
    return sorted(
        name
        for name in names
        if not name.endswith("/")
        and name.casefold().endswith(".csv")
        and not any(token in name.casefold() for token in excluded_tokens)
    )


def inspect_zip(path: Path) -> ZipInspection:
    """Validate ZIP magic, central directory, CRCs, and unique usable CSV member."""

    try:
        with path.open("rb") as source:
            magic = source.read(4)
    except OSError as error:
        raise DownloadError(f"cannot read archive {path.name}: {error}") from error
    if not magic.startswith(ZIP_MAGIC_PREFIXES):
        raise DownloadError(f"archive {path.name} does not have ZIP magic bytes")
    try:
        with zipfile.ZipFile(path) as archive:
            names = tuple(sorted(archive.namelist()))
            corrupt_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise DownloadError(f"archive {path.name} has an unreadable ZIP directory") from error
    if corrupt_member is not None:
        raise DownloadError(f"archive {path.name} failed CRC validation at {corrupt_member}")
    usable = _usable_csv_members(names)
    if len(usable) != 1:
        raise DownloadError(
            f"archive {path.name} must contain exactly one usable data CSV; found {usable}"
        )
    return ZipInspection(selected_csv_member=usable[0], zip_members=names)


def _record_for_file(month: YearMonth, path: Path, inspection: ZipInspection) -> SourceFileRecord:
    return SourceFileRecord(
        year=month.year,
        month=month.month,
        url=archive_url(month),
        archive_filename=path.name,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        selected_csv_member=inspection.selected_csv_member,
        zip_members=inspection.zip_members,
    )


def _response_is_transient(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code <= 599


def download_archive(
    month: YearMonth,
    output_directory: Path,
    *,
    client: HTTPClient,
    expected_record: Mapping[str, object] | None = None,
    overwrite: bool = False,
    timeout: tuple[float, float] = (10.0, 120.0),
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    chunk_size: int = 1024 * 1024,
) -> DownloadFileResult:
    """Download one archive through a temporary file and atomically publish it."""

    if max_attempts <= 0:
        raise DownloadError("max_attempts must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / archive_filename(month)
    partial_path = final_path.with_name(f"{final_path.name}.part")
    partial_path.unlink(missing_ok=True)

    if final_path.exists() and not overwrite:
        if expected_record is None:
            raise ChecksumConflictError(
                f"{final_path.name} exists without a trusted source-manifest record; "
                "use --overwrite to replace it"
            )
        expected_checksum = expected_record.get("sha256")
        actual_checksum = sha256_file(final_path)
        if not isinstance(expected_checksum, str) or actual_checksum != expected_checksum:
            raise ChecksumConflictError(
                f"{final_path.name} checksum conflicts with source manifest; use --overwrite"
            )
        inspection = inspect_zip(final_path)
        record = _record_for_file(month, final_path, inspection)
        return DownloadFileResult(month, "skipped", record, bytes_transferred=0)

    url = archive_url(month)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        response: HTTPResponse | None = None
        try:
            response = client.get(url, stream=True, timeout=timeout)
            if response.status_code < 200 or response.status_code >= 300:
                error = DownloadError(f"HTTP {response.status_code} for {url}")
                if not _response_is_transient(response.status_code):
                    raise error
                raise requests.RequestException(str(error))
            digest = hashlib.sha256()
            transferred = 0
            with partial_path.open("wb") as destination:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    destination.write(chunk)
                    digest.update(chunk)
                    transferred += len(chunk)
            inspection = inspect_zip(partial_path)
            os.replace(partial_path, final_path)
            record = SourceFileRecord(
                year=month.year,
                month=month.month,
                url=url,
                archive_filename=final_path.name,
                byte_size=transferred,
                sha256=digest.hexdigest(),
                selected_csv_member=inspection.selected_csv_member,
                zip_members=inspection.zip_members,
            )
            return DownloadFileResult(month, "downloaded", record, transferred)
        except (OSError, DownloadError, requests.RequestException) as error:
            last_error = error
            partial_path.unlink(missing_ok=True)
            permanent = isinstance(error, DownloadError) and not str(error).startswith("HTTP 5")
            if permanent or attempt == max_attempts:
                break
            sleep(backoff_seconds * (2 ** (attempt - 1)))
        finally:
            if response is not None:
                response.close()
    raise DownloadError(
        f"failed to download {month.iso()} after {max_attempts} attempts: {last_error}"
    ) from last_error


def source_manifest_payload(
    records: Iterable[SourceFileRecord],
    *,
    start: YearMonth,
    end: YearMonth,
    expected_archive_count: int,
) -> dict[str, object]:
    """Build the stable source-manifest payload without volatile telemetry."""

    sorted_records = sorted(records, key=lambda record: (record.year, record.month))
    if len(sorted_records) != expected_archive_count:
        raise DownloadError(
            f"expected {expected_archive_count} archive records, got {len(sorted_records)}"
        )
    return {
        "schema_version": 1,
        "dataset_name": BTS_DATASET_NAME,
        "source_directory_url": BTS_SOURCE_DIRECTORY_URL,
        "start_month": start.iso(),
        "end_month": end.iso(),
        "expected_archive_count": expected_archive_count,
        "aggregate_archive_bytes": sum(record.byte_size for record in sorted_records),
        "files": [record.to_manifest_dict() for record in sorted_records],
    }


def _manifest_record_lookup(manifest_path: Path) -> dict[str, Mapping[str, object]]:
    if not manifest_path.exists():
        return {}
    payload = read_manifest(manifest_path)
    records = payload.get("files")
    if not isinstance(records, list):
        raise DownloadError("source manifest files must be a list")
    return {
        str(record["archive_filename"]): record
        for record in records
        if isinstance(record, dict) and "archive_filename" in record
    }


def download_archives(
    months: Iterable[YearMonth],
    output_directory: Path,
    manifest_path: Path,
    *,
    expected_archive_count: int,
    client: HTTPClient | None = None,
    overwrite: bool = False,
    timeout: tuple[float, float] = (10.0, 120.0),
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[DownloadSummary, dict[str, object]]:
    """Download a complete month set sequentially and write its stable manifest."""

    requested = tuple(months)
    if len(requested) != expected_archive_count:
        raise DownloadError(
            f"expected {expected_archive_count} requested months, got {len(requested)}"
        )
    active_client: HTTPClient = client if client is not None else requests.Session()
    existing = _manifest_record_lookup(manifest_path)
    results: list[DownloadFileResult] = []
    for month in requested:
        filename = archive_filename(month)
        try:
            result = download_archive(
                month,
                output_directory,
                client=active_client,
                expected_record=existing.get(filename),
                overwrite=overwrite,
                timeout=timeout,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
                sleep=sleep,
            )
        except DownloadError as error:
            result = DownloadFileResult(month, "failed", None, 0, str(error))
        results.append(result)
    summary = DownloadSummary(tuple(results))
    if summary.failed:
        raise DownloadBatchError(summary)
    records = [result.record for result in results if result.record is not None]
    payload = source_manifest_payload(
        records,
        start=requested[0],
        end=requested[-1],
        expected_archive_count=expected_archive_count,
    )
    return summary, write_manifest(manifest_path, payload)
