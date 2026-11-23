# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""S3 cache for base benchmark data that is expensive to generate.

``data-gen`` only generates data. This module decides which benchmarks' base data is cached, keys it
on the generator source, restores the directory from S3 before ``data-gen`` runs (which then skips
generating it), and uploads the directory after ``data-gen`` generated it.

A cache entry is ``s3://bucket/prefix/<key>/`` holding the directory's files plus ``manifest.json``,
which is written last so that an entry without it is incomplete.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Protocol, cast
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from rich.console import Console

from .config import Benchmark

# Cached entries older than this are treated as missing and overwritten by the next upload.
RETENTION = timedelta(days=1)

MANIFEST = "manifest.json"

# Files transfer concurrently, and each file above the part size is split into parts that are
# fetched with concurrent ranged GETs or sent as a concurrent multipart upload.
_FILE_CONCURRENCY = 8
_PART_CONCURRENCY = 16
_PART_SIZE = 16 * 1024 * 1024

console = Console(stderr=True)

# Sibling format directories written by ``vortex-bench`` beside the cached base directory.
# Keep in sync with ``Format::name()`` in ``vortex-bench/src/lib.rs``, ``PARQUET_SNAPPY_DIR`` in
# ``vortex-bench/src/parquet_copies.rs``, and ``formats_to_capture`` in ``scripts/capture-file-sizes.py``.
_MANAGED_FORMAT_DIRS: frozenset[str] = frozenset(
    {
        "arrow-ipc",
        "csv",
        "parquet",
        "parquet-snappy",
        "vortex-file-compressed",
        "vortex-compact",
        "vortex-spatial-native",
        "duckdb",
        "lance",
        "litdata",
    }
)


def _marker_path(dataset: CachedDataset) -> Path:
    """Path to the fingerprint marker stored beside ``dataset.directory``."""
    return dataset.directory.with_name(f".{dataset.directory.name}.cache-key")


def _has_matching_local_cache(dataset: CachedDataset) -> bool:
    """Return whether ``dataset.directory`` exists and its sibling marker matches ``dataset.key``."""
    if not dataset.directory.is_dir():
        return False
    marker = _marker_path(dataset)
    if not marker.is_file():
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == dataset.key
    except OSError:
        return False


def _write_marker(dataset: CachedDataset) -> None:
    """Write the completed fingerprint marker beside ``dataset.directory``."""
    marker = _marker_path(dataset)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{dataset.key}\n", encoding="utf-8")


def _remove_stale_derived_dirs(dataset: CachedDataset) -> None:
    """Remove sibling derived-format directories owned by the generator."""
    parent = dataset.directory.parent
    if not parent.is_dir():
        return
    for entry in parent.iterdir():
        if entry.name == dataset.directory.name or entry.name not in _MANAGED_FORMAT_DIRS:
            continue
        if entry.is_symlink():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)


def _invalidate_local(dataset: CachedDataset) -> None:
    """Clear marker, obsolete base directory, and sibling derived-format directories."""
    _marker_path(dataset).unlink(missing_ok=True)
    _remove_stale_derived_dirs(dataset)
    if dataset.directory.is_symlink() or dataset.directory.is_file():
        dataset.directory.unlink(missing_ok=True)
    elif dataset.directory.is_dir():
        shutil.rmtree(dataset.directory)


@contextmanager
def prepare_dataset(dataset: CachedDataset, cache: S3DataCache | None = None) -> Iterator[bool]:
    """Coordinate local cache validation, optional S3 restore/save, and clean regeneration.

    Yields ``True`` when the local dataset already matches ``dataset.key`` (or was restored from
    S3), and ``False`` when the caller must generate ``dataset.directory`` from scratch.
    """
    if cache is not None:
        restored = cache.restore(dataset)
    elif _has_matching_local_cache(dataset):
        console.print(f"{dataset.directory} already matches {dataset.key}")
        restored = True
    else:
        _invalidate_local(dataset)
        restored = False

    yield restored

    if not restored:
        if cache is not None:
            cache.save(dataset)
        elif dataset.directory.is_dir():
            _write_marker(dataset)


class S3Client(Protocol):
    """The subset of a boto3 S3 client the cache uses."""

    def get_object(self, Bucket: str, Key: str) -> dict[str, object]: ...  # noqa: N803

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> object: ...  # noqa: N803

    def download_file(self, bucket: str, key: str, filename: str, Config: object = None) -> None: ...  # noqa: N803

    def upload_file(self, filename: str, bucket: str, key: str, Config: object = None) -> None: ...  # noqa: N803


@dataclass(frozen=True)
class CachedDataset:
    """A directory ``data-gen`` writes in one piece and skips generating when it exists."""

    key: str
    directory: Path


def cached_dataset(benchmark: Benchmark, options: dict[str, str], workspace_root: Path) -> CachedDataset | None:
    """Return the cacheable base data ``data-gen`` writes for ``benchmark``, if any."""
    data_dir = workspace_root / "vortex-bench" / "data"
    if benchmark is Benchmark.STATPOPGEN:
        # data-gen downloads `scale-factor` thousand variants (default 1).
        n_rows = int(options.get("scale-factor", "1")) * 1000
        return _dataset(
            f"statpopgen-{n_rows}",
            data_dir / "statpopgen" / str(n_rows) / "parquet",
            workspace_root,
            ["vortex-bench/src/statpopgen", "vortex-bench/src/utils/parquet.rs"],
        )
    if benchmark is Benchmark.CLICKBENCH_SORTED:
        return _dataset(
            "clickbench-sorted",
            data_dir / "clickbench-sorted" / "parquet",
            workspace_root,
            ["vortex-bench/src/clickbench/data.rs"],
        )
    return None


def _dataset(name: str, directory: Path, workspace_root: Path, sources: list[str]) -> CachedDataset:
    """Key ``directory`` on ``name`` and the contents of the generator ``sources``."""
    hasher = hashlib.sha256()
    for source in sources:
        path = workspace_root / source
        for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if file.is_file():
                hasher.update(file.relative_to(workspace_root).as_posix().encode())
                hasher.update(file.read_bytes())
    return CachedDataset(key=f"{name}-{hasher.hexdigest()[:16]}", directory=directory)


class S3DataCache:
    """Cached datasets stored under ``s3://bucket/prefix/<dataset key>/``."""

    def __init__(self, url: str, client: S3Client | None = None) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"S3 cache URL must look like s3://bucket/prefix, got {url!r}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self._client = client
        self._transfer_config = TransferConfig(
            multipart_threshold=_PART_SIZE,
            multipart_chunksize=_PART_SIZE,
            max_concurrency=_PART_CONCURRENCY,
        )

    @property
    def client(self) -> S3Client:
        if self._client is None:
            # botocore pools 10 connections per client by default, which would cap every file and
            # part thread at 10 requests in flight.
            config = Config(max_pool_connections=_FILE_CONCURRENCY * _PART_CONCURRENCY)
            self._client = cast("S3Client", boto3.client("s3", config=config))
        return self._client

    def object_key(self, dataset: CachedDataset, file: str) -> str:
        return "/".join(part for part in (self.prefix, dataset.key, file) if part)

    def _cached_files(self, dataset: CachedDataset) -> list[str] | None:
        """Return the files of a complete entry younger than ``RETENTION``, or None."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.object_key(dataset, MANIFEST))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        modified = response["LastModified"]
        assert isinstance(modified, datetime)
        if datetime.now(UTC) - modified >= RETENTION:
            return None
        body = cast("BinaryIO", response["Body"])
        files = json.loads(body.read())["files"]
        assert isinstance(files, list)
        return [str(file) for file in files]

    def restore(self, dataset: CachedDataset) -> bool:
        """Download the dataset if a fresh entry exists. Return whether ``data-gen`` can skip it.

        Only a local directory whose sibling fingerprint marker matches ``dataset.key`` counts as
        restored. Stale or unmarked local data is invalidated before downloading, and the directory
        only appears once every file is downloaded. S3 errors are reported and treated as a miss,
        so a cache outage only costs regeneration.
        """
        if _has_matching_local_cache(dataset):
            console.print(f"{dataset.directory} already matches {dataset.key}")
            return True

        _invalidate_local(dataset)

        staging = dataset.directory.with_name(f".{dataset.directory.name}.partial")
        try:
            files = self._cached_files(dataset)
            if files is None:
                console.print(f"No complete cache for {dataset.key} newer than {RETENTION}")
                return False

            console.print(f"Restoring {len(files)} cached file(s) for {dataset.key}")
            if staging.is_symlink() or staging.is_file():
                staging.unlink()
            elif staging.exists():
                shutil.rmtree(staging)
            staging.parent.mkdir(parents=True, exist_ok=True)
            staging.mkdir(exist_ok=False)

            def download(file: str) -> None:
                path = staging / file
                path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(
                    self.bucket, self.object_key(dataset, file), str(path), Config=self._transfer_config
                )

            with ThreadPoolExecutor(_FILE_CONCURRENCY) as pool:
                list(pool.map(download, files))
            os.replace(staging, dataset.directory)
            _write_marker(dataset)
        except (BotoCoreError, ClientError, OSError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            console.print(f"[yellow]S3 cache restore failed for {dataset.key}, regenerating: {exc}[/yellow]")
            return False
        return True

    def save(self, dataset: CachedDataset) -> None:
        """Record the local marker and upload the generated directory.

        S3 errors are reported but do not fail data generation.
        """
        if not dataset.directory.is_dir():
            raise FileNotFoundError(f"data-gen did not write {dataset.directory}")
        _write_marker(dataset)
        files = sorted(
            path.relative_to(dataset.directory).as_posix() for path in dataset.directory.rglob("*") if path.is_file()
        )

        def upload(file: str) -> None:
            self.client.upload_file(
                str(dataset.directory / file),
                self.bucket,
                self.object_key(dataset, file),
                Config=self._transfer_config,
            )

        console.print(f"Uploading {len(files)} file(s) to the cache for {dataset.key}")
        try:
            with ThreadPoolExecutor(_FILE_CONCURRENCY) as pool:
                list(pool.map(upload, files))
            manifest = json.dumps({"files": files}).encode()
            self.client.put_object(Bucket=self.bucket, Key=self.object_key(dataset, MANIFEST), Body=manifest)
        except (BotoCoreError, ClientError, OSError) as exc:
            console.print(f"[yellow]S3 cache upload failed for {dataset.key}: {exc}[/yellow]")
