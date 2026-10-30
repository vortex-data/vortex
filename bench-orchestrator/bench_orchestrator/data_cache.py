# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""S3 cache for base benchmark data that is expensive to generate.

``data-gen`` only generates data. This module decides which benchmarks' base data is cached, keys it
on the generator source, restores the directory from S3 before ``data-gen`` runs (which then skips
generating it), and uploads the directory after ``data-gen`` generated it.

A cache entry is ``s3://bucket/prefix/<key>/`` holding the directory's files plus ``manifest.json``,
which is written last so that an entry without it is incomplete.
"""

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
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
            ["vortex-bench/src/statpopgen"],
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

        An existing local directory counts as restored. The directory only appears once every file
        is downloaded. S3 errors are reported and treated as a miss, so a cache outage only costs
        regeneration.
        """
        if dataset.directory.exists():
            console.print(f"{dataset.directory} already exists locally")
            return True

        staging = dataset.directory.with_name(f".{dataset.directory.name}.partial")
        try:
            files = self._cached_files(dataset)
            if files is None:
                console.print(f"No complete cache for {dataset.key} newer than {RETENTION}")
                return False

            console.print(f"Restoring {len(files)} cached file(s) for {dataset.key}")
            shutil.rmtree(staging, ignore_errors=True)

            def download(file: str) -> None:
                path = staging / file
                path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(
                    self.bucket, self.object_key(dataset, file), str(path), Config=self._transfer_config
                )

            with ThreadPoolExecutor(_FILE_CONCURRENCY) as pool:
                list(pool.map(download, files))
            os.replace(staging, dataset.directory)
        except (BotoCoreError, ClientError, OSError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            console.print(f"[yellow]S3 cache restore failed for {dataset.key}, regenerating: {exc}[/yellow]")
            return False
        return True

    def save(self, dataset: CachedDataset) -> None:
        """Upload the generated directory. S3 errors are reported but do not fail data generation."""
        if not dataset.directory.is_dir():
            raise FileNotFoundError(f"data-gen did not write {dataset.directory}")
        files = sorted(
            path.relative_to(dataset.directory).as_posix() for path in dataset.directory.rglob("*") if path.is_file()
        )

        def upload(file: str) -> None:
            self.client.upload_file(
                str(dataset.directory / file), self.bucket, self.object_key(dataset, file), Config=self._transfer_config
            )

        console.print(f"Uploading {len(files)} file(s) to the cache for {dataset.key}")
        try:
            with ThreadPoolExecutor(_FILE_CONCURRENCY) as pool:
                list(pool.map(upload, files))
            manifest = json.dumps({"files": files}).encode()
            self.client.put_object(Bucket=self.bucket, Key=self.object_key(dataset, MANIFEST), Body=manifest)
        except (BotoCoreError, ClientError, OSError) as exc:
            console.print(f"[yellow]S3 cache upload failed for {dataset.key}: {exc}[/yellow]")
