# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import io
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bench_orchestrator.config import Benchmark
from bench_orchestrator.data_cache import MANIFEST, RETENTION, CachedDataset, S3DataCache, cached_dataset
from botocore.exceptions import ClientError


class FakeS3:
    def __init__(self, storage: Path) -> None:
        self.storage = storage
        self.modified: dict[str, datetime] = {}
        self.downloads: list[str] = []
        self.uploads: list[str] = []

    def _write(self, key: str, data: bytes) -> None:
        (self.storage / key).parent.mkdir(parents=True, exist_ok=True)
        (self.storage / key).write_bytes(data)
        self.modified[key] = datetime.now(UTC)

    def get_object(self, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803
        if Key not in self.modified:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"LastModified": self.modified[Key], "Body": io.BytesIO((self.storage / Key).read_bytes())}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
        self._write(Key, Body)

    def download_file(self, bucket: str, key: str, filename: str, Config: object = None) -> None:  # noqa: N803
        self.downloads.append(key)
        shutil.copyfile(self.storage / key, filename)

    def upload_file(self, filename: str, bucket: str, key: str, Config: object = None) -> None:  # noqa: N803
        self.uploads.append(key)
        self._write(key, Path(filename).read_bytes())


FILES = ("a.parquet", "nested/b.parquet")


@pytest.fixture
def dataset(tmp_path: Path) -> CachedDataset:
    return CachedDataset(key="bench-abc", directory=tmp_path / "data" / "parquet")


def _generate(dataset: CachedDataset) -> None:
    for file in FILES:
        path = dataset.directory / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file)


def test_round_trip(tmp_path: Path, dataset: CachedDataset) -> None:
    s3 = FakeS3(tmp_path / "s3")
    cache = S3DataCache("s3://bucket/data-gen-cache/", client=s3)

    assert not cache.restore(dataset)
    _generate(dataset)
    cache.save(dataset)
    assert s3.uploads == ["data-gen-cache/bench-abc/a.parquet", "data-gen-cache/bench-abc/nested/b.parquet"]
    assert "data-gen-cache/bench-abc/manifest.json" in s3.modified

    shutil.rmtree(dataset.directory)
    assert cache.restore(dataset)
    assert [(dataset.directory / file).read_text() for file in FILES] == list(FILES)
    assert not dataset.directory.with_name(".parquet.partial").exists()


def test_stale_or_incomplete_entry_is_a_miss(tmp_path: Path, dataset: CachedDataset) -> None:
    s3 = FakeS3(tmp_path / "s3")
    cache = S3DataCache("s3://bucket/prefix", client=s3)
    _generate(dataset)
    cache.save(dataset)
    shutil.rmtree(dataset.directory)
    manifest = cache.object_key(dataset, MANIFEST)

    s3.modified[manifest] = datetime.now(UTC) - RETENTION - timedelta(minutes=1)
    assert not cache.restore(dataset)

    # Without the manifest, which is written last, the entry is incomplete.
    del s3.modified[manifest]
    assert not cache.restore(dataset)
    assert s3.downloads == []
    assert not dataset.directory.exists()


def test_failed_restore_leaves_no_directory(tmp_path: Path, dataset: CachedDataset) -> None:
    s3 = FakeS3(tmp_path / "s3")
    cache = S3DataCache("s3://bucket", client=s3)
    _generate(dataset)
    cache.save(dataset)
    shutil.rmtree(dataset.directory)
    (s3.storage / cache.object_key(dataset, FILES[1])).unlink()

    assert not cache.restore(dataset)
    assert not dataset.directory.exists()
    assert not dataset.directory.with_name(".parquet.partial").exists()


def test_existing_local_directory_skips_s3(dataset: CachedDataset) -> None:
    _generate(dataset)
    # An empty fake S3 would report a miss, so a hit proves S3 was not consulted.
    assert S3DataCache("s3://bucket", client=FakeS3(Path("/nonexistent"))).restore(dataset)


def test_save_requires_generated_directory(tmp_path: Path, dataset: CachedDataset) -> None:
    with pytest.raises(FileNotFoundError):
        S3DataCache("s3://bucket", client=FakeS3(tmp_path)).save(dataset)


def test_rejects_non_s3_url() -> None:
    with pytest.raises(ValueError):
        S3DataCache("gs://bucket/prefix")


def test_cached_datasets(tmp_path: Path) -> None:
    source = tmp_path / "vortex-bench" / "src" / "statpopgen" / "download_vcf.rs"
    source.parent.mkdir(parents=True)
    source.write_text("v1")

    dataset = cached_dataset(Benchmark.STATPOPGEN, {"scale-factor": "100"}, tmp_path)
    assert dataset is not None
    assert dataset.directory == tmp_path / "vortex-bench" / "data" / "statpopgen" / "100000" / "parquet"
    assert dataset.key.startswith("statpopgen-100000-")

    source.write_text("v2")
    changed = cached_dataset(Benchmark.STATPOPGEN, {"scale-factor": "100"}, tmp_path)
    assert changed is not None
    assert changed.key != dataset.key

    sorted_clickbench = cached_dataset(Benchmark.CLICKBENCH_SORTED, {}, tmp_path)
    assert sorted_clickbench is not None
    assert sorted_clickbench.directory == tmp_path / "vortex-bench" / "data" / "clickbench-sorted" / "parquet"
    assert cached_dataset(Benchmark.TPCH, {}, tmp_path) is None
