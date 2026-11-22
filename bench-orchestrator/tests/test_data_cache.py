# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for local dataset fingerprint validation and atomic cache replacement."""

import io
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from bench_orchestrator.config import Benchmark
from bench_orchestrator.data_cache import MANIFEST, CachedDataset, S3DataCache, cached_dataset
from botocore.exceptions import ClientError


class FakeS3Client:
    """In-memory S3 client for testing S3DataCache."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, datetime]] = {}
        self.get_object_calls: list[str] = []
        self.download_calls: list[str] = []
        self.upload_calls: list[str] = []
        self.fail_on_download_key: str | None = None

    def put_dataset(self, cache: S3DataCache, dataset: CachedDataset, files: dict[str, bytes]) -> None:
        now = datetime.now(UTC)
        for rel, content in files.items():
            self.objects[(cache.bucket, cache.object_key(dataset, rel))] = (content, now)
        manifest = json.dumps({"files": sorted(files)}).encode()
        self.objects[(cache.bucket, cache.object_key(dataset, MANIFEST))] = (manifest, now)

    def get_object(self, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803
        self.get_object_calls.append(Key)
        entry = self.objects.get((Bucket, Key))
        if entry is None:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject")
        body, modified = entry
        return {"Body": io.BytesIO(body), "LastModified": modified}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> object:  # noqa: N803
        self.objects[(Bucket, Key)] = (Body, datetime.now(UTC))
        return {}

    def download_file(self, bucket: str, key: str, filename: str, Config: object = None) -> None:  # noqa: N803
        self.download_calls.append(key)
        if self.fail_on_download_key is not None and key.endswith(self.fail_on_download_key):
            Path(filename).write_bytes(b"partial")
            raise OSError("simulated network interruption")
        entry = self.objects.get((bucket, key))
        if entry is None:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject")
        Path(filename).write_bytes(entry[0])

    def upload_file(self, filename: str, bucket: str, key: str, Config: object = None) -> None:  # noqa: N803
        self.upload_calls.append(key)
        self.objects[(bucket, key)] = (Path(filename).read_bytes(), datetime.now(UTC))


def _make_dataset(base_dir: Path, key: str) -> CachedDataset:
    return CachedDataset(key=key, directory=base_dir / "clickbench-sorted" / "parquet")


def test_matching_key_skips_remote_and_generation(tmp_path: Path) -> None:
    fake = FakeS3Client()
    cache = S3DataCache("s3://bucket/prefix", client=fake)
    ds_v1 = _make_dataset(tmp_path, "clickbench-sorted-aaaa1111")
    fake.put_dataset(cache, ds_v1, {"hits.parquet": b"v1-data"})

    assert cache.restore(ds_v1) is True
    assert (ds_v1.directory / "hits.parquet").read_bytes() == b"v1-data"
    assert len(fake.get_object_calls) == 1

    # Second restore with same key must be a local hit without touching S3.
    assert cache.restore(ds_v1) is True
    assert len(fake.get_object_calls) == 1
    assert (ds_v1.directory / "hits.parquet").read_bytes() == b"v1-data"


def test_changed_key_replaces_obsolete_local_cache_from_remote(tmp_path: Path) -> None:
    fake = FakeS3Client()
    cache = S3DataCache("s3://bucket/prefix", client=fake)
    ds_v1 = _make_dataset(tmp_path, "clickbench-sorted-aaaa1111")
    ds_v2 = _make_dataset(tmp_path, "clickbench-sorted-bbbb2222")
    fake.put_dataset(cache, ds_v1, {"hits.parquet": b"v1-data"})
    fake.put_dataset(cache, ds_v2, {"hits.parquet": b"v2-data"})

    assert cache.restore(ds_v1) is True
    # Simulate a derived format directory written from v1 base data.
    derived = ds_v1.directory.parent / "vortex-file-compressed"
    derived.mkdir()
    (derived / "hits.vortex").write_bytes(b"stale-vortex")

    # Switching to v2 must replace the obsolete local directory and clear stale derived formats.
    assert cache.restore(ds_v2) is True
    assert (ds_v2.directory / "hits.parquet").read_bytes() == b"v2-data"
    assert not derived.exists()

    # Marker must live outside the dataset directory so it is not scanned as data.
    assert list(ds_v2.directory.iterdir()) == [ds_v2.directory / "hits.parquet"]


def test_changed_key_on_remote_miss_clears_obsolete_dir_for_regeneration(tmp_path: Path) -> None:
    fake = FakeS3Client()
    cache = S3DataCache("s3://bucket/prefix", client=fake)
    ds_v1 = _make_dataset(tmp_path, "clickbench-sorted-aaaa1111")
    ds_v2 = _make_dataset(tmp_path, "clickbench-sorted-bbbb2222")
    fake.put_dataset(cache, ds_v1, {"hits.parquet": b"v1-data"})

    assert cache.restore(ds_v1) is True
    derived = ds_v1.directory.parent / "vortex-compact"
    derived.mkdir()
    (derived / "hits.vortex").write_bytes(b"stale-compact")

    # v2 is not in remote cache: restore must return False and clear the obsolete local directory
    # so data-gen sees an empty destination instead of skipping generation.
    assert cache.restore(ds_v2) is False
    assert not ds_v2.directory.exists()
    assert not derived.exists()

    # Simulate data-gen writing fresh v2 output and saving to cache.
    ds_v2.directory.mkdir(parents=True)
    (ds_v2.directory / "hits.parquet").write_bytes(b"generated-v2")
    cache.save(ds_v2)

    # Marker must NOT be uploaded to S3.
    uploaded_keys = {k for _, k in fake.objects if ds_v2.key in k}
    assert uploaded_keys == {
        cache.object_key(ds_v2, "hits.parquet"),
        cache.object_key(ds_v2, MANIFEST),
    }

    # Subsequent restore for v2 is now a local hit.
    calls_before = len(fake.get_object_calls)
    assert cache.restore(ds_v2) is True
    assert len(fake.get_object_calls) == calls_before
    assert (ds_v2.directory / "hits.parquet").read_bytes() == b"generated-v2"


def test_unmarked_local_directory_does_not_qualify_as_hit(tmp_path: Path) -> None:
    fake = FakeS3Client()
    cache = S3DataCache("s3://bucket/prefix", client=fake)
    ds = _make_dataset(tmp_path, "clickbench-sorted-aaaa1111")
    ds.directory.mkdir(parents=True)
    (ds.directory / "hits.parquet").write_bytes(b"unmarked-legacy-data")
    fake.put_dataset(cache, ds, {"hits.parquet": b"fresh-remote-data"})

    assert cache.restore(ds) is True
    assert len(fake.get_object_calls) == 1
    assert (ds.directory / "hits.parquet").read_bytes() == b"fresh-remote-data"


def test_interrupted_replacement_does_not_qualify_as_hit_and_retry_succeeds(tmp_path: Path) -> None:
    fake = FakeS3Client()
    cache = S3DataCache("s3://bucket/prefix", client=fake)
    ds_v1 = _make_dataset(tmp_path, "clickbench-sorted-aaaa1111")
    ds_v2 = _make_dataset(tmp_path, "clickbench-sorted-bbbb2222")
    fake.put_dataset(cache, ds_v1, {"a.parquet": b"v1-a"})
    fake.put_dataset(cache, ds_v2, {"a.parquet": b"v2-a", "b.parquet": b"v2-b"})

    assert cache.restore(ds_v1) is True

    # Interrupt remote download of v2 mid-transfer.
    fake.fail_on_download_key = "b.parquet"
    assert cache.restore(ds_v2) is False
    # Obsolete/incomplete directory must not remain as a valid hit for v2 (or v1).
    assert not ds_v2.directory.exists()

    # Simulate an interrupted local generation that left partial files without saving.
    ds_v2.directory.mkdir(parents=True)
    (ds_v2.directory / "a.parquet").write_bytes(b"incomplete-local")

    # Retry succeeds: the incomplete unmarked directory is NOT accepted as a hit.
    fake.fail_on_download_key = None
    assert cache.restore(ds_v2) is True
    assert (ds_v2.directory / "a.parquet").read_bytes() == b"v2-a"
    assert (ds_v2.directory / "b.parquet").read_bytes() == b"v2-b"


def test_concurrent_prepare_serializes_writers(tmp_path: Path) -> None:
    fake = FakeS3Client()
    cache = S3DataCache("s3://bucket/prefix", client=fake)
    ds = _make_dataset(tmp_path, "clickbench-sorted-cccc3333")

    gen_count = 0
    gen_lock = threading.Lock()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        nonlocal gen_count
        try:
            barrier.wait(timeout=5)
            with cache.prepare(ds) as hit:
                if not hit:
                    with gen_lock:
                        gen_count += 1
                    time.sleep(0.05)
                    ds.directory.mkdir(parents=True, exist_ok=True)
                    (ds.directory / "hits.parquet").write_bytes(b"generated-once")
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors
    assert gen_count == 1
    assert (ds.directory / "hits.parquet").read_bytes() == b"generated-once"


def test_statpopgen_fingerprint_change_invalidates_local_cache(tmp_path: Path) -> None:
    # Build a minimal fake workspace with statpopgen and parquet sources.
    spg_dir = tmp_path / "vortex-bench" / "src" / "statpopgen"
    spg_dir.mkdir(parents=True)
    gen_file = spg_dir / "mod.rs"
    gen_file.write_text("// v1\n", encoding="utf-8")
    pq_file = tmp_path / "vortex-bench" / "src" / "utils" / "parquet.rs"
    pq_file.parent.mkdir(parents=True)
    pq_file.write_text("// parquet v1\n", encoding="utf-8")

    fake = FakeS3Client()
    cache = S3DataCache("s3://bucket/prefix", client=fake)

    ds_v1 = cached_dataset(Benchmark.STATPOPGEN, {"scale-factor": "1"}, tmp_path)
    assert ds_v1 is not None
    fake.put_dataset(cache, ds_v1, {"data.parquet": b"spg-v1"})
    assert cache.restore(ds_v1) is True
    assert (ds_v1.directory / "data.parquet").read_bytes() == b"spg-v1"

    # Change generator source -> fingerprint changes -> local cache must not reuse v1.
    gen_file.write_text("// v2\n", encoding="utf-8")
    ds_v2 = cached_dataset(Benchmark.STATPOPGEN, {"scale-factor": "1"}, tmp_path)
    assert ds_v2 is not None
    assert ds_v2.key != ds_v1.key
    assert ds_v2.directory == ds_v1.directory

    assert cache.restore(ds_v2) is False
    assert not ds_v2.directory.exists()
