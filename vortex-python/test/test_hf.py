# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the Hugging Face integration in :mod:`vortex.hf`.

These tests run a small local HTTP server that mimics the Hub's ``resolve`` endpoint
(including HTTP Range support), so no network access or Hugging Face account is needed.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pytest
import vortex.store
from vortex.hf import HFLocation, resolve_url, store_and_path
from vortex.hf import token as hf_token
from vortex.store import HTTPStore
from vortex.torch_ import VortexMapDataset

import vortex as vx

# --- A tiny local stand-in for the Hugging Face Hub ---


@dataclass
class _RequestLog:
    method: str
    path: str
    range: str | None
    status: int
    bytes_sent: int


@dataclass
class _HubState:
    root: Path
    required_token: str | None = None
    requests: list[_RequestLog] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def bytes_served(self) -> int:
        with self.lock:
            return sum(r.bytes_sent for r in self.requests)

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()


class _HubRequestHandler(BaseHTTPRequestHandler):
    """Serves files from a directory with HTTP Range support, like the Hub's resolve endpoint."""

    protocol_version = "HTTP/1.1"
    state: ClassVar[_HubState]  # assigned on the handler subclass per server

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def _record(self, status: int, bytes_sent: int) -> None:
        with self.state.lock:
            self.state.requests.append(
                _RequestLog(self.command, self.path, self.headers.get("Range"), status, bytes_sent)
            )

    def _error(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()
        self._record(status, 0)

    def _serve(self, send_body: bool) -> None:
        if self.state.required_token is not None:
            if self.headers.get("Authorization") != f"Bearer {self.state.required_token}":
                return self._error(401)

        relpath = unquote(urlsplit(self.path).path).lstrip("/")
        target = (self.state.root / relpath).resolve()
        if not target.is_relative_to(self.state.root.resolve()) or not target.is_file():
            return self._error(404)

        data = target.read_bytes()
        size = len(data)

        range_header = self.headers.get("Range")
        if range_header is None:
            start, end, status = 0, size - 1, 200
        else:
            spec = range_header.removeprefix("bytes=")
            if spec.startswith("-"):
                start, end = max(size - int(spec[1:]), 0), size - 1
            else:
                start_s, _, end_s = spec.partition("-")
                start, end = int(start_s), int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end:
                return self._error(416)
            status = 206

        body = data[start : end + 1]
        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if send_body:
            self.wfile.write(body)
        self._record(status, len(body) if send_body else 0)


@pytest.fixture
def local_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_HubState]:
    root = tmp_path / "hub"
    root.mkdir()
    state = _HubState(root=root)
    handler = type("Handler", (_HubRequestHandler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


def _publish(state: _HubState, repo_path: str, table: pa.Table) -> Path:
    """Write `table` as a Vortex file into the fake hub under a dataset resolve path."""
    target = state.root / repo_path
    target.parent.mkdir(parents=True, exist_ok=True)
    vx.io.write(table, str(target))
    return target


def _sample_table(num_rows: int = 1000, seed: int = 0) -> pa.Table:
    rng = random.Random(seed)
    return pa.table(
        {
            "x": [rng.randrange(1 << 40) for _ in range(num_rows)],
            "s": [rng.randbytes(8).hex() for _ in range(num_rows)],
        }
    )


# --- URL parsing and translation ---


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "hf://datasets/my-org/my-data/file.vortex",
            HFLocation(repo_id="my-org/my-data", path="file.vortex"),
        ),
        (
            "hf://datasets/my-org/my-data@v1.0/dir/file.vortex",
            HFLocation(repo_id="my-org/my-data", path="dir/file.vortex", revision="v1.0"),
        ),
        (
            "hf://datasets/my-org/my-data@refs%2Fconvert%2Fparquet/x.vortex",
            HFLocation(repo_id="my-org/my-data", path="x.vortex", revision="refs/convert/parquet"),
        ),
        (
            "hf://my-org/my-model/weights.vortex",
            HFLocation(repo_id="my-org/my-model", path="weights.vortex", repo_type="model"),
        ),
        (
            "hf://spaces/my-org/my-space/data.vortex",
            HFLocation(repo_id="my-org/my-space", path="data.vortex", repo_type="space"),
        ),
        ("hf://datasets/my-org/my-data", HFLocation(repo_id="my-org/my-data")),
    ],
)
def test_parse(url: str, expected: HFLocation) -> None:
    assert HFLocation.parse(url) == expected


@pytest.mark.parametrize(
    "url",
    ["s3://bucket/file", "hf://", "hf://datasets", "hf://datasets/only-namespace", "hf://datasets/org/name@/x"],
)
def test_parse_invalid(url: str) -> None:
    with pytest.raises(ValueError):
        HFLocation.parse(url)


def test_resolve_url() -> None:
    assert (
        resolve_url("hf://datasets/my-org/my-data@v2/dir/file.vortex")
        == "https://huggingface.co/datasets/my-org/my-data/resolve/v2/dir/file.vortex"
    )
    assert (
        resolve_url("hf://my-org/my-model/weights.vortex")
        == "https://huggingface.co/my-org/my-model/resolve/main/weights.vortex"
    )


def test_resolve_url_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.com/")
    assert (
        resolve_url("hf://datasets/my-org/my-data/file.vortex")
        == "https://hub.example.com/datasets/my-org/my-data/resolve/main/file.vortex"
    )


def test_token_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)

    assert hf_token() is None
    assert hf_token("hf_explicit") == "hf_explicit"

    (tmp_path / "token").write_text("hf_from_file\n")
    assert hf_token() == "hf_from_file"

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert hf_token() == "hf_from_env"
    assert hf_token("hf_explicit") == "hf_explicit"


def test_store_from_url() -> None:
    store = vortex.store.from_url("hf://datasets/my-org/my-data/dir")
    assert isinstance(store, HTTPStore)
    assert store.url == "https://huggingface.co/datasets/my-org/my-data/resolve/main/dir"


def test_store_from_url_rejects_store_config() -> None:
    with pytest.raises(ValueError, match="client_options"):
        vortex.store.from_url("hf://datasets/my-org/my-data", mkdir=True)


def test_store_and_path_requires_file_path() -> None:
    with pytest.raises(ValueError, match="file path"):
        store_and_path("hf://datasets/my-org/my-data")


def test_auth_header_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    store, path = store_and_path("hf://datasets/my-org/my-data/file.vortex")
    assert path == "file.vortex"
    headers = (store.client_options or {}).get("default_headers")
    assert headers is not None
    assert headers["authorization"] == b"Bearer hf_secret" or headers["authorization"] == "Bearer hf_secret"


# --- End-to-end reads against the local hub ---


def test_open_hf_url(local_hub: _HubState) -> None:
    table = _sample_table()
    _publish(local_hub, "datasets/test-org/test-repo/resolve/main/data/train.vortex", table)

    vxf = vx.open("hf://datasets/test-org/test-repo/data/train.vortex")
    assert len(vxf) == table.num_rows

    result = vxf.scan().read_all().to_arrow_table()
    assert result.column("x").to_pylist() == table.column("x").to_pylist()
    assert result.column("s").to_pylist() == table.column("s").to_pylist()


def test_open_hf_url_revision(local_hub: _HubState) -> None:
    table = _sample_table(num_rows=10)
    _publish(local_hub, "datasets/test-org/test-repo/resolve/v1.0/data.vortex", table)

    vxf = vx.open("hf://datasets/test-org/test-repo@v1.0/data.vortex")
    assert len(vxf) == 10


def test_open_hf_url_is_lazy(local_hub: _HubState) -> None:
    table = _sample_table(num_rows=20_000)
    target = _publish(local_hub, "datasets/test-org/test-repo/resolve/main/data.vortex", table)
    file_size = target.stat().st_size

    local_hub.reset()
    vxf = vx.open("hf://datasets/test-org/test-repo/data.vortex")
    projected = vxf.scan(["x"]).read_all()
    assert len(projected) == 20_000

    # A projected scan must not download the whole file, and every read is a ranged read.
    assert 0 < local_hub.bytes_served < file_size
    assert all(r.range is not None for r in local_hub.requests if r.method == "GET")


def test_open_hf_url_with_token(local_hub: _HubState, monkeypatch: pytest.MonkeyPatch) -> None:
    local_hub.required_token = "hf_test_token"
    table = _sample_table(num_rows=10)
    _publish(local_hub, "datasets/test-org/gated-repo/resolve/main/data.vortex", table)

    with pytest.raises(Exception, match="401|Unauthorized"):
        vx.open("hf://datasets/test-org/gated-repo/data.vortex")

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    vxf = vx.open("hf://datasets/test-org/gated-repo/data.vortex")
    assert len(vxf) == 10


def test_vortex_hf_open_explicit_token(local_hub: _HubState) -> None:
    local_hub.required_token = "hf_explicit_token"
    table = _sample_table(num_rows=10)
    _publish(local_hub, "datasets/test-org/gated-repo/resolve/main/data.vortex", table)

    vxf = vx.hf.open("hf://datasets/test-org/gated-repo/data.vortex", token="hf_explicit_token")
    assert len(vxf) == 10


# --- Map-style (torch DataLoader compatible) dataset ---


def test_map_dataset(tmp_path: Path) -> None:
    path = tmp_path / "points.vortex"
    vx.io.write(pa.table({"x": list(range(100)), "y": [i * i for i in range(100)]}), str(path))

    ds = VortexMapDataset(str(path))
    assert len(ds) == 100
    assert ds[3] == {"x": 3, "y": 9}
    assert ds[-1] == {"x": 99, "y": 9801}
    assert ds.__getitems__([7, 3, 7]) == [{"x": 7, "y": 49}, {"x": 3, "y": 9}, {"x": 7, "y": 49}]
    with pytest.raises(IndexError):
        ds[100]


def test_map_dataset_projection(tmp_path: Path) -> None:
    path = tmp_path / "points.vortex"
    vx.io.write(pa.table({"x": list(range(10)), "y": list(range(10))}), str(path))

    ds = VortexMapDataset(str(path), projection=["y"])
    assert ds[4] == {"y": 4}
    assert ds.__getitems__([1, 0]) == [{"y": 1}, {"y": 0}]


def test_map_dataset_over_hf_url(local_hub: _HubState) -> None:
    table = _sample_table(num_rows=500)
    _publish(local_hub, "datasets/test-org/test-repo/resolve/main/train.vortex", table)

    ds = VortexMapDataset("hf://datasets/test-org/test-repo/train.vortex")
    assert len(ds) == 500
    assert ds[42] == {"x": table.column("x")[42].as_py(), "s": table.column("s")[42].as_py()}
    rows = cast("list[dict[str, object]]", ds.__getitems__([499, 0, 250]))
    assert [r["x"] for r in rows] == [table.column("x")[i].as_py() for i in (499, 0, 250)]


# --- Hugging Face `datasets` library integration ---


def test_register_datasets_builder(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    vx.hf.register_datasets()

    shard1, shard2 = tmp_path / "part-0.vortex", tmp_path / "part-1.vortex"
    vx.io.write(pa.table({"x": [1, 2, 3], "s": ["a", "b", "c"]}), str(shard1))
    vx.io.write(pa.table({"x": [4, 5], "s": ["d", "e"]}), str(shard2))

    ds = datasets.load_dataset(
        "vortex",
        data_files={"train": [str(shard1), str(shard2)]},
        cache_dir=str(tmp_path / "cache"),
    )["train"]

    assert ds.num_rows == 5
    assert ds["x"] == [1, 2, 3, 4, 5]
    assert ds["s"] == ["a", "b", "c", "d", "e"]
    assert ds.features["x"].dtype == "int64"
    assert ds.features["s"].dtype == "string"


def test_register_datasets_builder_columns(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    vx.hf.register_datasets()

    shard = tmp_path / "part-0.vortex"
    vx.io.write(pa.table({"x": [1, 2, 3], "s": ["a", "b", "c"]}), str(shard))

    ds = datasets.load_dataset(
        "vortex",
        data_files=str(shard),
        columns=["x"],
        cache_dir=str(tmp_path / "cache"),
    )["train"]

    assert ds.column_names == ["x"]
    assert ds["x"] == [1, 2, 3]


def test_dataset_to_vortex_roundtrip(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")

    ds = datasets.Dataset.from_dict({"x": [1, 2, 3], "s": ["a", "b", "c"]})
    path = tmp_path / "converted.vortex"
    vx.hf.dataset_to_vortex(ds, str(path))

    result = vx.open(str(path)).scan().read_all().to_arrow_table()
    assert result.column("x").to_pylist() == [1, 2, 3]
    assert result.column("s").to_pylist() == ["a", "b", "c"]


# --- Builder pushdown options: filters, limit, indices, on_bad_files, counting ---


def _write_two_shards(tmp_path: Path) -> list[str]:
    """Two shards with global rows x=[1..5], s=[a..e]."""
    shard1, shard2 = tmp_path / "part-0.vortex", tmp_path / "part-1.vortex"
    vx.io.write(pa.table({"x": [1, 2, 3], "s": ["a", "b", "c"]}), str(shard1))
    vx.io.write(pa.table({"x": [4, 5], "s": ["d", "e"]}), str(shard2))
    return [str(shard1), str(shard2)]


def _load(tmp_path: Path, **kwargs):
    datasets = pytest.importorskip("datasets")
    vx.hf.register_datasets()
    return datasets.load_dataset(
        "vortex",
        data_files=_write_two_shards(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        **kwargs,
    )["train"]


def test_builder_filters_and(tmp_path: Path) -> None:
    ds = _load(tmp_path, filters=[("x", ">", 1), ("x", "<", 5)])
    assert ds["x"] == [2, 3, 4]


def test_builder_filters_or(tmp_path: Path) -> None:
    ds = _load(tmp_path, filters=[[("x", "==", 1)], [("x", "==", 5)]])
    assert ds["x"] == [1, 5]


def test_builder_filters_in(tmp_path: Path) -> None:
    ds = _load(tmp_path, filters=[("s", "in", ["a", "e"])])
    assert ds["x"] == [1, 5]


def test_builder_filters_not_in(tmp_path: Path) -> None:
    ds = _load(tmp_path, filters=[("s", "not in", ["a", "e"])])
    assert ds["x"] == [2, 3, 4]


def test_builder_filters_expr(tmp_path: Path) -> None:
    import vortex.expr as ve

    ds = _load(tmp_path, filters=ve.column("x") >= 4)
    assert ds["x"] == [4, 5]


def test_builder_filters_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filter operator"):
        _load(tmp_path, filters=[("x", "~", 1)])


def test_builder_filters_streaming(tmp_path: Path) -> None:
    ds = _load(tmp_path, filters=[("x", ">", 1), ("x", "<", 5)], streaming=True)
    assert [row["x"] for row in ds] == [2, 3, 4]


def test_builder_limit_across_shards(tmp_path: Path) -> None:
    ds = _load(tmp_path, limit=4)
    assert ds["x"] == [1, 2, 3, 4]


def test_builder_limit_with_filters(tmp_path: Path) -> None:
    ds = _load(tmp_path, filters=[("x", ">", 1)], limit=2)
    assert ds["x"] == [2, 3]


def test_builder_indices_across_shards(tmp_path: Path) -> None:
    # Unsorted on purpose: rows come back in ascending row order.
    ds = _load(tmp_path, indices=[4, 0, 2])
    assert ds["x"] == [1, 3, 5]


def test_builder_indices_out_of_range(tmp_path: Path) -> None:
    ds = _load(tmp_path, indices=[0, 99], streaming=True)
    with pytest.raises(IndexError, match="total row count"):
        list(ds)


def test_builder_indices_with_filters_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="indices cannot be combined"):
        _load(tmp_path, indices=[0], filters=[("x", ">", 1)])


def test_builder_on_bad_files(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    vx.hf.register_datasets()
    bad = tmp_path / "bad.vortex"
    bad.write_bytes(b"this is not a vortex file")
    good = tmp_path / "good.vortex"
    vx.io.write(pa.table({"x": [1, 2, 3]}), str(good))
    data_files = [str(bad), str(good)]

    ds = datasets.load_dataset(
        "vortex", data_files=data_files, on_bad_files="skip", cache_dir=str(tmp_path / "c1")
    )["train"]
    assert ds["x"] == [1, 2, 3]

    ds = datasets.load_dataset(
        "vortex", data_files=data_files, on_bad_files="warn", cache_dir=str(tmp_path / "c2")
    )["train"]
    assert ds["x"] == [1, 2, 3]

    with pytest.raises(Exception):  # noqa: B017 - datasets wraps the underlying error
        datasets.load_dataset("vortex", data_files=data_files, cache_dir=str(tmp_path / "c3"))

    with pytest.raises(ValueError, match="on_bad_files"):
        datasets.load_dataset(
            "vortex", data_files=data_files, on_bad_files="bogus", cache_dir=str(tmp_path / "c4")
        )


def test_builder_generate_num_examples(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    vx.hf.register_datasets()
    shards = _write_two_shards(tmp_path)

    builder = datasets.load_dataset_builder(
        "vortex", data_files=shards, cache_dir=str(tmp_path / "cache")
    )
    assert list(builder._generate_num_examples(files=[[shards[0]], [shards[1]]])) == [3, 2]

    limited = datasets.load_dataset_builder(
        "vortex", data_files=shards, limit=2, cache_dir=str(tmp_path / "cache2")
    )
    with pytest.raises(NotImplementedError):
        list(limited._generate_num_examples(files=[[shards[0]]]))
