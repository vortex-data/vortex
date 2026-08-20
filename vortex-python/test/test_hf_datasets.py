# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, cast

import datasets as hf_datasets
import pyarrow as pa
import pytest
import vortex.datasets as vx_datasets
from typing_extensions import override
from vortex.store import HfStore

import vortex as vx
import vortex.expr as ve


def test_datasets_module_is_lazy_exported():
    assert vx.datasets is vx_datasets


def write_vortex(path: Path, rows: list[dict[str, object]]) -> None:
    vx.io.write(pa.Table.from_pylist(rows), str(path))


class _RangeRequestHandler(BaseHTTPRequestHandler):
    """Serves files from `directory` with just enough HTTP range support for Vortex scans."""

    directory: ClassVar[Path]

    def do_HEAD(self) -> None:
        data = self._read_file()
        if data is None:
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        data = self._read_file()
        if data is None:
            return
        range_match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if range_match is None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            _ = self.wfile.write(data)
            return
        start_group, end_group = range_match.groups()
        if start_group:
            start = int(start_group)
            end = int(end_group) if end_group else len(data) - 1
        else:
            start, end = max(0, len(data) - int(end_group)), len(data) - 1
        end = min(end, len(data) - 1)
        chunk = data[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        _ = self.wfile.write(chunk)

    def _read_file(self) -> bytes | None:
        file = self.directory / self.path.lstrip("/")
        if not file.is_file():
            self.send_error(404)
            return None
        return file.read_bytes()

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def http_file_server(tmp_path: Path) -> Iterator[str]:
    handler = type("Handler", (_RangeRequestHandler,), {"directory": tmp_path})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_load_dataset_streaming_local_splits(tmp_path: Path):
    write_vortex(
        tmp_path / "train-0000.vortex",
        [
            {"text": "zero", "label": 0, "tokens": 10},
            {"text": "one", "label": 1, "tokens": 11},
        ],
    )
    write_vortex(
        tmp_path / "train-0001.vortex",
        [
            {"text": "two", "label": 0, "tokens": 12},
            {"text": "three", "label": 1, "tokens": 13},
        ],
    )
    write_vortex(tmp_path / "validation.vortex", [{"text": "valid", "label": 1, "tokens": 20}])

    dataset = vx_datasets.load_dataset(
        tmp_path,
        data_files={"train": "train-*.vortex", "validation": "validation.vortex"},
    )

    assert isinstance(dataset, hf_datasets.IterableDatasetDict)
    assert list(dataset) == ["train", "validation"]
    assert list(dataset["train"].take(3)) == [
        {"label": 0, "text": "zero", "tokens": 10},
        {"label": 1, "text": "one", "tokens": 11},
        {"label": 0, "text": "two", "tokens": 12},
    ]
    assert list(dataset["validation"]) == [{"label": 1, "text": "valid", "tokens": 20}]


def test_streaming_select_columns_pushes_projection(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero", "label": 0}, {"text": "one", "label": 1}])

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    selected = dataset.select_columns(["text"])

    assert isinstance(selected, vx_datasets.VortexIterableDataset)
    assert selected._vortex_columns == ("text",)  # pyright: ignore[reportPrivateUsage]
    assert list(selected) == [{"text": "zero"}, {"text": "one"}]


def test_streaming_filter_accepts_vortex_expression(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [
            {"text": "zero", "label": 0},
            {"text": "one", "label": 1},
            {"text": "two", "label": 0},
        ],
    )

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    filtered = dataset.filter(ve.column("label") == 1)

    assert isinstance(filtered, vx_datasets.VortexIterableDataset)
    assert list(filtered) == [{"label": 1, "text": "one"}]


def test_load_dataset_materializes_to_hf_dataset(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero", "label": 0}, {"text": "one", "label": 1}])

    dataset = vx_datasets.load_dataset(
        tmp_path / "train.vortex",
        split="train",
        streaming=False,
        columns=["text", "label"],
        keep_in_memory=True,
    )

    assert isinstance(dataset, hf_datasets.Dataset)
    assert dataset.to_list() == [{"text": "zero", "label": 0}, {"text": "one", "label": 1}]


def test_streaming_resume_with_limit_reads_full_limit(tmp_path: Path):
    rows: list[dict[str, object]] = [{"idx": i} for i in range(10)]
    write_vortex(tmp_path / "train.vortex", rows)

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train", limit=8, batch_size=2),
    )
    examples = cast(
        vx_datasets._VortexExamplesIterable,  # pyright: ignore[reportPrivateUsage]
        dataset._ex_iterable,  # pyright: ignore[reportPrivateUsage]
    )
    # Simulate resuming after the first two rows of the file were already yielded.
    examples._state_dict = {  # pyright: ignore[reportPrivateUsage]
        "file_idx": 0,
        "file_row_idx": 2,
        "num_yielded": 2,
        "type": type(examples).__name__,
    }

    produced = [row for _key, table in examples._iter_arrow() for row in table.to_pylist()]  # pyright: ignore[reportPrivateUsage]

    # The limit of 8 must still be honored: six rows remain after the two already yielded.
    assert produced == rows[2:8]


def test_streaming_state_dict_resume_mid_batch_is_exact(tmp_path: Path):
    rows: list[dict[str, object]] = [{"idx": i} for i in range(10)]
    write_vortex(tmp_path / "train.vortex", rows)

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train", batch_size=4),
    )
    iterator = iter(dataset)
    # Stop mid-batch: five rows consumed, one row into the second batch of four.
    consumed = [next(iterator) for _ in range(5)]
    state = dataset.state_dict()

    resumed = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train", batch_size=4),
    )
    resumed.load_state_dict(state)

    assert consumed + list(resumed) == rows


def write_sharded_dataset(tmp_path: Path, num_files: int = 4, rows_per_file: int = 3) -> list[int]:
    for file_idx in range(num_files):
        rows: list[dict[str, object]] = [{"idx": file_idx * rows_per_file + i} for i in range(rows_per_file)]
        write_vortex(tmp_path / f"train-{file_idx:04d}.vortex", rows)
    return list(range(num_files * rows_per_file))


def test_streaming_shuffle_multiple_files(tmp_path: Path):
    # Regression test: shuffle() interleaves shards through iterables that read
    # sleep_on_threads_shutdown on every leaf; this used to raise AttributeError.
    indices = write_sharded_dataset(tmp_path)

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path, split="train"),
    )
    shuffled = dataset.shuffle(seed=42, buffer_size=4)

    assert sorted(row["idx"] for row in shuffled) == indices


def test_interleave_vortex_datasets(tmp_path: Path):
    write_vortex(tmp_path / "a.vortex", [{"idx": 0}, {"idx": 1}])
    write_vortex(tmp_path / "b.vortex", [{"idx": 10}, {"idx": 11}])

    left = cast(vx_datasets.VortexIterableDataset, vx_datasets.load_dataset(tmp_path / "a.vortex", split="train"))
    right = cast(vx_datasets.VortexIterableDataset, vx_datasets.load_dataset(tmp_path / "b.vortex", split="train"))
    interleaved = hf_datasets.interleave_datasets([left, right], stopping_strategy="all_exhausted")

    assert sorted(row["idx"] for row in interleaved) == [0, 1, 10, 11]


def test_streaming_take_splits_limit_across_shards(tmp_path: Path):
    _ = write_sharded_dataset(tmp_path)

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path, split="train"),
    )
    limited = dataset.take(4)
    assert isinstance(limited, vx_datasets.VortexIterableDataset)
    examples = limited._ex_iterable  # pyright: ignore[reportPrivateUsage]

    # DataLoader worker / distributed sharding must split the pushed-down limit so the shards
    # together yield exactly take(n) rows, mirroring TakeExamplesIterable.split_number.
    shards = [examples.shard_data_sources(2, index) for index in range(2)]
    rows = [row["idx"] for shard in shards for _key, row in shard]

    assert rows == [0, 1, 6, 7]


def test_streaming_take_then_shuffle_keeps_row_set(tmp_path: Path):
    _ = write_sharded_dataset(tmp_path)

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path, split="train"),
    )
    limited = dataset.take(4)

    # Shuffling after take() may only reorder the taken rows, never select different ones.
    taken = sorted(row["idx"] for row in limited)
    shuffled = sorted(row["idx"] for row in limited.shuffle(seed=7, buffer_size=16))

    assert shuffled == taken


@pytest.mark.skipif(
    not hasattr(hf_datasets.IterableDataset, "reshard"), reason="IterableDataset.reshard requires datasets>=5"
)
def test_streaming_reshard_keeps_all_rows(tmp_path: Path):
    rows: list[dict[str, object]] = [{"idx": i} for i in range(4)]
    write_vortex(tmp_path / "train.vortex", rows)

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )

    assert list(dataset.reshard()) == rows


def test_streaming_filter_expr_with_kwargs_is_rejected(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero", "label": 0}])

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    with pytest.raises(ValueError, match="not supported with a Vortex expression"):
        _ = dataset.filter(ve.column("label") == 1, with_indices=True)


def test_streaming_filter_after_take_is_rejected(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero", "label": 0}, {"text": "one", "label": 1}])

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    limited = dataset.take(1)
    assert isinstance(limited, vx_datasets.VortexIterableDataset)
    with pytest.raises(ValueError, match="after a row limit"):
        _ = limited.filter(ve.column("label") == 1)


def test_streaming_filter_then_take_pushes_down(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [{"text": "a", "label": 0}, {"text": "b", "label": 1}, {"text": "c", "label": 1}],
    )

    dataset = cast(
        vx_datasets.VortexIterableDataset,
        vx_datasets.load_dataset(tmp_path / "train.vortex", split="train"),
    )
    result = dataset.filter(ve.column("label") == 1).take(1)

    assert isinstance(result, vx_datasets.VortexIterableDataset)
    assert list(result) == [{"text": "b", "label": 1}]


def test_streaming_filter_and_limit_combined(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [{"text": "a", "label": 0}, {"text": "b", "label": 1}, {"text": "c", "label": 1}],
    )

    # Vortex cannot scan with a filter and a limit at once; load_dataset must still honor both.
    dataset = vx_datasets.load_dataset(
        tmp_path / "train.vortex", split="train", filter=ve.column("label") == 1, limit=1
    )

    assert isinstance(dataset, vx_datasets.VortexIterableDataset)
    assert list(dataset) == [{"text": "b", "label": 1}]


def test_materialize_filter_and_limit_combined(tmp_path: Path):
    write_vortex(
        tmp_path / "train.vortex",
        [{"text": "a", "label": 0}, {"text": "b", "label": 1}, {"text": "c", "label": 1}],
    )

    dataset = vx_datasets.load_dataset(
        tmp_path / "train.vortex",
        split="train",
        streaming=False,
        filter=ve.column("label") == 1,
        limit=1,
        keep_in_memory=True,
    )

    assert isinstance(dataset, hf_datasets.Dataset)
    assert dataset.to_list() == [{"text": "b", "label": 1}]


def test_materialize_filter_with_num_proc(tmp_path: Path):
    """`num_proc` forks worker processes and pickles the filter into each one."""
    for shard in range(2):
        write_vortex(
            tmp_path / f"train-{shard}.vortex",
            [{"text": f"{shard}-{i}", "label": i % 2} for i in range(50)],
        )

    dataset = vx_datasets.load_dataset(
        tmp_path,
        data_files="*.vortex",
        split="train",
        streaming=False,
        filter=ve.column("label") == 1,
        keep_in_memory=True,
        num_proc=2,
    )

    assert isinstance(dataset, hf_datasets.Dataset)
    rows = dataset.to_list()
    assert len(rows) == 50
    assert {cast(int, row["label"]) for row in rows} == {1}


def test_load_dataset_multi_split_without_mapping_raises(tmp_path: Path):
    write_vortex(tmp_path / "train.vortex", [{"text": "zero"}])

    with pytest.raises(ValueError, match="data_files"):
        _ = vx_datasets.load_dataset(tmp_path, split=["train", "validation"])


def test_load_dataset_streams_from_http_url(tmp_path: Path, http_file_server: str, monkeypatch: pytest.MonkeyPatch):
    # The test server is plain http; object_store only allows https unless ALLOW_HTTP is set.
    monkeypatch.setenv("ALLOW_HTTP", "true")
    rows: list[dict[str, object]] = [{"idx": i, "text": f"row {i}"} for i in range(20)]
    write_vortex(tmp_path / "train.vortex", rows)

    dataset = vx_datasets.load_dataset(f"{http_file_server}/train.vortex", split="train", batch_size=8)

    assert isinstance(dataset, vx_datasets.VortexIterableDataset)
    assert list(dataset.select_columns(["idx"]).take(3)) == [{"idx": 0}, {"idx": 1}, {"idx": 2}]
    assert list(dataset) == rows


def test_load_dataset_url_in_data_files(tmp_path: Path, http_file_server: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALLOW_HTTP", "true")
    write_vortex(tmp_path / "remote.vortex", [{"idx": 0}])

    dataset = vx_datasets.load_dataset(
        tmp_path, data_files={"train": f"{http_file_server}/remote.vortex"}, split="train"
    )

    assert list(dataset) == [{"idx": 0}]


class _FakeHfApi:
    repo_files: ClassVar[list[str]] = ["README.md", "train.vortex", "data/validation.vortex", "notes/readme.txt"]

    def __init__(self, token: bool | str | None = None):
        self.token: bool | str | None = token

    def list_repo_files(self, repo_id: str, *, repo_type: str | None = None, revision: str | None = None) -> list[str]:
        assert repo_id == "org/name"
        assert repo_type == "dataset"
        assert revision in (None, "refs/convert/parquet")
        return list(self.repo_files)


@pytest.mark.parametrize("token", [None, True])
def test_hub_streaming_resolves_to_hf_uris_without_download(token: bool | None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vx_datasets, "HfApi", _FakeHfApi)

    files, store = vx_datasets._resolve_data_files(  # pyright: ignore[reportPrivateUsage]
        "org/name",
        data_files=None,
        split="train",
        revision=None,
        token=token,
        cache_dir=None,
        local_files_only=False,
        streaming=True,
    )

    # Vortex resolves `hf://` itself, including the saved login that `token=True` asks for, so
    # there is no store and no URL building here.
    assert store is None
    assert files == {
        "train": [
            "hf://datasets/org/name/data/validation.vortex",
            "hf://datasets/org/name/train.vortex",
        ]
    }


def test_hub_streaming_with_token_false_forces_anonymous_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vx_datasets, "HfApi", _FakeHfApi)

    files, store = vx_datasets._resolve_data_files(  # pyright: ignore[reportPrivateUsage]
        "org/name",
        data_files=None,
        split="train",
        revision=None,
        token=False,
        cache_dir=None,
        local_files_only=False,
        streaming=True,
    )

    # `token=False` must suppress the credentials Vortex would otherwise read from the environment,
    # which an `hf://` URI cannot express, so it gets an anonymous store instead.
    assert isinstance(store, HfStore)
    assert files == {"train": ["data/validation.vortex", "train.vortex"]}


def test_hub_streaming_with_token_uses_authenticated_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vx_datasets, "HfApi", _FakeHfApi)

    files, store = vx_datasets._resolve_data_files(  # pyright: ignore[reportPrivateUsage]
        "org/name",
        data_files=None,
        split="train",
        revision=None,
        token="hf_fake_token",
        cache_dir=None,
        local_files_only=False,
        streaming=True,
    )

    # An explicitly passed token cannot reach the Vortex reader, so this is the one path that still
    # builds a store; the files are then paths within the repository.
    assert isinstance(store, HfStore)
    assert files == {"train": ["data/validation.vortex", "train.vortex"]}


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("hf://datasets/org/name", ("org/name", None, None)),
        ("hf://datasets/org/name@main", ("org/name", "main", None)),
        ("hf://datasets/org/name/train.vortex", ("org/name", None, "train.vortex")),
        ("hf://datasets/org/name/data/nested/*.vortex", ("org/name", None, "data/nested/*.vortex")),
        (
            "hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/*.vortex",
            ("org/name", "refs/convert/parquet", "data/*.vortex"),
        ),
    ],
)
def test_parse_hf_uri(uri: str, expected: tuple[str, str | None, str | None]):
    assert vx_datasets._parse_hf_uri(uri) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("uri", ["hf://org/name/file.vortex", "hf://datasets/name-only", "hf://datasets/org/@main"])
def test_parse_hf_uri_invalid(uri: str):
    with pytest.raises(ValueError, match="hf://"):
        _ = vx_datasets._parse_hf_uri(uri)  # pyright: ignore[reportPrivateUsage]


def test_hf_uri_streaming_resolves_file_and_directory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vx_datasets, "HfApi", _FakeHfApi)

    def resolve(path: str):
        return vx_datasets._resolve_data_files(  # pyright: ignore[reportPrivateUsage]
            path,
            data_files=None,
            split="train",
            revision=None,
            token=None,
            cache_dir=None,
            local_files_only=False,
            streaming=True,
        )

    files, store = resolve("hf://datasets/org/name/train.vortex")
    assert store is None
    assert files == {"train": ["hf://datasets/org/name/train.vortex"]}

    # A glob-free path naming a directory selects the default Vortex files beneath it.
    files, _store = resolve("hf://datasets/org/name/data")
    assert files == {"train": ["hf://datasets/org/name/data/validation.vortex"]}


def test_hf_uri_slash_revision_stays_percent_encoded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vx_datasets, "HfApi", _FakeHfApi)

    files, store = vx_datasets._resolve_data_files(  # pyright: ignore[reportPrivateUsage]
        "hf://datasets/org/name@refs%2Fconvert%2Fparquet",
        data_files=None,
        split="train",
        revision=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
        streaming=True,
    )

    # A revision containing `/` needs no store of its own any more: Vortex percent-encodes it when
    # it builds the `resolve` URL, so it only has to survive round-tripping through the `hf://` URI.
    assert store is None
    assert files == {
        "train": [
            "hf://datasets/org/name@refs%2Fconvert%2Fparquet/data/validation.vortex",
            "hf://datasets/org/name@refs%2Fconvert%2Fparquet/train.vortex",
        ]
    }


def test_hf_uri_revision_conflict_raises():
    with pytest.raises(ValueError, match="Conflicting revisions"):
        _ = vx_datasets.load_dataset("hf://datasets/org/name@main", revision="other")


def test_hf_uri_with_path_and_data_files_raises():
    with pytest.raises(ValueError, match="data_files"):
        _ = vx_datasets.load_dataset("hf://datasets/org/name/train.vortex", data_files="x.vortex")


def test_hf_url_in_data_files_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="data_files"):
        _ = vx_datasets.load_dataset(tmp_path, data_files={"train": "hf://datasets/org/name/train.vortex"})


def test_local_directory_in_data_files(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    write_vortex(tmp_path / "sub" / "train.vortex", [{"idx": 0}])

    dataset = vx_datasets.load_dataset(tmp_path, data_files={"train": "sub"}, split="train")

    assert list(dataset) == [{"idx": 0}]


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**/*.vortex", "train.vortex", True),
        ("**/*.vortex", "data/nested/train.vortex", True),
        ("**/*.vortex", "train.parquet", False),
        ("*.vortex", "train.vortex", True),
        ("*.vortex", "data/train.vortex", False),
        ("data/**/*.vortex", "data/train.vortex", True),
        ("data/**/*.vortex", "data/a/b/train.vortex", True),
        ("data/**/*.vortex", "other/train.vortex", False),
        ("train-?.vortex", "train-1.vortex", True),
        ("train-[01].vortex", "train-2.vortex", False),
    ],
)
def test_glob_match(pattern: str, path: str, expected: bool):
    assert vx_datasets._glob_match(path, pattern) is expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("repo_type", ["dataset", "datasets", "model", "space"])
def test_hf_store_accepts_each_repo_type(repo_type: str):
    assert isinstance(HfStore("org/name", repo_type=repo_type), HfStore)


def test_hf_store_rejects_unknown_repo_type():
    with pytest.raises(ValueError, match="repository type"):
        _ = HfStore("org/name", repo_type="notarepo")


@pytest.mark.parametrize("token", [None, True, False, "hf_explicit_token"])
def test_hf_store_accepts_each_token_spelling(token: bool | str | None):
    # `token` follows huggingface_hub's convention: None/True use the environment, False forces an
    # anonymous read, and a string is used directly.
    assert isinstance(HfStore("org/name", token=token), HfStore)


def test_hf_store_rejects_unusable_token():
    with pytest.raises(ValueError, match="token"):
        _ = HfStore("org/name", token="bad\nvalue")


def test_hf_store_reads_a_repository_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Point the store at a local server standing in for the Hub, so this exercises the real read
    # path: HfStore is rooted at the repository revision and the path is relative to it.
    monkeypatch.setenv("ALLOW_HTTP", "true")
    rows: list[dict[str, object]] = [{"idx": i} for i in range(4)]
    resolve_dir = tmp_path / "datasets" / "org" / "name" / "resolve" / "main"
    resolve_dir.mkdir(parents=True)
    write_vortex(resolve_dir / "train.vortex", rows)

    handler = type("Handler", (_RangeRequestHandler,), {"directory": tmp_path})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        store = HfStore("org/name", endpoint=endpoint)
        assert vx.open("train.vortex", store=store).to_arrow().read_all().to_pylist() == rows
    finally:
        server.shutdown()
