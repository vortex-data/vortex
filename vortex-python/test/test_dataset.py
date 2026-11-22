# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from collections.abc import Iterator
from pathlib import Path

import duckdb
import polars
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pd
import pytest
import vortex._runtime as vx_runtime
import vortex.dataset as vx_dataset

import vortex as vx


def record(x: int, columns: list[str] | set[str] | None = None) -> dict[str, int | str | float]:
    return {
        k: v
        for k, v in {"index": x, "string": str(x), "bool": x % 2 == 0, "float": math.sqrt(x)}.items()
        if columns is None or k in columns
    }


@pytest.fixture(scope="session")
def ds(tmpdir_factory) -> vx.dataset.VortexDataset:
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"

    assert not os.path.exists(fname)

    a = pa.array([record(x) for x in range(1_000_000)])
    vx.io.write(vx.array(a), str(fname))
    return vx.dataset.VortexDataset.from_path(str(fname))


def test_schema(ds: pd.Dataset) -> None:
    assert ds.schema == pa.schema(
        [("index", pa.int64()), ("string", pa.string_view()), ("bool", pa.bool_()), ("float", pa.float64())]
    )


def test_scanner_schema(ds: vx.dataset.VortexDataset) -> None:
    scanner = vx.dataset.VortexScanner(ds)
    assert scanner.schema == pa.schema(
        [("index", pa.int64()), ("string", pa.string_view()), ("bool", pa.bool_()), ("float", pa.float64())]
    )


def test_head(ds: pd.Dataset) -> None:
    assert ds.head(1).to_pylist() == [{"index": 0, "string": "0", "bool": True, "float": 0.0}]


@pytest.mark.parametrize("num_rows", [0, 2_000_000])
def test_head_matches_to_table(ds: pd.Dataset, num_rows: int) -> None:
    assert ds.head(num_rows) == ds.to_table().slice(0, num_rows)


def test_head_with_fewer_matching_rows_than_requested(ds: pd.Dataset) -> None:
    expression = pc.field("index") >= 999_995
    assert ds.head(10, columns=["index"], filter=expression).column("index").to_pylist() == list(
        range(999_995, 1_000_000)
    )
    assert ds.head(10, filter=pc.field("index") < 0).num_rows == 0


def test_head_with_filter_matching_more_rows_than_requested(ds: pd.Dataset) -> None:
    expression = pc.field("bool")
    expected = ds.to_table(filter=expression).slice(0, 5_000)
    assert ds.head(5_000, filter=expression) == expected


def test_take(ds: pd.Dataset) -> None:
    assert ds.take(pa.array([10, 50, 1_000, 999_999])).to_pylist() == [
        {"index": 10, "string": "10", "bool": True, "float": math.sqrt(10)},
        {"index": 50, "string": "50", "bool": True, "float": math.sqrt(50.0)},
        {"index": 1000, "string": "1000", "bool": True, "float": math.sqrt(1000.0)},
        {"index": 999999, "string": "999999", "bool": False, "float": math.sqrt(999999.0)},
    ]


@pytest.mark.parametrize("indices", [[999_999, 10], [50, 10, 50], [10, 10], [1_000, 10, 999_999, 50]])
def test_take_unordered(ds: pd.Dataset, indices: list[int]) -> None:
    assert ds.take(pa.array(indices)).column("index").to_pylist() == indices


def test_take_unordered_on_filtered_dataset_requires_sorted_indices(ds: vx.dataset.VortexDataset) -> None:
    filtered = ds.filter(pc.field("index") >= 50)
    assert filtered.take(pa.array([50, 999_999])).column("index").to_pylist() == [50, 999_999]
    with pytest.raises(RuntimeError, match="strictly increasing"):
        filtered.take(pa.array([999_999, 10]))


def test_fragment_take_unordered(ds: vx.dataset.VortexDataset) -> None:
    fragment = list(ds.get_fragments())[-1]
    assert fragment.take(pa.array([999_999, 999_998, 999_999])).column("index").to_pylist() == [
        999_999,
        999_998,
        999_999,
    ]
    assert fragment.take(pa.array([999_999, 10, 999_998])).column("index").to_pylist() == [999_999, 999_998]


def test_to_batches(ds: pd.Dataset) -> None:
    assert sum(len(x) for x in ds.to_batches(columns=["float", "bool"])) == 1_000_000

    schema = pa.struct([("string", pa.string_view()), ("bool", pa.bool_())])

    chunk0 = next(ds.to_batches(columns=["string", "bool"]))
    assert chunk0.to_struct_array() == pa.array(
        [record(x, columns=["string", "bool"]) for x in range(len(chunk0))], type=schema
    )


def test_use_threads_configures_worker_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    current_workers = 3
    calls: list[int | None] = []

    def fake_worker_threads() -> int:
        return current_workers

    def fake_set_worker_threads(count: int | None) -> None:
        nonlocal current_workers
        calls.append(count)
        current_workers = 11 if count is None else count

    monkeypatch.setattr(vx_runtime, "_worker_threads", fake_worker_threads)
    monkeypatch.setattr(vx_runtime, "_set_worker_threads", fake_set_worker_threads)
    monkeypatch.setattr(vx_runtime, "_UNTHREADED_LEASES", vx_runtime._UnthreadedLeaseManager())

    with vx_runtime._temporary_worker_threads(True):
        assert current_workers == 3

    assert current_workers == 3
    assert calls == []

    with vx_runtime._temporary_worker_threads(False):
        assert current_workers == 0

    assert current_workers == 3
    assert calls == [0, 3]

    calls.clear()
    reader = pa.RecordBatchReader.from_batches(
        pa.schema([("x", pa.int64())]),
        [
            pa.record_batch([pa.array([1])], names=["x"]),
            pa.record_batch([pa.array([2])], names=["x"]),
        ],
    )

    batches = list(vx_dataset._read_batches_with_temporary_worker_threads(reader, True))

    assert [batch.to_pylist() for batch in batches] == [[{"x": 1}], [{"x": 2}]]
    assert current_workers == 3
    assert calls == []


def _make_two_batch_reader(values: tuple[int, int]) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(
        pa.schema([("x", pa.int64())]),
        [
            pa.record_batch([pa.array([values[0]])], names=["x"]),
            pa.record_batch([pa.array([values[1]])], names=["x"]),
        ],
    )


class _FakeWorkerPool:
    def __init__(self, initial: int) -> None:
        self.workers = initial


def _patch_worker_pool(monkeypatch: pytest.MonkeyPatch, initial: int = 4) -> _FakeWorkerPool:
    pool = _FakeWorkerPool(initial=initial)

    def fake_worker_threads() -> int:
        return pool.workers

    def fake_set_worker_threads(count: int | None) -> None:
        assert count is not None
        pool.workers = count

    monkeypatch.setattr(vx_runtime, "_worker_threads", fake_worker_threads)
    monkeypatch.setattr(vx_runtime, "_set_worker_threads", fake_set_worker_threads)
    monkeypatch.setattr(vx_runtime, "_UNTHREADED_LEASES", vx_runtime._UnthreadedLeaseManager())
    return pool


@pytest.mark.parametrize("close_first_created_first", [True, False])
def test_overlapping_unthreaded_readers_restore_baseline(
    monkeypatch: pytest.MonkeyPatch, close_first_created_first: bool
) -> None:
    pool = _patch_worker_pool(monkeypatch, initial=4)
    gen1 = vx_dataset._read_batches_with_temporary_worker_threads(_make_two_batch_reader((1, 2)), False)
    gen2 = vx_dataset._read_batches_with_temporary_worker_threads(_make_two_batch_reader((3, 4)), False)

    assert next(gen1).to_pylist() == [{"x": 1}]
    assert pool.workers == 0
    assert next(gen2).to_pylist() == [{"x": 3}]
    assert pool.workers == 0

    first_to_close, second_to_close = (gen1, gen2) if close_first_created_first else (gen2, gen1)
    first_to_close.close()
    assert pool.workers == 0
    second_to_close.close()
    assert pool.workers == 4


def test_overlapping_reader_error_still_restores_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _patch_worker_pool(monkeypatch, initial=5)

    class FailingReader:
        def __iter__(self) -> Iterator[pa.RecordBatch]:
            yield pa.record_batch([pa.array([1])], names=["x"])
            raise RuntimeError("boom")

    gen_ok = vx_dataset._read_batches_with_temporary_worker_threads(_make_two_batch_reader((10, 20)), False)
    gen_err = vx_dataset._read_batches_with_temporary_worker_threads(FailingReader(), False)  # ty: ignore[invalid-argument-type]

    assert next(gen_ok).to_pylist() == [{"x": 10}]
    assert pool.workers == 0
    assert next(gen_err).to_pylist() == [{"x": 1}]
    assert pool.workers == 0

    with pytest.raises(RuntimeError, match="boom"):
        next(gen_err)
    assert pool.workers == 0

    gen_ok.close()
    assert pool.workers == 5


def test_reentrant_release_during_acquire_does_not_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakeWorkerPool(initial=4)
    manager = vx_runtime._UnthreadedLeaseManager()
    stale_gen: int | None = None
    triggered = False

    def fake_worker_threads() -> int:
        nonlocal triggered
        if stale_gen is not None and not triggered:
            triggered = True
            manager.release(stale_gen)
        return pool.workers

    def fake_set_worker_threads(count: int | None) -> None:
        assert count is not None
        pool.workers = count

    monkeypatch.setattr(vx_runtime, "_worker_threads", fake_worker_threads)
    monkeypatch.setattr(vx_runtime, "_set_worker_threads", fake_set_worker_threads)
    monkeypatch.setattr(vx_runtime, "_UNTHREADED_LEASES", manager)

    stale_gen = manager.acquire()
    manager.reset_after_fork()
    assert pool.workers == 4

    gen = manager.acquire()
    assert triggered
    assert pool.workers == 0

    manager.release(gen)
    assert pool.workers == 4


@pytest.mark.parametrize("close_first_then_second", [True, False])
def test_public_record_batch_reader_close_restores_worker_count_while_alive(
    ds: vx_dataset.VortexDataset, monkeypatch: pytest.MonkeyPatch, close_first_then_second: bool
) -> None:
    pool = _patch_worker_pool(monkeypatch, initial=4)
    reader1 = ds.to_record_batch_reader(batch_size=1024, use_threads=False)
    reader2 = ds.to_record_batch_reader(batch_size=1024, use_threads=False)

    assert isinstance(reader1, pa.RecordBatchReader)
    assert reader1.read_next_batch().num_rows > 0
    assert pool.workers == 0

    assert reader2.read_next_batch().num_rows > 0
    assert pool.workers == 0

    first_to_close, second_to_close = (reader1, reader2) if close_first_then_second else (reader2, reader1)
    first_to_close.close()
    assert pool.workers == 0

    second_to_close.close()
    assert pool.workers == 4
    # Keep both closed reader objects alive across the restored-count assertion.
    assert reader1 is not None and reader2 is not None

    # Verify native PyArrow consumers (e.g. Scanner.from_batches) accept the returned reader.
    scanner = pd.Scanner.from_batches(ds.to_record_batch_reader(batch_size=1024, use_threads=False))
    assert scanner.to_table().num_rows == ds.count_rows()
    assert pool.workers == 4


@pytest.mark.parametrize("batch_size", [1234, 8192, 1 << 31])
def test_to_batch_size(ds: pd.Dataset, batch_size: int) -> None:
    batch_sizes = [len(x) for x in ds.to_batches(batch_size=batch_size)]
    n_rows = ds.count_rows()
    assert sum(batch_sizes) == n_rows
    if n_rows < batch_size:
        assert batch_sizes == [n_rows]
    if n_rows % batch_size == 0:
        assert batch_sizes == [batch_size for _ in batch_sizes]
    else:
        assert batch_sizes[:-1] == [batch_size for _ in batch_sizes[:-1]]
        assert batch_sizes[-1] == n_rows % batch_size


def test_to_table(ds: pd.Dataset) -> None:
    tbl = ds.to_table(columns=["bool", "float"], filter=pc.field("float") > 100)
    # TODO(aduffy): add back once pyarrow supports casting to/from string_view
    # assert 0 == len(tbl.filter(pc.field("string") <= "10000"))
    assert tbl.slice(0, 10) == pa.Table.from_struct_array(
        pa.array([record(x, columns={"float", "bool"}) for x in range(10001, 10011)])
    )

    assert ds.to_table(columns=["bool", "string"]).schema == pa.schema(
        [("bool", pa.bool_()), ("string", pa.string_view())]
    )
    assert ds.to_table(columns=["string", "bool"]).schema == pa.schema(
        [("string", pa.string_view()), ("bool", pa.bool_())]
    )


def test_to_record_batch_reader_with_polars(ds: pd.Dataset) -> None:
    pldf = polars.scan_pyarrow_dataset(ds).collect()
    assert len(pldf) == 1_000_000
    assert pldf.schema["index"] == polars.Int64
    assert pldf.schema["string"] == polars.Utf8
    assert pldf.schema["bool"] == polars.Boolean
    assert pldf.schema["float"] == polars.Float64


def test_filter(ds: vx.dataset.VortexDataset) -> None:
    tbl = ds.to_table(filter=(pc.field("string") >= "950000") & (pc.field("float") < 975.0))
    assert len(tbl) == 6176

    tbl = ds.to_table(filter=(pc.field("index") < 10))
    assert len(tbl) == 10

    tbl = ds.to_table(filter=((pc.field("index") + 1) < 10))
    assert len(tbl) == 9

    tbl = ds.to_table(filter=((pc.field("index") - 1) < 10))
    assert len(tbl) == 11

    tbl = ds.to_table(filter=((pc.field("index") * 2) < 10))
    assert len(tbl) == 5

    tbl = ds.to_table(filter=((pc.field("index") / 2) < 10))
    assert len(tbl) == 20


def test_filter_with_nested_null_dtype(tmp_path: Path) -> None:
    path = tmp_path / "test.vortex"

    batch = pa.RecordBatch.from_pylist(
        [
            {"a": 0, "b": {"x": None}},
            {"a": 1, "b": {"x": None}},
        ]
    )

    arr = vx.array(batch.to_struct_array())
    vx.io.write(vx.ArrayIterator.from_iter(arr.dtype, iter([arr])), str(path))

    dataset = vx.open(str(path)).to_dataset()
    actual = dataset.to_table(filter=pc.field("a") == 0)

    assert actual.to_pylist() == [{"a": 0, "b": {"x": None}}]


def test_duckdb(ds: vx.dataset.VortexDataset) -> None:
    _ = ds  # DuckDB resolves the dataset from the SQL query below.
    conn = duckdb.default_connection()

    tbl = conn.sql("select * from ds where string >= '950000' and float < 975.0").to_arrow_table()
    assert len(tbl) == 6176
    assert tbl.schema == pa.schema(
        [("index", pa.int64()), ("string", pa.utf8()), ("bool", pa.bool_()), ("float", pa.float64())]
    )

    tbl = conn.sql("select * from ds").to_arrow_table()
    assert len(tbl) == 1_000_000
    assert tbl.schema == pa.schema(
        [("index", pa.int64()), ("string", pa.utf8()), ("bool", pa.bool_()), ("float", pa.float64())]
    )
    assert tbl.take([0]).to_pylist()[0] == record(0)
    assert tbl.take([950_000]).to_pylist()[0] == record(950_000)

    tbl = conn.sql("select string as hi_mom, float as yolo from ds").to_arrow_table()
    assert len(tbl) == 1_000_000
    assert tbl.schema == pa.schema([("hi_mom", pa.utf8()), ("yolo", pa.float64())])


def test_fragment_schema(ds: vx.dataset.VortexDataset) -> None:
    fragments = ds.get_fragments()
    for i, f in enumerate(fragments):
        assert f.physical_schema == pa.schema(
            [("index", pa.int64()), ("string", pa.string_view()), ("bool", pa.bool_()), ("float", pa.float64())]
        ), (f, i)

    assert ds.head(1).to_pylist() == [{"index": 0, "string": "0", "bool": True, "float": 0.0}]


def test_fragment_reads_accept_only_the_dataset_schema(ds: vx.dataset.VortexDataset) -> None:
    fragment = next(iter(ds.get_fragments()))
    expected = fragment.to_table()

    # DataFusion passes the dataset schema to unify it with the fragment schema.
    assert fragment.scanner(ds.schema).to_table() == expected
    assert fragment.to_table(ds.schema) == expected
    assert pa.Table.from_batches(fragment.to_batches(ds.schema)) == expected

    other = pa.schema([("index", pa.int32())])
    with pytest.raises(ValueError, match="schema of the dataset"):
        fragment.scanner(other)
    with pytest.raises(ValueError, match="schema of the dataset"):
        fragment.to_table(other)
    with pytest.raises(ValueError, match="schema of the dataset"):
        list(fragment.to_batches(other))


def test_fragment_take(ds: vx.dataset.VortexDataset) -> None:
    fragments = list(ds.get_fragments())
    assert fragments[0].take(pa.array([10, 50, 1_000])).to_pylist() == [
        {"index": 10, "string": "10", "bool": True, "float": math.sqrt(10)},
        {"index": 50, "string": "50", "bool": True, "float": math.sqrt(50.0)},
        {"index": 1000, "string": "1000", "bool": True, "float": math.sqrt(1000.0)},
    ]

    for f in fragments[1:-1]:
        assert f.take(pa.array([10, 50, 1_000, 999_999])).to_pylist() == []

    assert fragments[-1].take(pa.array([999_999])).to_pylist() == [
        {"index": 999999, "string": "999999", "bool": False, "float": math.sqrt(999999.0)},
    ]


def test_fragment_to_batches(ds: vx.dataset.VortexDataset) -> None:
    fragments = list(ds.get_fragments())

    assert sum(len(x) for f in fragments for x in f.to_batches(columns=["float", "bool"])) == 1_000_000

    schema = pa.struct([("string", pa.string_view()), ("bool", pa.bool_())])

    chunk0 = next(fragments[0].to_batches(columns=["string", "bool"]))
    assert chunk0.to_struct_array() == pa.array(
        [record(x, columns=["string", "bool"]) for x in range(len(chunk0))], type=schema
    )


@pytest.mark.parametrize("batch_size", [1234, 8192, 1 << 31])
def test_fragment_to_batch_size(ds: vx.dataset.VortexDataset, batch_size: int) -> None:
    fragments = list(ds.get_fragments())

    remainder = 0
    for f in fragments:
        # The fragments have ranges based on the natural splits.
        # We read batches in units of `batch_size` though.
        batch_sizes = [len(batch) for batch in f.to_batches(batch_size=batch_size)]
        n_rows = f.count_rows() - remainder
        assert sum(batch_sizes) == n_rows

        if n_rows < batch_size:
            assert batch_sizes == [n_rows]
        elif n_rows % batch_size == 0:
            assert batch_sizes == [batch_size for _ in batch_sizes]
        else:
            last_batch_size = n_rows % batch_size
            assert batch_sizes[:-1] == [batch_size for _ in batch_sizes[:-1]]
            assert batch_sizes[-1] == last_batch_size


def test_fragment_to_table(ds: vx.dataset.VortexDataset) -> None:
    fragments = list(ds.get_fragments())

    frag_row_count = 0

    for f in fragments:
        assert f.to_table(columns=["bool", "string"]).schema == pa.schema(
            [("bool", pa.bool_()), ("string", pa.string_view())]
        )
        assert f.to_table(columns=["string", "bool"]).schema == pa.schema(
            [("string", pa.string_view()), ("bool", pa.bool_())]
        )

        frag_row_count += len(f.to_table(columns=["bool", "float"], filter=pc.field("float") > 100))

    assert frag_row_count == 989_999


def test_get_fragments(ds: vx.dataset.VortexDataset) -> None:
    assert ds.count_rows() == 1_000_000

    assert ds.count_rows() == sum(f.count_rows() for f in ds.get_fragments())

    filter_expr = vx.expr.column("string") > "5"
    assert ds.count_rows(filter=filter_expr) == sum(f.count_rows(filter=filter_expr) for f in ds.get_fragments())

    ds_filtered = ds.filter(filter_expr)
    assert ds_filtered.count_rows() == sum(f.count_rows() for f in ds_filtered.get_fragments())

    assert ds.to_table() == pa.concat_tables(f.to_table() for f in ds.get_fragments())
    assert ds_filtered.to_table() == pa.concat_tables(f.to_table() for f in ds_filtered.get_fragments())


@pytest.mark.parametrize(
    "expression",
    [
        pc.field("x").isin([1, 3]),
        ~pc.field("x").isin([1, 3]),
        pc.field("x").isin(pa.array([], pa.int32())),
        pc.field("s").isin(["a", "c"]),
        pc.field("f").isin([1.5, 3.5]),
    ],
)
def test_filter_is_in(tmp_path: Path, expression: pc.Expression) -> None:
    table = pa.table(
        {
            "x": pa.array([1, 2, None, 3], pa.int32()),
            "s": ["a", "b", None, "c"],
            "f": [1.5, 2.5, None, 3.5],
        }
    )
    path = tmp_path / "is_in.vortex"
    vx.io.write(vx.array(table), str(path))

    actual = vx.open(str(path)).to_dataset().to_table(filter=expression)
    expected = pd.dataset(table).to_table(filter=expression)
    assert actual.to_pylist() == expected.to_pylist()


def test_filter_is_in_with_a_null_member_is_unsupported(tmp_path: Path) -> None:
    table = pa.table({"x": pa.array([1, None], pa.int32())})
    path = tmp_path / "is_in_null.vortex"
    vx.io.write(vx.array(table), str(path))

    # Substrait does not carry whether PyArrow skips nulls, which decides a null row.
    with pytest.raises(NotImplementedError):
        vx.open(str(path)).to_dataset().to_table(filter=pc.field("x").isin([1, None]))


def test_zero_batch_size_rejected(ds: vx.dataset.VortexDataset) -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.to_record_batch_reader(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        list(ds.to_batches(batch_size=0))
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.count_rows(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        ds.count_rows(filter=pc.field("index") >= 0, batch_size=0)
