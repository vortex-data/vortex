# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from pathlib import Path

import duckdb
import polars
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pd
import pytest
import vortex.dataset as vx_dataset

import vortex as vx


def record(x: int, columns: list[str] | set[str] | None = None) -> dict[str, int | str | float]:
    return {
        k: v
        for k, v in {"index": x, "string": str(x), "bool": x % 2 == 0, "float": math.sqrt(x)}.items()
        if columns is None or k in columns
    }


@pytest.fixture(scope="session")
def ds(tmpdir_factory) -> vx.dataset.VortexDataset:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    assert not os.path.exists(fname)  # pyright: ignore[reportUnknownArgumentType]

    a = pa.array([record(x) for x in range(1_000_000)])
    vx.io.write(vx.array(a), str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.dataset.VortexDataset.from_path(str(fname))  # pyright: ignore[reportUnknownArgumentType]


def test_schema(ds: pd.Dataset):
    assert ds.schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )


def test_scanner_schema(ds: vx.dataset.VortexDataset):
    scanner = vx.dataset.VortexScanner(ds)
    assert scanner.schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )


def test_head(ds: pd.Dataset):
    assert ds.head(1).to_pylist() == [{"index": 0, "string": "0", "bool": True, "float": 0.0}]


def test_take(ds: pd.Dataset):
    assert ds.take(pa.array([10, 50, 1_000, 999_999])).to_pylist() == [
        {"index": 10, "string": "10", "bool": True, "float": math.sqrt(10)},
        {"index": 50, "string": "50", "bool": True, "float": math.sqrt(50.0)},
        {"index": 1000, "string": "1000", "bool": True, "float": math.sqrt(1000.0)},
        {"index": 999999, "string": "999999", "bool": False, "float": math.sqrt(999999.0)},
    ]


def test_to_batches(ds: pd.Dataset):
    assert sum(len(x) for x in ds.to_batches(columns=["float", "bool"])) == 1_000_000

    schema = pa.struct([("string", pa.string_view()), ("bool", pa.bool_())])

    chunk0 = next(ds.to_batches(columns=["string", "bool"]))
    assert chunk0.to_struct_array() == pa.array(
        [record(x, columns=["string", "bool"]) for x in range(len(chunk0))], type=schema
    )


def test_use_threads_configures_worker_pool(monkeypatch: pytest.MonkeyPatch):
    current_workers = 3
    calls: list[int | None] = []

    def fake_worker_threads() -> int:
        return current_workers

    def fake_set_worker_threads(count: int | None) -> None:
        nonlocal current_workers
        calls.append(count)
        current_workers = 11 if count is None else count

    monkeypatch.setattr(vx_dataset, "_worker_threads", fake_worker_threads)
    monkeypatch.setattr(vx_dataset, "_set_worker_threads", fake_set_worker_threads)

    with vx_dataset._temporary_worker_threads(True):  # pyright: ignore[reportPrivateUsage]
        assert current_workers == 11

    assert current_workers == 3

    with vx_dataset._temporary_worker_threads(False):  # pyright: ignore[reportPrivateUsage]
        assert current_workers == 0

    assert current_workers == 3
    assert calls == [None, 3, 0, 3]

    calls.clear()
    reader = pa.RecordBatchReader.from_batches(
        pa.schema([("x", pa.int64())]),
        [
            pa.record_batch([pa.array([1])], names=["x"]),
            pa.record_batch([pa.array([2])], names=["x"]),
        ],
    )

    batches = list(vx_dataset._read_batches_with_temporary_worker_threads(reader, True))  # pyright: ignore[reportPrivateUsage]

    assert [batch.to_pylist() for batch in batches] == [[{"x": 1}], [{"x": 2}]]
    assert current_workers == 3
    assert calls == [None, 3]


@pytest.mark.parametrize("batch_size", [1234, 8192, 1 << 31])
def test_to_batch_size(ds: pd.Dataset, batch_size: int):
    batch_sizes = [len(x) for x in ds.to_batches(batch_size=batch_size)]
    n_rows = ds.count_rows()
    if n_rows < batch_size:
        assert batch_sizes == [n_rows]
    if n_rows % batch_size == 0:
        assert batch_sizes == [batch_size for _ in batch_sizes]
    else:
        assert batch_sizes[:-1] == [batch_size for _ in batch_sizes[:-1]]
        assert batch_sizes[-1] == n_rows % batch_size


def test_to_table(ds: pd.Dataset):
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


def test_to_record_batch_reader_with_polars(ds: pd.Dataset):
    pldf = polars.scan_pyarrow_dataset(ds).collect()  # pyright: ignore[reportUnknownMemberType]
    assert len(pldf) == 1_000_000
    assert pldf.schema["index"] == polars.Int64
    assert pldf.schema["string"] == polars.Utf8
    assert pldf.schema["bool"] == polars.Boolean
    assert pldf.schema["float"] == polars.Float64


def test_filter(ds: vx.dataset.VortexDataset):
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


def test_filter_with_nested_null_dtype(tmp_path: Path):
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


def test_duckdb(ds: vx.dataset.VortexDataset):
    assert ds  # pyright cannot determine that ds is used by duckdb.execute

    tbl = duckdb.execute("select * from ds where string >= '950000' and float < 975.0").arrow().read_all()
    assert len(tbl) == 6176
    assert tbl.schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.utf8())]
    )

    tbl = duckdb.execute("select * from ds").arrow().read_all()
    assert len(tbl) == 1_000_000
    assert tbl.schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.utf8())]
    )
    assert tbl.take([0]).to_pylist()[0] == record(0)
    assert tbl.take([950_000]).to_pylist()[0] == record(950_000)

    tbl = duckdb.execute("select string as hi_mom, float as yolo from ds").arrow().read_all()
    assert len(tbl) == 1_000_000
    assert tbl.schema == pa.schema([("hi_mom", pa.utf8()), ("yolo", pa.float64())])


def test_fragment_schema(ds: vx.dataset.VortexDataset):
    fragments = ds.get_fragments()
    for i, f in enumerate(fragments):
        assert f.physical_schema == pa.schema(
            [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
        ), (f, i)

    assert ds.head(1).to_pylist() == [{"index": 0, "string": "0", "bool": True, "float": 0.0}]


def test_fragment_take(ds: vx.dataset.VortexDataset):
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


def test_fragment_to_batches(ds: vx.dataset.VortexDataset):
    fragments = list(ds.get_fragments())

    assert sum(len(x) for f in fragments for x in f.to_batches(columns=["float", "bool"])) == 1_000_000

    schema = pa.struct([("string", pa.string_view()), ("bool", pa.bool_())])

    chunk0 = next(fragments[0].to_batches(columns=["string", "bool"]))
    assert chunk0.to_struct_array() == pa.array(
        [record(x, columns=["string", "bool"]) for x in range(len(chunk0))], type=schema
    )


@pytest.mark.parametrize("batch_size", [1234, 8192, 1 << 31])
def test_fragment_to_batch_size(ds: vx.dataset.VortexDataset, batch_size: int):
    fragments = list(ds.get_fragments())

    remainder = 0
    for f in fragments:
        # The fragments have ranges based on the natural splits.
        # We read batches in units of `batch_size` though.
        batch_sizes = [len(batch) for batch in f.to_batches(batch_size=batch_size)]
        n_rows = f.count_rows() - remainder

        if n_rows < batch_size:
            assert batch_sizes == [n_rows]
        elif n_rows % batch_size == 0:
            assert batch_sizes == [batch_size for _ in batch_sizes]
        else:
            last_batch_size = n_rows % batch_size
            assert batch_sizes[:-1] == [batch_size for _ in batch_sizes[:-1]]
            assert batch_sizes[-1] == last_batch_size


def test_fragment_to_table(ds: vx.dataset.VortexDataset):
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


def test_get_fragments(ds: vx.dataset.VortexDataset):
    assert len(list(ds.get_fragments())) == 26

    assert ds.count_rows() == sum(f.count_rows() for f in ds.get_fragments())

    filter_expr = vx.expr.column("string") > "5"
    assert ds.count_rows(filter=filter_expr) == sum(f.count_rows(filter=filter_expr) for f in ds.get_fragments())

    ds_filtered = ds.filter(filter_expr)
    assert ds_filtered.count_rows() == sum(f.count_rows() for f in ds_filtered.get_fragments())

    assert ds.to_table() == pa.concat_tables(f.to_table() for f in ds.get_fragments())
    assert ds_filtered.to_table() == pa.concat_tables(f.to_table() for f in ds_filtered.get_fragments())
