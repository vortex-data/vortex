# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

import duckdb
import polars
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pd
import pytest

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
    arr = vx.compress(vx.array(a))
    vx.io.write(arr, str(fname))  # pyright: ignore[reportUnknownArgumentType]
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
    assert tbl.slice(0, 10) == pa.Table.from_struct_array(  # pyright: ignore[reportUnknownMemberType]
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


def test_duckdb(ds: vx.dataset.VortexDataset):
    assert ds  # pyright cannot determine that ds is used by duckdb.execute
    # This would be a nice test but we do not support IsNotNull which duckdb uses
    # tbl = duckdb.execute("select * from ds where string >= '950000' and float < 975.0").arrow()
    # assert len(tbl) == 10_000
    # assert tbl.schema == pa.schema(
    #     [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.utf8())]
    # )

    tbl = duckdb.execute("select * from ds").arrow()
    assert len(tbl) == 1_000_000
    assert tbl.schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.utf8())]
    )
    assert tbl.take([0]).to_pylist()[0] == record(0)
    assert tbl.take([950_000]).to_pylist()[0] == record(950_000)

    tbl = duckdb.execute("select string as hi_mom, float as yolo from ds").arrow()
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
    assert len(fragments) == 1
    f = fragments[0]
    assert f.take(pa.array([10, 50, 1_000, 999_999])).to_pylist() == [
        {"index": 10, "string": "10", "bool": True, "float": math.sqrt(10)},
        {"index": 50, "string": "50", "bool": True, "float": math.sqrt(50.0)},
        {"index": 1000, "string": "1000", "bool": True, "float": math.sqrt(1000.0)},
        {"index": 999999, "string": "999999", "bool": False, "float": math.sqrt(999999.0)},
    ]


def test_fragment_to_batches(ds: vx.dataset.VortexDataset):
    fragments = list(ds.get_fragments())
    assert len(fragments) == 1
    f = fragments[0]

    assert sum(len(x) for x in f.to_batches(columns=["float", "bool"])) == 1_000_000

    schema = pa.struct([("string", pa.string_view()), ("bool", pa.bool_())])

    chunk0 = next(f.to_batches(columns=["string", "bool"]))
    assert chunk0.to_struct_array() == pa.array(
        [record(x, columns=["string", "bool"]) for x in range(len(chunk0))], type=schema
    )


@pytest.mark.parametrize("batch_size", [1234, 8192, 1 << 31])
def test_fragment_to_batch_size(ds: vx.dataset.VortexDataset, batch_size: int):
    fragments = list(ds.get_fragments())
    assert len(fragments) == 1
    f = fragments[0]

    batch_sizes = [len(x) for x in f.to_batches(batch_size=batch_size)]
    n_rows = f.count_rows()
    if n_rows < batch_size:
        assert batch_sizes == [n_rows]
    if n_rows % batch_size == 0:
        assert batch_sizes == [batch_size for _ in batch_sizes]
    else:
        assert batch_sizes[:-1] == [batch_size for _ in batch_sizes[:-1]]
        assert batch_sizes[-1] == n_rows % batch_size


def test_fragment_to_table(ds: vx.dataset.VortexDataset):
    fragments = list(ds.get_fragments())
    assert len(fragments) == 1
    f = fragments[0]

    tbl = f.to_table(columns=["bool", "float"], filter=pc.field("float") > 100)
    assert tbl.slice(0, 10) == pa.Table.from_struct_array(  # pyright: ignore[reportUnknownMemberType]
        pa.array([record(x, columns={"float", "bool"}) for x in range(10001, 10011)])
    )

    assert f.to_table(columns=["bool", "string"]).schema == pa.schema(
        [("bool", pa.bool_()), ("string", pa.string_view())]
    )
    assert f.to_table(columns=["string", "bool"]).schema == pa.schema(
        [("string", pa.string_view()), ("bool", pa.bool_())]
    )
