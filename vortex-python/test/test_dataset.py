import math
import os

import duckdb
import polars
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import vortex as vx
import vortex.dataset
import vortex.io


def record(x: int, columns=None) -> dict:
    return {
        k: v
        for k, v in {"index": x, "string": str(x), "bool": x % 2 == 0, "float": math.sqrt(x)}.items()
        if columns is None or k in columns
    }


@pytest.fixture(scope="session")
def ds(tmpdir_factory) -> vortex.dataset.VortexDataset:
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"

    if not os.path.exists(fname):
        a = pa.array([record(x) for x in range(1_000_000)])
        arr = vx.compress(vx.array(a))
        vortex.io.write(arr, str(fname))
    return vortex.dataset.VortexDataset.from_path(str(fname))


def test_schema(ds):
    assert ds.schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )


def test_head(ds):
    assert ds.head(1).to_pylist() == [{"index": 0, "string": "0", "bool": True, "float": 0.0}]


def test_take(ds):
    assert ds.take(pa.array([10, 50, 1_000, 999_999])).to_pylist() == [
        {"index": 10, "string": "10", "bool": True, "float": math.sqrt(10)},
        {"index": 50, "string": "50", "bool": True, "float": math.sqrt(50.0)},
        {"index": 1000, "string": "1000", "bool": True, "float": math.sqrt(1000.0)},
        {"index": 999999, "string": "999999", "bool": False, "float": math.sqrt(999999.0)},
    ]


def test_to_batches(ds):
    assert sum(len(x) for x in ds.to_batches(columns=["float", "bool"])) == 1_000_000

    schema = pa.struct([("string", pa.string_view()), ("bool", pa.bool_())])

    chunk0 = next(ds.to_batches(columns=["string", "bool"]))
    assert chunk0.to_struct_array() == pa.array(
        [record(x, columns=["string", "bool"]) for x in range(len(chunk0))], type=schema
    )


def test_to_table(ds):
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


def test_to_record_batch_reader_with_polars(ds):
    pldf = polars.scan_pyarrow_dataset(ds).collect()
    assert len(pldf) == 1_000_000
    assert pldf.schema["index"] == polars.Int64
    assert pldf.schema["string"] == polars.Utf8
    assert pldf.schema["bool"] == polars.Boolean
    assert pldf.schema["float"] == polars.Float64


def test_to_record_batch_reader_with_duckdb(ds):
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
