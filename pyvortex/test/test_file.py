import math
import os

import pyarrow as pa
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
def vxf(tmpdir_factory) -> vortex.VortexFile:
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"

    if not os.path.exists(fname):
        a = pa.array([record(x) for x in range(1_000_000)])
        arr = vx.compress(vx.array(a))
        vortex.io.write_path(arr, str(fname))
    return vortex.open(str(fname))


def test_schema(vxf):
    assert vxf.dtype.to_arrow_schema() == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )


def test_head(vxf):
    rr: pa.RecordBatchReader = vxf.to_arrow()
    assert isinstance(rr, pa.RecordBatchReader)
    tbl = rr.read_all()
    print(tbl)


def test_take(vxf):
    assert vxf.take(pa.array([10, 50, 1_000, 999_999])).to_pylist() == [
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
