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
        vortex.io.write(arr, str(fname))
    return vortex.open(str(fname))


def test_dtype(vxf):
    assert vxf.dtype.to_arrow_schema() == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )


def test_row_count(vxf):
    assert len(vxf) == 1_000_000


def test_scan(vxf):
    vxf.scan()


def test_to_arrow_batch_size(vxf):
    assert len(list(vxf.to_arrow(batch_size=1_000_000))) == 1, "batch_size=1_000_000"
    assert len(list(vxf.to_arrow(batch_size=1_000))) == 1_000, "batch_size=1_000"


def test_to_arrow_columns(vxf):
    rbr = vxf.to_arrow(projection=["string", "bool"])
    assert rbr.schema == pa.schema([("string", pa.string_view()), ("bool", pa.bool_())])
