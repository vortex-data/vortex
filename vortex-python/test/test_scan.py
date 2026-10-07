# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve
from vortex.scan import RepeatedScan


def record(x: int, columns: list[str] | set[str] | None = None) -> dict[str, int | str | float]:
    return {
        k: v
        for k, v in {"index": x, "string": str(x), "bool": x % 2 == 0, "float": math.sqrt(x)}.items()
        if columns is None or k in columns
    }


@pytest.fixture(scope="session")
def vxscan(vxfile: vx.VortexFile) -> vx.RepeatedScan:
    return vxfile.to_repeated_scan()


@pytest.fixture(scope="session")
def vxfile(tmpdir_factory) -> vx.VortexFile:
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"

    if not os.path.exists(fname):
        a = pa.array([record(x) for x in range(1_000)])
        arr = vx.compress(vx.array(a))
        vx.io.write(arr, str(fname))
    return vx.open(str(fname))


def test_execute(vxscan: RepeatedScan) -> None:
    for _ in vxscan.execute():
        pass


def test_execute_row_range(vxscan: RepeatedScan) -> None:
    total_rows = 0
    for rb in vxscan.execute(row_range=(10, 20)):
        total_rows += len(rb)
    assert total_rows == 10


def test_scalar_at(vxscan: RepeatedScan) -> None:
    scalar = vxscan.scalar_at(10)
    assert scalar.as_py() == {
        "index": 10,
        "string": "10",
        "bool": True,
        "float": math.sqrt(10),
    }


def test_scan_with_cast(vxfile: vx.VortexFile) -> None:
    actual = vxfile.scan(expr=ve.cast(ve.column("index"), vx.int_(16)) == ve.literal(vx.int_(16), 1)).read_all()
    expected = pa.array(
        [{"index": 1, "string": pa.scalar("1", pa.string_view()), "bool": False, "float": math.sqrt(1)}]
    )
    assert str(actual.to_arrow_array()) == str(expected)


def test_scanner_property_projected(vxfile: vx.VortexFile) -> None:
    assert vxfile.to_dataset().scanner(columns=["bool"]).projected_schema == pa.schema([("bool", pa.bool_())])


def test_scanner_property_dataset_schema(vxfile: vx.VortexFile) -> None:
    assert vxfile.to_dataset().scanner().dataset_schema == pa.schema(
        [("index", pa.int64()), ("string", pa.string_view()), ("bool", pa.bool_()), ("float", pa.float64())]
    )


@pytest.mark.parametrize("row_range", [None, (1_234, 387_654)])
def test_to_arrow_preserves_split_order(tmp_path, row_range: tuple[int, int] | None) -> None:
    # Several splits, so Arrow conversions run concurrently and must still come back in order.
    fname = str(tmp_path / "many_splits.vortex")
    n = 500_000
    vx.io.write(
        pa.table({"index": pa.array(range(n), type=pa.int64()), "string": [str(x * 7919) for x in range(n)]}),
        fname,
    )
    vxf = vx.open(fname)

    if row_range is None:
        reader = vxf.scan().to_arrow()
        expected = list(range(n))
    else:
        reader = vxf.to_repeated_scan().execute(row_range=row_range).to_arrow()
        expected = list(range(*row_range))

    batches = list(reader)
    assert len(batches) > 1
    assert pa.Table.from_batches(batches).column("index").to_pylist() == expected


def test_to_arrow_from_python_iterator() -> None:
    chunks = [vx.array(pa.table({"index": pa.array([i, i + 1], type=pa.int64())})) for i in range(0, 20, 2)]
    reader = vx.ArrayIterator.from_iter(chunks[0].dtype, iter(chunks)).to_arrow()
    assert pa.Table.from_batches(list(reader)).column("index").to_pylist() == list(range(20))


def test_to_arrow_with_schema(vxfile: vx.VortexFile) -> None:
    schema = pa.schema([("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string())])
    table = pa.Table.from_batches(list(vxfile.scan(["bool", "float", "index", "string"]).to_arrow(schema=schema)))
    assert table.schema == schema
    assert table.column("string").to_pylist() == [str(x) for x in range(1_000)]
