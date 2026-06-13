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
def vxfile(tmpdir_factory) -> vx.VortexFile:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    if not os.path.exists(fname):  # pyright: ignore[reportUnknownArgumentType]
        a = pa.array([record(x) for x in range(1_000)])
        arr = vx.compress(vx.array(a))
        vx.io.write(arr, str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname))  # pyright: ignore[reportUnknownArgumentType]


def test_execute(vxscan: RepeatedScan):
    for _ in vxscan.execute():
        pass


def test_execute_row_range(vxscan: RepeatedScan):
    total_rows = 0
    for rb in vxscan.execute(row_range=(10, 20)):
        total_rows += len(rb)
    assert total_rows == 10


def test_scalar_at(vxscan: RepeatedScan):
    scalar = vxscan.scalar_at(10)
    assert scalar.as_py() == {
        "index": 10,
        "string": "10",
        "bool": True,
        "float": math.sqrt(10),
    }


def test_scan_with_cast(vxfile: vx.VortexFile):
    actual = vxfile.scan(expr=ve.cast(ve.column("index"), vx.int_(16)) == ve.literal(vx.int_(16), 1)).read_all()
    expected = pa.array(
        [{"index": 1, "string": pa.scalar("1", pa.string_view()), "bool": False, "float": math.sqrt(1)}]
    )
    assert str(actual.to_arrow_array()) == str(expected)


@pytest.mark.parametrize("expr", [ve.column("nonexistent") > 1, ve.column("string") > 1])
def test_scan_filter_bind_errors_raise_python_exception(vxfile: vx.VortexFile, expr: ve.Expr):
    with pytest.raises(RuntimeError):
        vxfile.scan(expr=expr).read_all()


def test_scan_filter_expr_rebinds_against_each_file(tmp_path):
    expr = ve.column("index") > 1

    left_path = tmp_path / "left.vortex"
    left_schema = pa.schema(
        [
            pa.field("index", pa.int64(), nullable=False),
            pa.field("left", pa.string(), nullable=False),
        ]
    )
    left_table = pa.Table.from_arrays(
        [
            pa.array([0, 2, 4], type=pa.int64()),
            pa.array(["a", "b", "c"], type=pa.string()),
        ],
        schema=left_schema,
    )
    vx.io.write(left_table, str(left_path))

    right_path = tmp_path / "right.vortex"
    right_schema = pa.schema(
        [
            pa.field("right", pa.bool_(), nullable=False),
            pa.field("index", pa.int64(), nullable=True),
        ]
    )
    right_table = pa.Table.from_arrays(
        [
            pa.array([False, True, True], type=pa.bool_()),
            pa.array([None, 1, 3], type=pa.int64()),
        ],
        schema=right_schema,
    )
    vx.io.write(right_table, str(right_path))

    left_rows = vx.open(str(left_path)).scan(expr=expr).read_all().to_arrow_array().to_pylist()
    right_rows = vx.open(str(right_path)).scan(expr=expr).read_all().to_arrow_array().to_pylist()

    assert left_rows == [{"index": 2, "left": "b"}, {"index": 4, "left": "c"}]
    assert right_rows == [{"right": True, "index": 3}]


def test_scanner_property_projected(vxfile: vx.VortexFile):
    assert vxfile.to_dataset().scanner(columns=["bool"]).projected_schema == pa.schema([("bool", pa.bool_())])


def test_scanner_property_dataset_schema(vxfile: vx.VortexFile):
    assert vxfile.to_dataset().scanner().dataset_schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )
