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


def test_scanner_property_projected(vxfile: vx.VortexFile):
    assert vxfile.to_dataset().scanner(columns=["bool"]).projected_schema == pa.schema([("bool", pa.bool_())])


def test_scanner_property_dataset_schema(vxfile: vx.VortexFile):
    assert vxfile.to_dataset().scanner().dataset_schema == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )
