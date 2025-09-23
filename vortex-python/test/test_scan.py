# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

import pyarrow as pa
import pytest

import vortex as vx
from vortex.scan import RepeatedScan


def record(x: int, columns: list[str] | set[str] | None = None) -> dict[str, int | str | float]:
    return {
        k: v
        for k, v in {"index": x, "string": str(x), "bool": x % 2 == 0, "float": math.sqrt(x)}.items()
        if columns is None or k in columns
    }


@pytest.fixture(scope="session")
def vxscan(tmpdir_factory) -> vx.RepeatedScan:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    if not os.path.exists(fname):  # pyright: ignore[reportUnknownArgumentType]
        a = pa.array([record(x) for x in range(1_000)])
        arr = vx.compress(vx.array(a))
        vx.io.write(arr, str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname)).to_repeated_scan()  # pyright: ignore[reportUnknownArgumentType]


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
