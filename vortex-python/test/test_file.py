# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

import pyarrow as pa
import pytest

import vortex as vx
from vortex.file import VortexFile


def record(x: int, columns: list[str] | set[str] | None = None) -> dict[str, int | str | float]:
    return {
        k: v
        for k, v in {"index": x, "string": str(x), "bool": x % 2 == 0, "float": math.sqrt(x)}.items()
        if columns is None or k in columns
    }


@pytest.fixture(scope="session")
def vxf(tmpdir_factory) -> vx.VortexFile:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    if not os.path.exists(fname):  # pyright: ignore[reportUnknownArgumentType]
        a = pa.array([record(x) for x in range(1_000_000)])
        arr = vx.compress(vx.array(a))
        vx.io.write(arr, str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname), without_segment_cache=True)  # pyright: ignore[reportUnknownArgumentType]


def test_dtype(vxf: VortexFile):
    assert vxf.dtype.to_arrow_schema() == pa.schema(
        [("bool", pa.bool_()), ("float", pa.float64()), ("index", pa.int64()), ("string", pa.string_view())]
    )


def test_row_count(vxf: VortexFile):
    assert len(vxf) == 1_000_000


def test_scan(vxf: VortexFile):
    for _ in vxf.scan():
        pass


def test_scan_with_indices(vxf: VortexFile):
    total_rows = 0
    for rb in vxf.scan(indices=vx.array([1, 10, 1_000, 999_999])):
        total_rows += len(rb)
    assert total_rows == 4


def test_to_arrow_batch_size(vxf: VortexFile):
    assert len(list(vxf.to_arrow(batch_size=1_000_000))) == 1, "batch_size=1_000_000"
    assert len(list(vxf.to_arrow(batch_size=1_000))) == 1_000, "batch_size=1_000"


def test_to_arrow_columns(vxf: VortexFile):
    rbr = vxf.to_arrow(projection=["string", "bool"])
    assert rbr.schema == pa.schema([("string", pa.string_view()), ("bool", pa.bool_())])


def test_empty_file(tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    # test for writing empty files with null columns
    # create an empty table with schema `empty: null`
    table = pa.Table.from_pydict({"empty": []})
    assert repr(table.schema) == "empty: null"

    # cast to Vortex array
    empty = vx.array(table)
    assert len(empty) == 0
    assert repr(empty.dtype) == 'struct({"empty": null()}, nullable=False)'

    # writing file should succeed
    empty_file = tmpdir_factory.mktemp("data") / "empty.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    vx.io.write(empty, str(empty_file))  # pyright: ignore[reportUnknownArgumentType]


def test_stream_pyarrow(tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    import pyarrow.parquet as pq

    data_dir = tmpdir_factory.mktemp("data")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    table = pa.Table.from_pydict(
        {
            "names": ["Alice", "Bob", "Carol"],
            "ages": [21, 22, 23],
        }
    )
    pq.write_table(table, str(data_dir / "names.parquet"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    df = pq.read_table(str(data_dir / "names.parquet"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    vx.io.write(df, str(data_dir / "names.vortex"))  # pyright: ignore[reportUnknownArgumentType]
