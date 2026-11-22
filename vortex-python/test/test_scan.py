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
    rows = [row for batch in vxscan.execute() for row in batch.to_pylist()]
    assert rows == [record(x) for x in range(1_000)]


def test_execute_row_range(vxscan: RepeatedScan) -> None:
    rows = [row for batch in vxscan.execute(row_range=(10, 20)) for row in batch.to_pylist()]
    assert rows == [record(x) for x in range(10, 20)]


CHUNKED_ROWS = 300_000


@pytest.fixture(scope="session")
def chunked_vxfile(tmpdir_factory) -> vx.VortexFile:
    """A file big enough to be written as several chunks, unlike a small table."""
    fname = tmpdir_factory.mktemp("data") / "chunked.vortex"
    vx.io.write(pa.table({"index": pa.array(range(CHUNKED_ROWS), type=pa.int64())}), str(fname))
    vxf = vx.open(str(fname))
    assert len(vxf.splits()) > 1
    return vxf


@pytest.fixture(scope="session")
def flat_vxfile(tmpdir_factory) -> vx.VortexFile:
    """A small table, written as a single flat chunk."""
    fname = tmpdir_factory.mktemp("data") / "flat.vortex"
    vx.io.write(pa.table({"index": pa.array(range(100), type=pa.int64())}), str(fname))
    vxf = vx.open(str(fname))
    assert len(vxf.splits()) == 1
    return vxf


@pytest.mark.parametrize(
    "row_range",
    [(0, 10), (CHUNKED_ROWS - 5, CHUNKED_ROWS), (CHUNKED_ROWS, CHUNKED_ROWS), (50, 10), (0, 0)],
)
def test_execute_row_range_within_bounds_is_unchanged(
    chunked_vxfile: vx.VortexFile, row_range: tuple[int, int]
) -> None:
    start, stop = row_range
    table = chunked_vxfile.to_repeated_scan().execute(row_range=row_range).read_all().to_arrow_table()
    rows = table.column("index")
    assert rows.to_pylist() == list(range(CHUNKED_ROWS))[start:stop]


@pytest.mark.parametrize("layout", ["flat", "chunked"])
@pytest.mark.parametrize(
    ("start_from_end", "stop_from_end"),
    [(-5, 100), (5, 9), (0, 1)],
    ids=["stop_past_end", "entirely_past_end", "one_past_end"],
)
def test_execute_row_range_past_the_end_raises(
    request: pytest.FixtureRequest, layout: str, start_from_end: int, stop_from_end: int
) -> None:
    """A range past the end is an error on every layout; on a chunked file it used to panic."""
    file: vx.VortexFile = request.getfixturevalue(f"{layout}_vxfile")
    rows = len(file)
    with pytest.raises(RuntimeError, match="OutOfBounds"):
        file.to_repeated_scan().execute(row_range=(rows + start_from_end, rows + stop_from_end))


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
    assert actual.to_arrow_array(combine_chunks=True) == expected


def test_scanner_property_projected(vxfile: vx.VortexFile) -> None:
    assert vxfile.to_dataset().scanner(columns=["bool"]).projected_schema == pa.schema([("bool", pa.bool_())])


def test_scanner_property_dataset_schema(vxfile: vx.VortexFile) -> None:
    assert vxfile.to_dataset().scanner().dataset_schema == pa.schema(
        [("index", pa.int64()), ("string", pa.string_view()), ("bool", pa.bool_()), ("float", pa.float64())]
    )
