# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
import pickle

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
def vxf(tmpdir_factory) -> vx.VortexFile:
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"

    if not os.path.exists(fname):
        a = pa.array([record(x) for x in range(1_000_000)])
        arr = vx.compress(vx.array(a))
        vx.io.write(arr, str(fname))
    return vx.open(str(fname), without_segment_cache=True)


def test_dtype(vxf: VortexFile) -> None:
    assert vxf.dtype.to_arrow_schema() == pa.schema(
        [("index", pa.int64()), ("string", pa.string_view()), ("bool", pa.bool_()), ("float", pa.float64())]
    )


def test_row_count(vxf: VortexFile) -> None:
    assert len(vxf) == 1_000_000


def test_scan(vxf: VortexFile) -> None:
    start = 0
    for batch in vxf.scan():
        assert batch.to_pylist() == [record(x) for x in range(start, start + len(batch))]
        start += len(batch)
    assert start == 1_000_000


def test_scan_with_indices(vxf: VortexFile) -> None:
    rows = [row for batch in vxf.scan(indices=vx.array([1, 10, 1_000, 999_999])) for row in batch.to_pylist()]
    assert rows == [record(x) for x in [1, 10, 1_000, 999_999]]


def test_read_url_with_indices(tmp_path) -> None:
    fname = tmp_path / "indices.vortex"
    vx.io.write(vx.compress(vx.array(pa.array([record(x) for x in range(100)]))), str(fname))
    result = vx.io.read_url(str(fname), indices=vx.array([1, 10, 42]))
    assert result.to_pylist() == [record(x) for x in [1, 10, 42]]


def test_to_arrow_batch_size(vxf: VortexFile) -> None:
    assert len(list(vxf.to_arrow(batch_size=1_000_000))) == 1, "batch_size=1_000_000"
    assert len(list(vxf.to_arrow(batch_size=1_000))) == 1_000, "batch_size=1_000"


def test_to_arrow_columns(vxf: VortexFile) -> None:
    rbr = vxf.to_arrow(projection=["string", "bool"])
    assert rbr.schema == pa.schema([("string", pa.string_view()), ("bool", pa.bool_())])


def test_to_arrow_offset_string_schema(vxf: VortexFile) -> None:
    schema = pa.schema([("string", pa.string())])
    table = vxf.to_arrow(projection=["string"], schema=schema).read_all()
    assert table.schema == schema
    assert table.column("string").combine_chunks() == pa.array(
        (str(i) for i in range(1_000_000)),
        type=pa.string(),
    )


def test_empty_file(tmpdir_factory) -> None:
    # test for writing empty files with null columns
    # create an empty table with schema `empty: null`
    table = pa.Table.from_pydict({"empty": []})
    assert repr(table.schema) == "empty: null"

    # cast to Vortex array
    empty = vx.array(table)
    assert len(empty) == 0
    assert repr(empty.dtype) == 'struct({"empty": null()}, nullable=False)'

    # writing file should succeed
    empty_file = tmpdir_factory.mktemp("data") / "empty.vortex"
    vx.io.write(empty, str(empty_file))
    assert vx.open(str(empty_file)).to_arrow().read_all().equals(table)


def test_stream_pyarrow(tmpdir_factory) -> None:
    import pyarrow.parquet as pq

    data_dir = tmpdir_factory.mktemp("data")
    table = pa.Table.from_pydict(
        {
            "names": ["Alice", "Bob", "Carol"],
            "ages": [21, 22, 23],
        }
    )
    pq.write_table(table, str(data_dir / "names.parquet"))

    df = pq.read_table(str(data_dir / "names.parquet"))
    vx.io.write(df, str(data_dir / "names.vortex"))
    assert vx.open(str(data_dir / "names.vortex")).to_arrow(schema=table.schema).read_all().equals(table)


@pytest.mark.parametrize("with_footer", [False, True])
def test_memory_map_reads_values(vxf: VortexFile, with_footer: bool) -> None:
    mapped = vx.open(vxf.path, memory_map=True, footer=vxf.footer if with_footer else None)
    rows = [0, 7, 10, 500_000, 999_999]
    assert mapped.scan(indices=vx.array(rows)).read_all().to_pylist() == [record(x) for x in rows]
    assert len(mapped) == 1_000_000


def test_memory_map_pickle_roundtrip_reads_values(vxf: VortexFile) -> None:
    mapped = pickle.loads(pickle.dumps(vx.open(vxf.path, memory_map=True)))
    assert len(mapped) == 1_000_000
    assert mapped.scan(indices=vx.array([7, 999_999])).read_all().to_pylist() == [record(7), record(999_999)]


def test_memory_map_rejects_segment_cache(vxf: VortexFile) -> None:
    with pytest.raises(ValueError, match="segment_cache"):
        vx.open(vxf.path, memory_map=True, segment_cache=vx.SegmentCache(1 << 20), cache_key="k")


def test_memory_map_rejects_remote_url() -> None:
    with pytest.raises(ValueError, match="local path"):
        vx.open("s3://bucket/key.vortex", memory_map=True)
