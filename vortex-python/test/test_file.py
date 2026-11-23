# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
import pickle
import subprocess
import sys
import textwrap

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


def test_zero_batch_size_rejected(vxf: VortexFile) -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        vxf.scan(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        vxf.to_repeated_scan(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        vxf.to_arrow(batch_size=0)


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


@pytest.mark.parametrize("as_reader", [False, True])
def test_write_reports_unsupported_arrow_type(tmp_path, as_reader: bool) -> None:
    table = pa.table({"elapsed": pa.array([1], pa.duration("ms"))})
    source = pa.RecordBatchReader.from_batches(table.schema, table.to_batches()) if as_reader else table

    with pytest.raises(ValueError, match="Duration"):
        vx.io.write(source, str(tmp_path / "x.vortex"))


def test_write_rejects_unsupported_object(tmp_path) -> None:
    with pytest.raises(TypeError, match="converted to an ArrayIterator"):
        vx.io.write(object(), str(tmp_path / "x.vortex"))  # ty: ignore[invalid-argument-type]


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


@pytest.mark.parametrize("mode", ["to_arrow", "write"])
def test_array_iterator_concurrent_transfer_does_not_deadlock(mode: str) -> None:
    script = textwrap.dedent(
        """
        import os
        import sys
        import tempfile
        import threading

        import pyarrow as pa
        import vortex as vx

        entry = threading.Event()
        release = threading.Event()
        transfer_started = threading.Event()
        schema = pa.schema([pa.field("value", pa.int64())])
        first_batch = vx.array(pa.table({"value": [10]}, schema=schema))
        second_batch = vx.array(pa.table({"value": [20]}, schema=schema))

        def delayed():
            entry.set()
            assert release.wait(5)
            yield first_batch
            yield second_batch

        it = vx.ArrayIterator.from_iter(first_batch.dtype, iter(delayed()))
        seen = []

        def advance():
            seen.append(next(it).to_arrow_table().to_pydict())

        def releaser():
            assert transfer_started.wait(5)
            release.set()

        producer = threading.Thread(target=advance)
        watcher = threading.Thread(target=releaser)
        producer.start()
        watcher.start()
        assert entry.wait(5)

        # Hold the GIL until the native transfer call detaches it while waiting on
        # the iterator mutex. Without detach(), watcher never gets the GIL to set
        # `release`, so a lock-order regression deterministically deadlocks here.
        sys.setswitchinterval(10.0)
        mode = sys.argv[1]
        if mode == "to_arrow":
            transfer_started.set()
            remaining = it.to_arrow().read_all().to_pydict()
        else:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "out.vortex")
                transfer_started.set()
                vx.io.write(it, path)
                remaining = vx.open(path).scan().read_all().to_arrow_table().to_pydict()

        watcher.join(5)
        producer.join(5)
        assert not watcher.is_alive()
        assert not producer.is_alive()
        assert seen == [{"value": [10]}]
        assert remaining == {"value": [20]}
        assert next(it, None) is None
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, mode],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


@pytest.mark.parametrize("method", ["next", "read_all", "to_arrow_array", "to_arrow", "write"])
def test_python_array_iterator_rejects_reentrant_access(tmp_path, method: str) -> None:
    batch = vx.array(pa.table({"a": [1, 2]}))
    holder: dict[str, vx.ArrayIterator] = {}

    def gen():
        it = holder["it"]
        with pytest.raises(RuntimeError, match="already being advanced"):
            if method == "next":
                next(it)
            elif method == "read_all":
                it.read_all()
            elif method == "to_arrow_array":
                it.to_arrow_array()
            elif method == "to_arrow":
                it.to_arrow()
            else:
                vx.io.write(it, str(tmp_path / "reentrant.vortex"))
        yield batch

    it = vx.ArrayIterator.from_iter(batch.dtype, gen())
    holder["it"] = it

    assert next(it).to_pylist() == [{"a": 1}, {"a": 2}]
    with pytest.raises(StopIteration):
        next(it)
