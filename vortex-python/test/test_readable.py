# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import gc
import io
import os
import pickle
import threading
import time
import weakref
from pathlib import Path
from typing import IO, cast

import numpy as np
import pyarrow as pa
import pytest
from typing_extensions import Buffer
from vortex.io import ReadAt, ReadBytesAt

import vortex as vx


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return pa.table({"index": pa.array(range(100_000), pa.int64()), "string": [str(i) for i in range(100_000)]})


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory, table: pa.Table) -> Path:
    path = tmp_path_factory.mktemp("data") / "readable.vortex"
    vx.io.write(table, str(path))
    return path


@pytest.fixture(scope="module")
def expected(path: Path) -> pa.Table:
    """The file as read through the native path reader, which every Python reader must match."""
    return vx.open(str(path)).to_arrow().read_all()


def read_all(source: ReadBytesAt | ReadAt | IO[bytes]) -> pa.Table:
    return vx.open_readable(source, without_segment_cache=True).to_arrow().read_all()


class PReadFile:
    """A `ReadAt` over a file descriptor, recording the most reads it saw in flight at once."""

    def __init__(self, path: Path, chunk: int | None = None) -> None:
        self._fd = os.open(path, os.O_RDONLY)
        self._chunk = chunk
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.reads = 0

    def size(self) -> int:
        return os.fstat(self._fd).st_size

    def read_into(self, offset: int, buffer: memoryview) -> int:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.reads += 1
        try:
            if self._chunk is not None:
                buffer = buffer[: self._chunk]
            return os.preadv(self._fd, [buffer], offset)
        finally:
            with self._lock:
                self._in_flight -= 1

    def close(self) -> None:
        os.close(self._fd)


def test_file_object(path: Path, expected: pa.Table) -> None:
    with open(path, "rb") as f:
        assert read_all(f).equals(expected)


def test_bytes_io(path: Path, expected: pa.Table) -> None:
    assert read_all(io.BytesIO(path.read_bytes())).equals(expected)


def test_reopen_with_footer(path: Path, expected: pa.Table) -> None:
    reader = PReadFile(path)
    try:
        footer = vx.open_readable(reader).footer
        assert footer.row_count == len(expected)

        reads = reader.reads
        vxf = vx.open_readable(reader, footer=footer, without_segment_cache=True)
        assert reader.reads == reads
        assert len(vxf) == len(expected)
        assert vxf.to_arrow().read_all().equals(expected)
    finally:
        reader.close()


def test_footer_from_larger_file_is_rejected(path: Path, tmp_path: Path) -> None:
    footer = vx.open(str(path)).footer
    small = tmp_path / "small.vortex"
    vx.io.write(pa.table({"index": pa.array([0], pa.int64())}), str(small))
    with open(small, "rb") as f, pytest.raises(Exception):
        vx.open_readable(f, footer=footer)


def test_footer_is_not_picklable(path: Path) -> None:
    with pytest.raises(TypeError):
        pickle.dumps(vx.open(str(path)).footer)


def test_read_only_file_object(path: Path, expected: pa.Table) -> None:
    class ReadOnly:
        def __init__(self, data: bytes) -> None:
            self._inner = io.BytesIO(data)

        def seek(self, offset: int, whence: int = 0) -> int:
            return self._inner.seek(offset, whence)

        def read(self, n: int) -> bytes:
            # Return at most 1000 bytes to exercise the short-read loop.
            return self._inner.read(min(n, 1000))

    # A deliberately minimal file object: only `seek` and `read`, which is not a full `IO[bytes]`.
    assert read_all(ReadOnly(path.read_bytes())).equals(expected)  # ty: ignore[invalid-argument-type]


def test_read_at(path: Path, expected: pa.Table) -> None:
    reader = PReadFile(path)
    try:
        assert isinstance(reader, ReadAt)
        vxf = vx.open_readable(reader, without_segment_cache=True)
        assert len(vxf) == expected.num_rows
        filtered = vxf.to_arrow(["index"], expr=vx.expr.column("index") < 10).read_all()
        assert filtered == pa.table({"index": pa.array(range(10), pa.int64())})
        assert reader.reads > 0
    finally:
        reader.close()


def test_read_at_short_reads(path: Path, expected: pa.Table) -> None:
    reader = PReadFile(path, chunk=777)
    try:
        assert read_all(reader).equals(expected)
    finally:
        reader.close()


def test_file_object_reads_are_serialized(path: Path) -> None:
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    class Tracking(io.FileIO):
        def readinto(self, buffer: memoryview) -> int | None:  # ty: ignore[invalid-method-override]
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                n = super().readinto(buffer)
                assert n is None or isinstance(n, int)
                return n
            finally:
                with lock:
                    in_flight -= 1

    with Tracking(path, "r") as f:
        read_all(f)
    assert max_in_flight == 1


def test_eof_is_an_error(path: Path) -> None:
    class Truncated(PReadFile):
        def size(self) -> int:
            return super().size() + 100

    reader = Truncated(path)
    try:
        with pytest.raises(Exception, match="0 bytes"):
            read_all(reader)
    finally:
        reader.close()


def test_reader_exception_propagates(path: Path) -> None:
    class Failing(PReadFile):
        def read_into(self, offset: int, buffer: memoryview) -> int:
            raise OSError("storage unavailable")

    reader = Failing(path)
    try:
        with pytest.raises(Exception, match="storage unavailable"):
            read_all(reader)
    finally:
        reader.close()


def test_retained_buffer_is_an_error(path: Path) -> None:
    kept: list[memoryview] = []

    class Retaining(PReadFile):
        def read_into(self, offset: int, buffer: memoryview) -> int:
            n = super().read_into(offset, buffer)
            kept.append(buffer[:1])
            return n

    reader = Retaining(path)
    try:
        with pytest.raises(Exception, match="retained a view"):
            read_all(reader)
        # The retained view stays valid: the allocation it points at was left to Python.
        assert all(len(bytes(view)) == 1 for view in kept)
    finally:
        reader.close()


def test_not_readable() -> None:
    with pytest.raises(TypeError, match="binary file object"):
        vx.open_readable(object())  # ty: ignore[invalid-argument-type]


def test_pickle_is_refused(path: Path) -> None:
    with open(path, "rb") as f:
        vxf = vx.open_readable(f)
        assert vxf.path == str(path)
        with pytest.raises(TypeError, match="Python readable"):
            pickle.dumps(vxf)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_read_at_concurrency_limit(path: Path, concurrency: int) -> None:
    class Slow(PReadFile):
        def read_into(self, offset: int, buffer: memoryview) -> int:
            time.sleep(0.01)
            return super().read_into(offset, buffer)

    reader = Slow(path)
    try:
        vxf = vx.open_readable(reader, without_segment_cache=True, concurrency=concurrency)
        vxf.to_arrow().read_all()
        assert 0 < reader.max_in_flight <= concurrency
    finally:
        reader.close()


def test_concurrency_rejected_for_file_object(path: Path) -> None:
    with open(path, "rb") as f, pytest.raises(TypeError, match="serialized"):
        vx.open_readable(f, concurrency=4)


class PReadBytes:
    """A `ReadBytesAt` that returns read-only NumPy arrays, so tests can see which ones Vortex keeps."""

    def __init__(self, path: Path, *, chunk: int | None = None, misalign: bool = False, writable: bool = False) -> None:
        self._data = path.read_bytes()
        self._chunk = chunk
        self._misalign = misalign
        self._writable = writable
        self.returned: list[weakref.ref[np.ndarray]] = []

    def size(self) -> int:
        return len(self._data)

    def read_at(self, offset: int, length: int) -> Buffer:
        if self._chunk is not None:
            length = min(length, self._chunk)
        # Allocate 64-byte aligned memory, then start one byte in to misalign it on request.
        start = 1 if self._misalign else 0
        raw = np.empty(length + 64 + start, dtype=np.uint8)
        base = (-raw.ctypes.data) % 64 + start
        out = raw[base : base + length]
        out[:] = np.frombuffer(self._data, dtype=np.uint8, count=length, offset=offset)
        out.setflags(write=self._writable)
        self.returned.append(weakref.ref(raw))
        # NumPy's stubs do not declare `__buffer__`, though `ndarray` implements the buffer protocol.
        return cast(Buffer, out)

    def alive(self) -> int:
        gc.collect()
        return sum(ref() is not None for ref in self.returned)


def test_read_bytes_at(path: Path, expected: pa.Table) -> None:
    reader = PReadBytes(path)
    assert isinstance(reader, ReadBytesAt)
    assert read_all(reader).equals(expected)


def test_read_bytes_at_keeps_suitable_buffers(path: Path) -> None:
    reader = PReadBytes(path)
    vxf = vx.open_readable(reader, without_segment_cache=True)
    array = vxf.scan().read_all()

    # The scanned array uses the returned buffers in place, and frees them along with itself.
    assert reader.alive() > 0
    del vxf, array
    assert reader.alive() == 0


def test_read_bytes_at_copies_writable_buffers(path: Path) -> None:
    reader = PReadBytes(path, writable=True)
    vxf = vx.open_readable(reader, without_segment_cache=True)
    array = vxf.scan().read_all()
    assert reader.alive() == 0
    del vxf, array


def test_read_bytes_at_misaligned_buffers(path: Path, expected: pa.Table) -> None:
    # Coalesced reads may need no alignment, so Vortex can still keep these; it realigns any
    # slice that needs it.
    assert read_all(PReadBytes(path, misalign=True)).equals(expected)


def test_read_bytes_at_short_reads(path: Path, expected: pa.Table) -> None:
    reader = PReadBytes(path, chunk=777)
    assert read_all(reader).equals(expected)
    assert reader.alive() == 0


def test_read_bytes_at_is_preferred_over_read_into(path: Path, expected: pa.Table) -> None:
    class Both(PReadBytes):
        def read_into(self, offset: int, buffer: memoryview) -> int:
            raise AssertionError("read_into must not be called")

    assert read_all(Both(path)).equals(expected)


def test_read_bytes_at_eof_is_an_error(path: Path) -> None:
    class Truncated(PReadBytes):
        def size(self) -> int:
            return super().size() + 100

        def read_at(self, offset: int, length: int) -> bytes:
            return self._data[offset : offset + length]

    with pytest.raises(Exception, match="0 bytes"):
        read_all(Truncated(path))


def test_read_bytes_at_overlong_result_is_an_error(path: Path) -> None:
    class Overlong(PReadBytes):
        def read_at(self, offset: int, length: int) -> bytes:
            return bytes(length + 1)

    with pytest.raises(Exception, match="returned"):
        read_all(Overlong(path))


def test_read_bytes_at_returns_bytes(path: Path, expected: pa.Table) -> None:
    class Pread(PReadBytes):
        def read_at(self, offset: int, length: int) -> bytes:
            return self._data[offset : offset + length]

    assert read_all(Pread(path)).equals(expected)
