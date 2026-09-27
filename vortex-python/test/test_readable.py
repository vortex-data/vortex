# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import io
import os
import pickle
import threading
from pathlib import Path

import pyarrow as pa
import pytest
from vortex.io import ReadAt

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


def read_all(source: object) -> pa.Table:
    return vx.open(source, without_segment_cache=True).to_arrow().read_all()  # ty: ignore[invalid-argument-type]


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


def test_pathlike(path: Path, expected: pa.Table) -> None:
    assert read_all(path).equals(expected)


def test_read_only_file_object(path: Path, expected: pa.Table) -> None:
    class ReadOnly:
        def __init__(self, data: bytes) -> None:
            self._inner = io.BytesIO(data)

        def seek(self, offset: int, whence: int = 0) -> int:
            return self._inner.seek(offset, whence)

        def read(self, n: int) -> bytes:
            # Return at most 1000 bytes to exercise the short-read loop.
            return self._inner.read(min(n, 1000))

    assert read_all(ReadOnly(path.read_bytes())).equals(expected)


def test_read_at(path: Path, expected: pa.Table) -> None:
    reader = PReadFile(path)
    try:
        assert isinstance(reader, ReadAt)
        vxf = vx.open(reader, without_segment_cache=True)
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
        vx.open(object())  # ty: ignore[invalid-argument-type]


def test_store_with_readable_is_an_error(path: Path) -> None:
    from vortex.store import LocalStore

    with open(path, "rb") as f, pytest.raises(TypeError, match="store"):
        vx.open(f, store=LocalStore())


def test_pickle_is_refused(path: Path) -> None:
    with open(path, "rb") as f:
        vxf = vx.open(f)
        assert vxf.path == str(path)
        with pytest.raises(TypeError, match="Python readable"):
            pickle.dumps(vxf)
