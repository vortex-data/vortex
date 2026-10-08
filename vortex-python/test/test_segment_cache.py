# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from pathlib import Path

import pyarrow as pa
import pytest

import vortex as vx


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "cached.vortex"
    rows = pa.table({"index": pa.array(range(50_000), pa.int64()), "string": [str(i) for i in range(50_000)]})
    vx.io.write(rows, str(path))
    return path


@pytest.fixture(scope="module")
def table(path: Path) -> pa.Table:
    """The file as read without a shared cache, which every cached read must match; strings read back as string_view."""
    return vx.open(str(path), without_segment_cache=True).to_arrow().read_all()


class CountingReader:
    """A `ReadBytesAt` over an in-memory copy of a file, counting the bytes it returns."""

    def __init__(self, path: Path) -> None:
        self._data = path.read_bytes()
        self.bytes_read = 0

    def size(self) -> int:
        return len(self._data)

    def read_at(self, offset: int, length: int) -> bytes:
        self.bytes_read += length
        return self._data[offset : offset + length]


def read_through(reader: CountingReader, cache: vx.SegmentCache, key: str, footer: vx.file.Footer) -> pa.Table:
    vxf = vx.open_readable(reader, footer=footer, segment_cache=cache, cache_key=key)
    return vxf.to_arrow().read_all()


def test_reopen_reuses_cached_segments(path: Path, table: pa.Table) -> None:
    cache = vx.SegmentCache(64 << 20)
    footer = vx.open(str(path)).footer

    first = CountingReader(path)
    assert read_through(first, cache, "file", footer).equals(table)
    assert first.bytes_read > 0
    assert cache.entry_count > 0
    assert 0 < cache.size_bytes <= 64 << 20

    # A new reader for the same file and key reads nothing: the footer and every segment are cached.
    again = CountingReader(path)
    assert read_through(again, cache, "file", footer).equals(table)
    assert again.bytes_read == 0


def test_keys_separate_files(path: Path, table: pa.Table) -> None:
    cache = vx.SegmentCache(64 << 20)
    footer = vx.open(str(path)).footer
    read_through(CountingReader(path), cache, "a", footer)

    other = CountingReader(path)
    assert read_through(other, cache, "b", footer).equals(table)
    assert other.bytes_read > 0


def test_clear(path: Path) -> None:
    cache = vx.SegmentCache(64 << 20)
    footer = vx.open(str(path)).footer
    read_through(CountingReader(path), cache, "file", footer)

    cache.clear()
    assert cache.entry_count == 0
    after = CountingReader(path)
    read_through(after, cache, "file", footer)
    assert after.bytes_read > 0


def test_capacity_is_respected(path: Path, table: pa.Table) -> None:
    cache = vx.SegmentCache(4096)
    footer = vx.open(str(path)).footer
    assert read_through(CountingReader(path), cache, "file", footer).equals(table)
    assert cache.size_bytes <= 4096


def test_native_open_shares_the_cache(path: Path, table: pa.Table) -> None:
    cache = vx.SegmentCache(64 << 20)
    first = vx.open(str(path), segment_cache=cache, cache_key="file")
    assert first.to_arrow().read_all().equals(table)

    # The Python reader sees segments cached by the native open of the same key.
    reader = CountingReader(path)
    assert read_through(reader, cache, "file", first.footer).equals(table)
    assert reader.bytes_read == 0


def test_native_open_with_footer(path: Path, table: pa.Table) -> None:
    footer = vx.open(str(path)).footer
    assert vx.open(str(path), footer=footer).to_arrow().read_all().equals(table)


def test_cache_key_and_cache_go_together(path: Path) -> None:
    cache = vx.SegmentCache(1 << 20)
    with pytest.raises(ValueError, match="cache_key"):
        vx.open(str(path), segment_cache=cache)
    with pytest.raises(ValueError, match="requires a segment_cache"):
        vx.open(str(path), cache_key="file")
    with pytest.raises(ValueError, match="without_segment_cache"):
        vx.open(str(path), segment_cache=cache, cache_key="file", without_segment_cache=True)


def test_segments_from_the_footer_read_are_shared(tmp_path: Path) -> None:
    # The footer read of a small file covers all of it, so the first open reads every segment.
    small = tmp_path / "small.vortex"
    vx.io.write(pa.table({"index": pa.array(range(1000), pa.int64())}), str(small))
    cache = vx.SegmentCache(64 << 20)

    first = CountingReader(small)
    vxf = vx.open_readable(first, segment_cache=cache, cache_key="small")
    vxf.scan(indices=vx.array([3, 700])).read_all()
    assert first.bytes_read == small.stat().st_size

    # A reopen given the footer skips that read, but finds its segments in the shared cache.
    again = CountingReader(small)
    reopened = vx.open_readable(again, footer=vxf.footer, segment_cache=cache, cache_key="small")
    reopened.scan(indices=vx.array([3, 700])).read_all()
    assert again.bytes_read == 0


def test_footer_read_segments_survive_a_full_cache(tmp_path: Path) -> None:
    # The footer read of a small file covers all of it. A cache too small to hold its segments must
    # not make the open read them again.
    small = tmp_path / "small.vortex"
    vx.io.write(pa.table({"index": pa.array(range(1000), pa.int64())}), str(small))
    cache = vx.SegmentCache(1)

    reader = CountingReader(small)
    vxf = vx.open_readable(reader, segment_cache=cache, cache_key="small")
    vxf.scan(indices=vx.array([3, 700])).read_all()
    assert reader.bytes_read == small.stat().st_size
