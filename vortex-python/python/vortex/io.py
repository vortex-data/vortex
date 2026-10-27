# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import Protocol, runtime_checkable

from typing_extensions import Buffer

from vortex._lib.io import VortexWriteOptions, read_url, write


@runtime_checkable
class ReadAt(Protocol):
    """
    A positional byte source that Vortex reads from, for storage only reachable from Python.

    Pass one to :func:`vortex.open_readable`. Binary file objects (``open(path, "rb")``,
    :class:`io.BytesIO`, fsspec and pyarrow files) are accepted too without implementing this
    protocol, but their reads are serialized because each one has to ``seek`` first. Implement
    :class:`ReadAt` when the underlying storage supports positional reads, such as :func:`os.pread`
    or HTTP range requests, so that Vortex can issue many reads at once: up to 192 by default, set
    with the ``concurrency`` argument of :func:`vortex.open_readable`.

    Reads are called from Vortex worker threads, potentially many concurrently, so implementations
    must be safe for concurrent use. Releasing the GIL while waiting on IO lets those reads overlap.

    Examples
    --------
    A reader over an open file descriptor:

    >>> import os
    >>> class PReadFile:
    ...     def __init__(self, path):
    ...         self._fd = os.open(path, os.O_RDONLY)
    ...
    ...     def size(self) -> int:
    ...         return os.fstat(self._fd).st_size
    ...
    ...     def read_into(self, offset: int, buffer: memoryview) -> int:
    ...         return os.preadv(self._fd, [buffer], offset)
    >>> vxf = vx.open_readable(PReadFile("data.vortex"))  # doctest: +SKIP
    """

    def size(self) -> int:
        """Total length of the source in bytes. Called once, when the file is opened."""
        ...

    def read_into(self, offset: int, buffer: memoryview) -> int:
        """
        Read bytes starting at absolute ``offset`` into the writable ``buffer``.

        Returns the number of bytes written. A short read is retried for the remainder, and
        returning ``0`` before ``buffer`` is full is an error. ``buffer`` is valid only for the
        duration of the call: do not keep it, or any view derived from it, after returning.
        """
        ...


@runtime_checkable
class ReadBytesAt(Protocol):
    """
    A positional byte source that returns its own buffers, for storage only reachable from Python.

    Pass one to :func:`vortex.open_readable`. Use it instead of :class:`ReadAt` when the storage
    already produces a buffer, such as ``bytes`` from :func:`os.pread` or the result of an
    ``obstore`` range request. Vortex keeps that buffer without copying it when it is read-only,
    C-contiguous, the full requested length, and aligned as Vortex needs. It copies any other
    buffer once. An object with both ``read_at`` and ``read_into`` is read through ``read_at``.

    Reads are called from Vortex worker threads, potentially many concurrently, so implementations
    must be safe for concurrent use. Up to 192 reads run at once by default, set with the
    ``concurrency`` argument of :func:`vortex.open_readable`.

    Examples
    --------
    A reader over an open file descriptor:

    >>> import os
    >>> class PReadBytes:
    ...     def __init__(self, path):
    ...         self._fd = os.open(path, os.O_RDONLY)
    ...
    ...     def size(self) -> int:
    ...         return os.fstat(self._fd).st_size
    ...
    ...     def read_at(self, offset: int, length: int) -> bytes:
    ...         return os.pread(self._fd, length, offset)
    >>> vxf = vx.open_readable(PReadBytes("data.vortex"))  # doctest: +SKIP
    """

    def size(self) -> int:
        """Total length of the source in bytes. Called once, when the file is opened."""
        ...

    def read_at(self, offset: int, length: int) -> Buffer:
        """
        Return up to ``length`` bytes starting at absolute ``offset``, as any buffer-protocol object.

        A shorter result is retried for the remainder, and an empty result before ``length`` bytes
        is an error. Vortex may keep the returned object for as long as it uses the bytes. Do not
        change its contents after returning it, even through a read-only view.
        """
        ...


__all__ = ["read_url", "write", "ReadAt", "ReadBytesAt", "VortexWriteOptions"]
