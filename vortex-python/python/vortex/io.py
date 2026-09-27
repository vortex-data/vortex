# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import Protocol, runtime_checkable

from vortex._lib.io import VortexWriteOptions, read_url, write


@runtime_checkable
class ReadAt(Protocol):
    """
    A positional byte source that Vortex reads from, for storage only reachable from Python.

    Pass one to :func:`vortex.open`. Binary file objects (``open(path, "rb")``, :class:`io.BytesIO`,
    fsspec and pyarrow files) are accepted too without implementing this protocol, but their reads
    are serialized because each one has to ``seek`` first. Implement :class:`ReadAt` when the
    underlying storage supports positional reads, such as :func:`os.pread` or HTTP range requests,
    so that Vortex can issue several reads at once.

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
    >>> vxf = vx.open(PReadFile("data.vortex"))  # doctest: +SKIP
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


__all__ = ["read_url", "write", "ReadAt", "VortexWriteOptions"]
