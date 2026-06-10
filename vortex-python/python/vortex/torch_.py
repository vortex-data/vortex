# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""PyTorch DataLoader integration for Vortex files.

:class:`VortexMapDataset` is a map-style dataset compatible with
``torch.utils.data.DataLoader``. It is duck-typed against the map-style dataset
protocol (``__len__``/``__getitem__``/``__getitems__``), so this module does not itself
depend on ``torch``.

Vortex files support cheap random access, so a :class:`VortexMapDataset` over a local
file or remote URL (including ``hf://`` Hugging Face Hub URLs) only reads the rows a
DataLoader requests, even with shuffling enabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pyarrow as pa

from .arrays import array as _array
from .file import VortexFile
from .file import open as _open
from .type_aliases import IntoProjection

if TYPE_CHECKING:
    from ._lib.scalar import ScalarPyType  # pyright: ignore[reportMissingModuleSource]


class VortexMapDataset:
    """A map-style dataset over a Vortex file, for use with ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    source : :class:`str` | :class:`vortex.VortexFile`
        A local path or URL (including ``hf://`` URLs) to a Vortex file, or an already
        opened :class:`vortex.VortexFile`.
    projection : :class:`vortex.Expr` | list[str] | None
        The columns to read, or else read all columns.

    Examples
    --------
    >>> import vortex as vx
    >>> from vortex.torch_ import VortexMapDataset
    >>> vx.io.write(vx.array([{"x": i, "y": i * i} for i in range(100)]), "points.vortex")
    >>> ds = VortexMapDataset("points.vortex")
    >>> len(ds)
    100
    >>> ds[3]
    {'x': 3, 'y': 9}
    >>> ds.__getitems__([7, 3, 7])
    [{'x': 7, 'y': 49}, {'x': 3, 'y': 9}, {'x': 7, 'y': 49}]
    """

    def __init__(self, source: str | VortexFile, *, projection: IntoProjection = None):
        self._file = source if isinstance(source, VortexFile) else _open(source)
        self._projection = projection
        self._scan = self._file.to_repeated_scan(projection)
        self._len = len(self._file)

    def __len__(self) -> int:
        return self._len

    def _normalize(self, index: int) -> int:
        if index < 0:
            index += self._len
        if not 0 <= index < self._len:
            raise IndexError(f"index {index} out of range for dataset of length {self._len}")
        return index

    def __getitem__(self, index: int) -> ScalarPyType:
        return self._scan.scalar_at(self._normalize(index)).as_py()

    def __getitems__(self, indices: Sequence[int]) -> list[ScalarPyType]:
        """Fetch a batch of rows in a single scan.

        The underlying file is read once with a sorted row-index pushdown, then rows are
        reordered to match ``indices``.
        """
        normalized = [self._normalize(i) for i in indices]
        unique = sorted(set(normalized))
        rows = (
            self._file.scan(self._projection, indices=_array(pa.array(unique, type=pa.uint64()))).read_all().to_pylist()
        )
        by_index = dict(zip(unique, rows, strict=True))
        return [by_index[i] for i in normalized]
