# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from typing import final

from ._lib import scan as _scan  # pyright: ignore[reportMissingModuleSource]
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]
from ._lib.scalar import Scalar  # pyright: ignore[reportMissingModuleSource]


@final
class RepeatedScan:
    """
    A prepared scan that is optimized fro repeated execution.
    """

    def __init__(self, scan: _scan.RepeatedScan):
        self._scan = scan

    def __getitem__(self, item):
        """Get rows from the scan. Returns a :class:`vortex.ArrayIterator`
        if item is a slice or a :class:`vortex.Scalar` if item is an int.

        Parameters
        ----------
        item : slice | int
            If slice, must not have a step.
        """
        if isinstance(item, slice):
            if item.step is not None:
                raise ValueError("slice step is not supported")
            start = item.start
            stop = item.stop
            return self.execute(row_range=(start, stop))
        elif isinstance(item, int):
            return self.scalar_at(item)
        else:
            raise TypeError(f"unexpected type for item: {type(item)}")

    def execute(
        self,
        *,
        row_range: tuple[int, int] | None = None,
    ) -> ArrayIterator:
        """Execute the scan returning a :class:`vortex.ArrayIterator`.

        Parameters
        ----------
        row_range : tuple[int, int] | slice | None
            If tuple, it is interpreted as [start, stop).
            If slice, must not have a step.

        Examples
        --------

        Scan a file with a structured column and nulls at multiple levels and in multiple columns.

        >>> import vortex as vx
        >>> import vortex.expr as ve
        >>> a = vx.array([
        ...     {'name': 'Joseph', 'age': 25},
        ...     {'name': None, 'age': 31},
        ...     {'name': 'Angela', 'age': None},
        ...     {'name': 'Mikhail', 'age': 57},
        ...     {'name': None, 'age': None},
        ... ])
        >>> vx.io.write(a, "a.vortex")
        >>> scan = vx.open("a.vortex").to_repeated_scan()
        >>> scan.execute(row_range=(1, 3)).read_all().to_arrow_array()
        <pyarrow.lib.StructArray object at ...>
        -- is_valid: all not null
        -- child 0 type: int64
          [
            31,
            null,
          ]
        -- child 1 type: string_view
          [
            null,
            "Angela",
          ]
        """
        if row_range is None:
            start = None
            stop = None
        elif isinstance(row_range, tuple):
            start, stop = row_range
        else:
            raise TypeError(f"unexpected type for rows: {type(row_range)}")

        return self._scan.execute(start=start, stop=stop)

    def scalar_at(self, index: int) -> Scalar:
        """Fetch a scalar from the scan returning a :class:`vortex.Scalar`.

        Parameters
        ----------
        index : int
            The row index to fetch. Raises an :class:`IndexError` if out of bounds or
            if the given row index was not included in the scan.

        Examples
        --------

        Scan a file with a structured column and nulls at multiple levels and in multiple columns.

        >>> import vortex as vx
        >>> import vortex.expr as ve
        >>> a = vx.array([
        ...     {'name': 'Joseph', 'age': 25},
        ...     {'name': None, 'age': 31},
        ...     {'name': 'Angela', 'age': None},
        ...     {'name': 'Mikhail', 'age': 57},
        ...     {'name': None, 'age': None},
        ... ])
        >>> vx.io.write(a, "a.vortex")
        >>> scan = vx.open("a.vortex").to_repeated_scan()
        >>> scan.scalar_at(1)
        <vortex.Scalar object at ...>
        """
        return self._scan.scalar_at(index)
