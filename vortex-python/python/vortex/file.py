# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, final

import pyarrow as pa

from ._lib import file as _file  # pyright: ignore[reportMissingModuleSource]
from ._lib.arrays import Array  # pyright: ignore[reportMissingModuleSource]
from ._lib.dtype import DType  # pyright: ignore[reportMissingModuleSource]
from ._lib.expr import Expr  # pyright: ignore[reportMissingModuleSource]
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]
from .dataset import VortexDataset
from .type_aliases import IntoProjection, RecordBatchReader

if TYPE_CHECKING:
    import polars


def open(path: str) -> VortexFile:
    """
    Lazily open a Vortex file located at the given path or URL.

    Parameters
    ----------
    path : :class:`str`
        A local path or URL to the Vortex file.

    Examples
    --------
    Open a Vortex file and perform a scan operation:

    >>> import vortex as vx
    >>> vxf = vx.open("data.vortex") # doctest: +SKIP
    >>> array_iterator = vxf.scan() # doctest: +SKIP

    See also: :class:`vortex.dataset.VortexDataset`
    """

    return VortexFile(_file.open(path))


@final
class VortexFile:
    def __init__(self, file: _file.VortexFile):
        self._file = file

    def __len__(self) -> int:
        return self._file.__len__()

    @property
    def dtype(self) -> DType:
        """The dtype of the file."""
        return self._file.dtype

    def scan(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
    ) -> ArrayIterator:
        """Scan the Vortex file returning a :class:`vortex.ArrayIterator`.

        Parameters
        ----------
        projection : :class:`vortex.Expr` | list[str] | None
            The projection expression to read, or else read all columns.
        expr : :class:`vortex.Expr` | None
            The predicate used to filter rows. The filter columns do not need to be in the projection.
        indices : :class:`vortex.Array` | None
            The indices of the rows to read. Must be sorted and non-null.
        batch_size : :class:`int` | None
            The number of rows to read per chunk.

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
        >>> vxf = vx.open("a.vortex")
        >>> vxf.scan().read_all().to_arrow_array()
        <pyarrow.lib.StructArray object at ...>
        -- is_valid: all not null
        -- child 0 type: int64
          [
            25,
            31,
            null,
            57,
            null
          ]
        -- child 1 type: string_view
          [
            "Joseph",
            null,
            "Angela",
            "Mikhail",
            null
          ]

        Read just the age column:

        >>> vxf.scan(['age']).read_all().to_arrow_array()
        <pyarrow.lib.StructArray object at ...>
        -- is_valid: all not null
        -- child 0 type: int64
          [
            25,
            31,
            null,
            57,
            null
          ]


        Keep rows with an age above 35. This will read O(N_KEPT) rows, when the file format allows.

        >>> vxf.scan(expr=ve.column("age") > 35).read_all().to_arrow_array()
        <pyarrow.lib.StructArray object at ...>
        -- is_valid: all not null
        -- child 0 type: int64
          [
            57
          ]
        -- child 1 type: string_view
          [
            "Mikhail"
          ]
        """
        return self._file.scan(projection, expr=expr, indices=indices, batch_size=batch_size)

    def to_arrow(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        batch_size: int | None = None,
    ) -> RecordBatchReader:
        """Scan the Vortex file as a :class:`pyarrow.RecordBatchReader`.

        Parameters
        ----------
        projection : :class:`vortex.Expr` | list[str] | None
            Either an expression over the columns of the file (only referenced columns will be read
            from the file) or an explicit list of desired columns.
        expr : :class:`vortex.Expr` | None
            The predicate used to filter rows. The filter columns need not appear in the projection.
        batch_size : :class:`int` | None
            The number of rows to read per chunk.

        """
        return self._file.to_arrow(projection, expr=expr, batch_size=batch_size)

    def to_dataset(self) -> VortexDataset:
        """Scan the Vortex file using the :class:`pyarrow.dataset.Dataset` API."""
        return VortexDataset(self._file.to_dataset())

    def to_polars(self) -> polars.LazyFrame:
        """Read the Vortex file as a pl.LazyFrame, supporting column pruning and predicate pushdown."""
        import polars as pl
        from polars.io.plugins import register_io_source

        from vortex.polars_ import polars_to_vortex

        schema = self.dtype.to_arrow_schema()

        def _io_source(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            _n_rows: int | None,
            _batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            vx_predicate: Expr | None = None if predicate is None else polars_to_vortex(predicate)

            reader = self.to_arrow(projection=with_columns, expr=vx_predicate)

            for batch in reader:
                batch = pl.DataFrame._from_arrow(batch, rechunk=False)  # pyright: ignore[reportPrivateUsage]
                # TODO(ngates): set sortedness on DataFrame based on stats?
                yield batch

            # Make sure we always yield at least one empty DataFrame
            yield pl.DataFrame._from_arrow(  # pyright: ignore[reportPrivateUsage]
                data=pa.RecordBatch.from_arrays(  # pyright: ignore[reportUnknownMemberType]
                    [pa.array([], type=field.type) for field in reader.schema],  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
                    schema=reader.schema,
                ),
            )

        # https://github.com/pola-rs/polars/pull/24125
        return register_io_source(_io_source, schema=schema)  # pyright: ignore[reportArgumentType]
