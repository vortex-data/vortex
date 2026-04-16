# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ._lib import file as _file  # pyright: ignore[reportMissingModuleSource]
from ._lib.arrays import Array  # pyright: ignore[reportMissingModuleSource]
from ._lib.dtype import DType  # pyright: ignore[reportMissingModuleSource]
from ._lib.expr import Expr  # pyright: ignore[reportMissingModuleSource]
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]
from .arrow.expression import ensure_vortex_expression
from .dataset import VortexDataset
from .scan import RepeatedScan
from .store import (
    AzureStore,
    GCSStore,
    HTTPStore,
    LocalStore,
    MemoryStore,
    S3Store,
)
from .type_aliases import IntoProjection, RecordBatchReader

if TYPE_CHECKING:
    import polars


def open(
    path: str,
    *,
    store: AzureStore | GCSStore | HTTPStore | LocalStore | MemoryStore | S3Store | None = None,
    without_segment_cache: bool = False,
) -> VortexFile:
    """
    Lazily open a Vortex file located at the given path or URL.

    Parameters
    ----------
    path : :class:`str`
        A local path or URL to the Vortex file.
    store :
        An object store created from the `vortex.store` package. By default
        the store is inferred based on the path
    without_segment_cache : :class:`bool`
        If true, disable the segment cache for this file, useful when memory is constrained.

    Examples
    --------
    Open a Vortex file and perform a scan operation:

    >>> import vortex as vx
    >>> vxf = vx.open("data.vortex") # doctest: +SKIP
    >>> array_iterator = vxf.scan() # doctest: +SKIP

    See also: :class:`vortex.dataset.VortexDataset`
    """

    return VortexFile(_file.open(path, store=store, without_segment_cache=without_segment_cache))


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

    @property
    def schema(self) -> pa.Schema:
        """The Arrow schema of the file."""
        return self.dtype.to_arrow_schema()

    def splits(self) -> list[tuple[int, int]]:
        return self._file.splits()

    def scan(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
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
        limit : :class:`int` | None
            The maximum number of rows to read after filtering. If None, read all rows.
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
        return self._file.scan(projection, expr=expr, limit=limit, indices=indices, batch_size=batch_size)

    def to_repeated_scan(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
    ) -> RepeatedScan:
        """Prepare a scan of the Vortex file for repeated reads, returning a :class:`vortex.RepeatedScan`.

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
        """
        return RepeatedScan(
            self._file.prepare(projection, expr=expr, limit=limit, indices=indices, batch_size=batch_size)
        )

    def to_arrow(
        self,
        columns: IntoProjection = None,
        *,
        projection: IntoProjection = None,
        filter: pc.Expression | Expr | None = None,
        limit: int | None = None,
        expr: Expr | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
        filter_policy: str = "pushdown",
    ) -> RecordBatchReader:
        """Scan the Vortex file as a :class:`pyarrow.RecordBatchReader`.

        Parameters
        ----------
        columns : :class:`vortex.Expr` | list[str] | None
            Either an expression over the columns of the file (only referenced columns will be read
            from the file) or an explicit list of desired columns.
        filter : :class:`vortex.Expr` | :class:`pyarrow.compute.Expression` | None
            The predicate used to filter rows. The filter columns need not appear in the projection.
        limit : :class:`int` | None
            The maximum number of rows to read after filtering. If None, read all rows.
        indices : :class:`vortex.Array` | None
            The indices of the rows to read. Must be sorted and non-null.
        batch_size : :class:`int` | None
            The number of rows to read per chunk.
        filter_policy : :class:`str`
            ``"pushdown"`` raises if a PyArrow filter cannot be pushed into Vortex. ``"fallback"``
            reads the requested rows and applies the PyArrow filter with Arrow after the scan.

        """
        columns = self._resolve_columns(columns, projection)
        filter = self._resolve_filter(filter, expr)
        self._check_filter_policy(filter_policy)

        planned_filter: Expr | None = None
        if filter is not None:
            if isinstance(filter, pc.Expression):
                try:
                    planned_filter = ensure_vortex_expression(filter, schema=self.schema)
                except Exception:
                    if filter_policy == "fallback":
                        return self._to_arrow_with_arrow_filter_fallback(
                            columns,
                            filter,
                            limit=limit,
                            indices=indices,
                            batch_size=batch_size,
                        )
                    raise
            else:
                planned_filter = ensure_vortex_expression(filter, schema=self.schema)

        return self._file.to_arrow(columns, expr=planned_filter, limit=limit, indices=indices, batch_size=batch_size)

    def to_table(
        self,
        columns: IntoProjection = None,
        *,
        projection: IntoProjection = None,
        filter: pc.Expression | Expr | None = None,
        limit: int | None = None,
        expr: Expr | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
        filter_policy: str = "pushdown",
    ) -> pa.Table:
        """Scan the Vortex file as a :class:`pyarrow.Table`."""
        return self.to_arrow(
            columns,
            projection=projection,
            filter=filter,
            limit=limit,
            expr=expr,
            indices=indices,
            batch_size=batch_size,
            filter_policy=filter_policy,
        ).read_all()

    @staticmethod
    def _resolve_columns(columns: IntoProjection, projection: IntoProjection) -> IntoProjection:
        if projection is not None:
            if columns is not None:
                raise ValueError("use either columns or projection, not both")
            return projection
        return columns

    @staticmethod
    def _resolve_filter(
        filter: pc.Expression | Expr | None,
        expr: Expr | None,
    ) -> pc.Expression | Expr | None:
        if expr is not None:
            if filter is not None:
                raise ValueError("use either filter or expr, not both")
            return expr
        return filter

    @staticmethod
    def _check_filter_policy(filter_policy: str) -> None:
        if filter_policy not in {"pushdown", "fallback"}:
            raise ValueError("filter_policy must be 'pushdown' or 'fallback'")

    def _to_arrow_with_arrow_filter_fallback(
        self,
        columns: IntoProjection,
        filter: pc.Expression,
        *,
        limit: int | None,
        indices: Array | None,
        batch_size: int | None,
    ) -> RecordBatchReader:
        if columns is not None and not isinstance(columns, list):
            raise ValueError("filter_policy='fallback' only supports list[str] column selections")

        table = self._file.to_arrow(None, expr=None, limit=None, indices=indices, batch_size=batch_size).read_all()
        table = self._arrow_filter_compatible_table(table)
        table = ds.dataset(table).to_table(filter=filter)
        if limit is not None:
            table = table.slice(0, limit)
        if columns is not None:
            table = table.select(columns)

        batches = table.to_batches(max_chunksize=batch_size) if batch_size is not None else table.to_batches()
        return pa.RecordBatchReader.from_batches(table.schema, batches)

    @staticmethod
    def _arrow_filter_compatible_table(table: pa.Table) -> pa.Table:
        fields = []
        changed = False
        for field in table.schema:
            if field.type == pa.string_view():
                fields.append(field.with_type(pa.string()))
                changed = True
            elif field.type == pa.binary_view():
                fields.append(field.with_type(pa.binary()))
                changed = True
            else:
                fields.append(field)
        if not changed:
            return table
        return table.cast(pa.schema(fields))

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
            n_rows: int | None,
            _batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            vx_predicate: Expr | None = None if predicate is None else polars_to_vortex(predicate)

            reader = self.to_arrow(columns=with_columns, filter=vx_predicate, limit=n_rows)

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
