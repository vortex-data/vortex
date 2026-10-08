# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, TYPE_CHECKING, final

import pyarrow as pa

from ._lib import file as _file
from ._lib.arrays import Array
from ._lib.dtype import DType
from ._lib.expr import Expr
from ._lib.file import Footer, SegmentCache
from ._lib.iter import ArrayIterator
from .dataset import VortexDataset
from .io import ReadAt, ReadBytesAt
from .scan import RepeatedScan
from .store import (
    AzureStore,
    CosStore,
    GCSStore,
    HfStore,
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
    store: AzureStore | CosStore | GCSStore | HfStore | HTTPStore | LocalStore | MemoryStore | S3Store | None = None,
    footer: Footer | None = None,
    without_segment_cache: bool = False,
    segment_cache: SegmentCache | None = None,
    cache_key: str | None = None,
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
    footer : :class:`vortex.file.Footer` | None
        The :attr:`VortexFile.footer` of an earlier open of the same file. Opening then does not
        read the footer. Vortex cannot check that it belongs to this file, so the file must be the
        same one and must not have changed since.
    without_segment_cache : :class:`bool`
        If true, disable the segment cache for this file, useful when memory is constrained.
    segment_cache : :class:`vortex.SegmentCache` | None
        A cache shared with other files, used in place of this file's own segment cache.
        Requires ``cache_key``.
    cache_key : :class:`str` | None
        Identifies this file's contents within ``segment_cache``. Files opened with the same key
        share cached segments, so the key must change whenever the file does.

    Examples
    --------
    Open a Vortex file and perform a scan operation:

    >>> import vortex as vx
    >>> vxf = vx.open("data.vortex") # doctest: +SKIP
    >>> array_iterator = vxf.scan() # doctest: +SKIP

    See also: :func:`vortex.open_readable`, :class:`vortex.dataset.VortexDataset`
    """

    return VortexFile(
        _file.open(
            path,
            store=store,
            footer=footer,
            without_segment_cache=without_segment_cache,
            segment_cache=segment_cache,
            cache_key=cache_key,
        )
    )


def open_readable(
    reader: ReadBytesAt | ReadAt | IO[bytes],
    *,
    footer: Footer | None = None,
    concurrency: int | None = None,
    without_segment_cache: bool = False,
    segment_cache: SegmentCache | None = None,
    cache_key: str | None = None,
) -> VortexFile:
    """
    Lazily open a Vortex file through a Python object that performs the IO itself.

    Use this for storage only reachable from Python. Storage with a native object store should be
    opened with :func:`vortex.open` instead, which does its IO without taking the GIL.

    Parameters
    ----------
    reader : :class:`vortex.io.ReadBytesAt` | :class:`vortex.io.ReadAt` | binary file object
        An object implementing :class:`vortex.io.ReadBytesAt` or :class:`vortex.io.ReadAt`, or a
        binary file object with ``seek`` and ``readinto`` (or ``read``), such as
        ``open(path, "rb")``, :class:`io.BytesIO` or an fsspec file. Vortex does not close it; keep
        it open for as long as the returned file, or anything scanned from it, is in use.
    footer : :class:`vortex.file.Footer` | None
        The :attr:`VortexFile.footer` of an earlier open of the same file. Opening then does no IO,
        which saves the footer read when a file is opened again and again. Vortex checks only that
        the footer fits within the size of ``reader``, so the file must not have changed since.
    concurrency : :class:`int` | None
        The most reads to have in flight at once through a :class:`vortex.io.ReadBytesAt` or
        :class:`vortex.io.ReadAt`, 192 by default. Not accepted for a file object, whose reads are
        serialized because each one has to ``seek`` first.
    without_segment_cache : :class:`bool`
        If true, disable the segment cache for this file, useful when memory is constrained.
    segment_cache : :class:`vortex.SegmentCache` | None
        A cache shared with other files, used in place of this file's own segment cache.
        Requires ``cache_key``.
    cache_key : :class:`str` | None
        Identifies this file's contents within ``segment_cache``. Files opened with the same key
        share cached segments, so the key must change whenever the file does.

    Examples
    --------
    Open a Vortex file through an fsspec file object:

    >>> import fsspec # doctest: +SKIP
    >>> import vortex as vx
    >>> with fsspec.open("memory://data.vortex", "rb") as f: # doctest: +SKIP
    ...     table = vx.open_readable(f).to_arrow().read_all()

    Open the same file again without reading its footer:

    >>> with fsspec.open("memory://data.vortex", "rb") as f: # doctest: +SKIP
    ...     footer = vx.open_readable(f).footer
    ...     vxf = vx.open_readable(f, footer=footer)
    """

    return VortexFile(
        _file.open_readable(
            reader,
            footer=footer,
            concurrency=concurrency,
            without_segment_cache=without_segment_cache,
            segment_cache=segment_cache,
            cache_key=cache_key,
        )
    )


@final
class VortexFile:
    def __init__(self, file: _file.VortexFile) -> None:
        self._file = file

    def __len__(self) -> int:
        return self._file.__len__()

    @property
    def dtype(self) -> DType:
        """The dtype of the file."""
        return self._file.dtype

    @property
    def path(self) -> str:
        """The path or URL this file was opened from."""
        return self._file.path

    @property
    def footer(self) -> Footer:
        """The parsed footer, to pass to :func:`vortex.open` or :func:`vortex.open_readable` to open this file again."""
        return self._file.footer

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
        -- child 0 type: string_view
          [
            "Joseph",
            null,
            "Angela",
            "Mikhail",
            null
          ]
        -- child 1 type: int64
          [
            25,
            31,
            null,
            57,
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
        -- child 0 type: string_view
          [
            "Mikhail"
          ]
        -- child 1 type: int64
          [
            57
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
        projection: IntoProjection = None,
        *,
        limit: int | None = None,
        expr: Expr | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
        schema: pa.Schema | None = None,
    ) -> RecordBatchReader:
        """Scan the Vortex file as a :class:`pyarrow.RecordBatchReader`.

        Parameters
        ----------
        projection : :class:`vortex.Expr` | list[str] | None
            Either an expression over the columns of the file (only referenced columns will be read
            from the file) or an explicit list of desired columns.
        expr : :class:`vortex.Expr` | None
            The predicate used to filter rows. The filter columns need not appear in the projection.
        indices : :class:`vortex.Array` | None
            The indices of the rows to read. Must be strictly increasing and non-null.
        batch_size : :class:`int` | None
            The number of rows to read per chunk.
        schema : :class:`pyarrow.Schema` | None
            The Arrow schema to return. Use ``pyarrow.string()`` for ``StringArray`` fields.
            Use ``pyarrow.binary()`` for ``BinaryArray`` fields.

        """
        return self._file.to_arrow(
            projection, expr=expr, limit=limit, indices=indices, batch_size=batch_size, schema=schema
        )

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

            reader = self.to_arrow(projection=with_columns, expr=vx_predicate, limit=n_rows)

            for batch in reader:
                batch = pl.DataFrame._from_arrow(batch, rechunk=False)
                # TODO(ngates): set sortedness on DataFrame based on stats?
                yield batch

            # Make sure we always yield at least one empty DataFrame
            yield pl.DataFrame._from_arrow(
                data=pa.RecordBatch.from_arrays(
                    [pa.array([], type=field.type) for field in reader.schema],
                    schema=reader.schema,
                ),
            )

        # https://github.com/pola-rs/polars/pull/24125
        return register_io_source(_io_source, schema=schema)  # ty: ignore[invalid-argument-type]
