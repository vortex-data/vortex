# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import os
import warnings
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from functools import reduce
from typing import TYPE_CHECKING, final

import pyarrow as pa
import pyarrow.dataset
from typing_extensions import override

from ._lib import dataset as _dataset
from ._lib import file as _file
from ._lib.runtime import set_worker_threads as _set_worker_threads
from ._lib.runtime import worker_threads as _worker_threads
from .arrays import array
from .arrow.expression import ensure_vortex_expression
from .expr import Expr, and_
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

if TYPE_CHECKING:
    import polars


@contextmanager
def _temporary_worker_threads(use_threads: bool) -> Generator[None]:
    previous_workers = _worker_threads()
    if use_threads:
        _set_worker_threads(None)
    else:
        _set_worker_threads(0)

    try:
        yield
    finally:
        _set_worker_threads(previous_workers)


def _read_batches_with_temporary_worker_threads(
    reader: pyarrow.RecordBatchReader, use_threads: bool
) -> Iterator[pyarrow.RecordBatch]:
    with _temporary_worker_threads(use_threads):
        yield from reader


@final
class VortexDataset(pyarrow.dataset.Dataset):
    """Read Vortex files with row filter and column selection pushdown.

    This class implements the :class:`.pyarrow.dataset.Dataset` interface which enables its use with
    Polars, DuckDB, Pandas and others.

    """

    def __init__(self, dataset: _dataset.VortexDataset, *, filters: list[Expr] | None = None) -> None:
        self._dataset = dataset
        self._filters: list[Expr] = filters or []

    @staticmethod
    def from_url(url: str) -> VortexDataset:
        return VortexDataset(_dataset.dataset_from_url(url))

    @staticmethod
    def from_path(path: str) -> VortexDataset:
        return VortexDataset(_file.open(path).to_dataset())

    @property
    @override
    def schema(self) -> pyarrow.Schema:
        return self._dataset.schema()

    @property
    def concurrency(self) -> int | None:
        """The maximum number of splits a read fetches and decodes ahead, unless it sets its own.

        ``None`` means four splits per worker thread.
        """
        return self._dataset.concurrency

    @property
    def file_count(self) -> int:
        """The number of Vortex files in the dataset."""
        return self._dataset.file_count

    @override
    def count_rows(
        self,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> int:
        """Count the number of rows in this dataset."""
        if batch_readahead is not None:
            raise ValueError("batch_readahead not supported")
        if fragment_readahead is not None:
            raise ValueError("fragment_readahead not supported")
        if fragment_scan_options is not None:
            raise ValueError("fragment_scan_options not supported")
        if cache_metadata is not None:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata setting.")
        del memory_pool
        with _temporary_worker_threads(use_threads):
            return self._dataset.count_rows(
                row_filter=self._filter_expression(filter),
                split_by=batch_size,
                row_range=_row_range,
                concurrency=concurrency,
            )

    def _filter_expression(self, expression: pyarrow.dataset.Expression | Expr | None) -> Expr | None:
        if expression is None:
            if self._filters:
                return reduce(and_, self._filters)
            return None
        return reduce(and_, [*self._filters, ensure_vortex_expression(expression, schema=self.schema)])

    @override
    def filter(self, expression: pyarrow.dataset.Expression | Expr) -> VortexDataset:
        """A new Dataset with a filter condition applied.

        Successively calling this method conjuncts all the filter expressions together.
        """
        return VortexDataset(
            self._dataset, filters=[*self._filters, ensure_vortex_expression(expression, schema=self.schema)]
        )

    @override
    def get_fragments(self, filter: pyarrow.dataset.Expression | Expr | None = None) -> Iterator[VortexFragment]:
        """A fragment for each file in the Dataset."""

        for left, right in self._dataset.splits():
            yield VortexFragment(self, (left, right))

    @override
    def head(
        self,
        num_rows: int,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> pyarrow.Table:
        """Load the first `num_rows` of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evaluates to ``True``. Any rows for which
            this expression evaluates to ``Null`` is removed.
        batch_size : int
            The maximum number of rows per batch.
        batch_readahead : int
            Not implemented.
        fragment_readahead : int
            Not implemented.
        fragment_scan_options : :class:`.pyarrow.dataset.FragmentScanOptions`
            Not implemented.
        use_threads : bool
            If ``True``, temporarily use available parallelism. If ``False``,
            temporarily disable Vortex background workers.
        memory_pool : :class:`.pyarrow.MemoryPool` | None
            Not implemented.
        concurrency : int | None
            The maximum number of splits fetched and decoded ahead of the consumer, across all the
            files of the dataset. Defaults to the dataset's :attr:`concurrency`.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        if batch_size is not None:
            raise ValueError("batch_size is not supported")
        if batch_readahead is not None:
            raise ValueError("batch_readahead not supported")
        if fragment_readahead is not None:
            raise ValueError("fragment_readahead not supported")
        if fragment_scan_options is not None:
            raise ValueError("fragment_scan_options not supported")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        if cache_metadata is not None:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata setting.")
        del memory_pool

        reader = self._dataset.to_record_batch_reader(
            columns=columns,
            row_filter=self._filter_expression(filter),
            row_range=_row_range,
            concurrency=concurrency,
        )
        # Stop reading once there are enough rows: a dataset can be far larger than its head.
        batches: list[pyarrow.RecordBatch] = []
        remaining = num_rows
        with _temporary_worker_threads(use_threads):
            for batch in reader:
                if remaining <= 0:
                    break
                batches.append(batch.slice(0, remaining))
                remaining -= batches[-1].num_rows
        return pyarrow.Table.from_batches(batches, schema=reader.schema)

    @override
    def join(
        self,
        right_dataset: pyarrow.dataset.Dataset,
        keys: str | list[str],
        right_keys: str | list[str] | None = None,
        join_type: str = "left outer",
        left_suffix: str | None = None,
        right_suffix: str | None = None,
        coalesce_keys: bool = True,
        use_threads: bool = True,
    ) -> pyarrow.dataset.InMemoryDataset:
        """Not implemented."""
        raise NotImplementedError("join")

    @override
    def join_asof(
        self,
        right_dataset: pyarrow.dataset.Dataset,
        on: str,
        by: str | list[str],
        tolerance: int,
        right_on: str | list[str] | None = None,
        right_by: str | list[str] | None = None,
    ) -> pyarrow.dataset.InMemoryDataset:
        """Not implemented."""
        raise NotImplementedError("join_asof")

    @override
    def replace_schema(self, schema: pyarrow.Schema) -> None:
        """Not implemented."""
        raise NotImplementedError("replace_schema")

    @override
    def scanner(
        self,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> pyarrow.dataset.Scanner:
        """Construct a :class:`.pyarrow.dataset.Scanner`.

        Parameters
        ----------
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evaluates to ``True``. Any rows for which
            this expression evaluates to ``Null`` is removed.
        batch_size : int
            The maximum number of rows per batch.
        batch_readahead : int
            Not implemented.
        fragment_readahead : int
            Not implemented.
        fragment_scan_options : :class:`.pyarrow.dataset.FragmentScanOptions`
            Not implemented.
        use_threads : bool
            If ``True``, temporarily use available parallelism. If ``False``,
            temporarily disable Vortex background workers.
        memory_pool : :class:`.pyarrow.MemoryPool` | None
            Not implemented.
        concurrency : int | None
            The maximum number of splits fetched and decoded ahead of the consumer, across all the
            files of the dataset. Defaults to the dataset's :attr:`concurrency`.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        return VortexScanner(
            self,
            columns,
            filter,
            batch_size,
            batch_readahead,
            fragment_readahead,
            fragment_scan_options,
            use_threads,
            cache_metadata,
            memory_pool,
            _row_range,
            concurrency=concurrency,
        )

    @override
    def sort_by(  # ty: ignore[invalid-method-override]
        self, sorting: str | list[tuple[str, str]], **kwargs: object
    ) -> pyarrow.dataset.InMemoryDataset:
        """Not implemented."""
        raise NotImplementedError("sort_by")

    @override
    def take(  # ty: ignore[invalid-method-override]
        self,
        indices: pyarrow.Array[
            pyarrow.Int8Scalar
            | pyarrow.Int16Scalar
            | pyarrow.Int32Scalar
            | pyarrow.Int64Scalar
            | pyarrow.UInt8Scalar
            | pyarrow.UInt16Scalar
            | pyarrow.UInt32Scalar
            | pyarrow.UInt64Scalar
        ],
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> pyarrow.Table:
        """Load a subset of rows identified by their absolute indices.

        Parameters
        ----------
        indices : :class:`.pyarrow.Array`
            A numeric array of absolute indices into `self` indicating which rows to keep.
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evaluates to ``True``. Any rows for which
            this expression evaluates to ``Null`` is removed.
        batch_size : int
            The maximum number of rows per batch.
        batch_readahead : int
            Not implemented.
        fragment_readahead : int
            Not implemented.
        fragment_scan_options : :class:`.pyarrow.dataset.FragmentScanOptions`
            Not implemented.
        use_threads : bool
            If ``True``, temporarily use available parallelism. If ``False``,
            temporarily disable Vortex background workers.
        cache_metadata : bool
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool` | None
            Not implemented.
        concurrency : int | None
            The maximum number of splits fetched and decoded ahead of the consumer, across all the
            files of the dataset. Defaults to the dataset's :attr:`concurrency`.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        with _temporary_worker_threads(use_threads):
            return self._dataset.to_array(
                columns=columns,
                row_filter=self._filter_expression(filter),
                indices=array(indices.cast(pa.uint64())),
                row_range=_row_range,
                concurrency=concurrency,
            ).to_arrow_table()

    def to_record_batch_reader(
        self,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> pyarrow.RecordBatchReader:
        """Construct a :class:`.pyarrow.RecordBatchReader`.

        Parameters
        ----------
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evaluates to ``True``. Any rows for which
            this expression evaluates to ``Null`` is removed.
        batch_size : int
            The maximum number of rows per batch.
        batch_readahead : int
            Not implemented.
        fragment_readahead : int
            Not implemented.
        fragment_scan_options : :class:`.pyarrow.dataset.FragmentScanOptions`
            Not implemented.
        use_threads : bool
            If ``True``, temporarily use available parallelism. If ``False``,
            temporarily disable Vortex background workers.
        memory_pool : :class:`.pyarrow.MemoryPool` | None
            Not implemented.
        concurrency : int | None
            The maximum number of splits fetched and decoded ahead of the consumer, across all the
            files of the dataset. Defaults to the dataset's :attr:`concurrency`.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        if batch_readahead is not None:
            raise ValueError("batch_readahead not supported")
        if fragment_readahead is not None:
            raise ValueError("fragment_readahead not supported")
        if fragment_scan_options is not None:
            raise ValueError("fragment_scan_options not supported")
        if cache_metadata is not None:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata setting.")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        del memory_pool
        reader = self._dataset.to_record_batch_reader(
            columns=columns,
            row_filter=self._filter_expression(filter),
            split_by=batch_size,
            row_range=_row_range,
            concurrency=concurrency,
        )
        return pyarrow.RecordBatchReader.from_batches(
            reader.schema, _read_batches_with_temporary_worker_threads(reader, use_threads)
        )

    @override
    def to_batches(
        self,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> Iterator[pyarrow.RecordBatch]:
        """Construct an iterator of :class:`.pyarrow.RecordBatch`.

        Parameters
        ----------
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evaluates to ``True``. Any rows for which
            this expression evaluates to ``Null`` is removed.
        batch_size : int
            The maximum number of rows per batch.
        batch_readahead : int
            Not implemented.
        fragment_readahead : int
            Not implemented.
        fragment_scan_options : :class:`.pyarrow.dataset.FragmentScanOptions`
            Not implemented.
        use_threads : bool
            If ``True``, temporarily use available parallelism. If ``False``,
            temporarily disable Vortex background workers.
        cache_metadata : bool
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool` | None
            Not implemented.
        concurrency : int | None
            The maximum number of splits fetched and decoded ahead of the consumer, across all the
            files of the dataset. Defaults to the dataset's :attr:`concurrency`.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        record_batch_reader = self.to_record_batch_reader(
            columns,
            filter,
            batch_size,
            batch_readahead,
            fragment_readahead,
            fragment_scan_options,
            use_threads,
            cache_metadata,
            memory_pool,
            _row_range,
            concurrency=concurrency,
        )
        yield from record_batch_reader

    @override
    def to_table(
        self,
        columns: list[str] | dict[str, pyarrow.dataset.Expression] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> pyarrow.Table:
        """Construct an Arrow :class:`.pyarrow.Table`.

        Parameters
        ----------
        columns : list of str, dict[str, :class:`.pyarrow.dataset.Expression`] | None
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evaluates to ``True``. Any rows for which
            this expression evaluates to ``Null`` is removed.
        batch_size : int
            The maximum number of rows per batch.
        batch_readahead : int
            Not implemented.
        fragment_readahead : int
            Not implemented.
        fragment_scan_options : :class:`.pyarrow.dataset.FragmentScanOptions`
            Not implemented.
        use_threads : bool
            If ``True``, temporarily use available parallelism. If ``False``,
            temporarily disable Vortex background workers.
        memory_pool : :class:`.pyarrow.MemoryPool` | None
            Not implemented.
        concurrency : int | None
            The maximum number of splits fetched and decoded ahead of the consumer, across all the
            files of the dataset. Defaults to the dataset's :attr:`concurrency`.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        if batch_size is not None:
            raise ValueError("batch_size is not supported")
        if batch_readahead is not None:
            raise ValueError("batch_readahead not supported")
        if fragment_readahead is not None:
            raise ValueError("fragment_readahead not supported")
        if fragment_scan_options is not None:
            raise ValueError("fragment_scan_options not supported")
        if cache_metadata is not None:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata setting.")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        del memory_pool

        if isinstance(columns, dict):
            raise ValueError(
                "VortexDataset does not currently support a dict of expressions as the 'column' parameter."
            )

        with _temporary_worker_threads(use_threads):
            return self._dataset.to_array(
                columns=columns,
                row_filter=self._filter_expression(filter),
                row_range=_row_range,
                concurrency=concurrency,
            ).to_arrow_table()

    def to_polars(self, *, concurrency: int | None = None) -> polars.LazyFrame:
        """Read the dataset as a :class:`polars.LazyFrame`, pushing column selection and filters into
        the scan of every file.

        Parameters
        ----------
        concurrency : int | None
            The maximum number of splits fetched and decoded ahead of Polars, across all the files of
            the dataset. Defaults to the dataset's :attr:`concurrency`.

        Examples
        --------

        >>> import pyarrow as pa
        >>> import polars as pl
        >>> import vortex as vx
        >>> for day in range(3):
        ...     vx.io.write(pa.table({"day": [day] * 4, "failure": [0, 1, 1, 0]}), f"day-{day}.vortex")
        >>> (
        ...     vx.open_dataset("day-*.vortex", concurrency=32)
        ...     .to_polars()
        ...     .filter(pl.col("failure") == 1)
        ...     .group_by("day")
        ...     .agg(pl.len().alias("failures"))
        ...     .sort("day")
        ...     .collect()
        ...     .to_dicts()
        ... )
        [{'day': 0, 'failures': 2}, {'day': 1, 'failures': 2}, {'day': 2, 'failures': 2}]
        """
        from vortex.polars_ import lazy_frame

        def read(columns: list[str] | None, predicate: Expr | None, limit: int | None) -> pyarrow.RecordBatchReader:
            # The dataset reads every file without a row limit; the frame trims the batches instead.
            del limit
            return self._dataset.to_record_batch_reader(
                columns=columns, row_filter=self._filter_expression(predicate), concurrency=concurrency
            )

        return lazy_frame(self.schema, read)


def from_url(url: str) -> VortexDataset:
    return VortexDataset(_dataset.dataset_from_url(url))


def open_dataset(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    store: AzureStore | CosStore | GCSStore | HfStore | HTTPStore | LocalStore | MemoryStore | S3Store | None = None,
    concurrency: int | None = None,
) -> VortexDataset:
    """Open many Vortex files as one :class:`VortexDataset`.

    The dataset reads the rows of every file in order, with column selection and filters pushed into
    the scan of each file. Reads keep up to ``concurrency`` splits in flight across file boundaries,
    so a dataset of many small files is read as concurrently as one large file. Opening the dataset
    reads only the footer of each file.

    Parameters
    ----------
    source : :class:`str` | :class:`os.PathLike` | list of them
        What to read: a Vortex file, a directory (every ``.vortex`` file below it, at any depth), a
        glob pattern such as ``"data/**/*.vortex"``, or a list of these, read in the order given.
        Each can be a local path or a URL such as ``s3://bucket/data/``; an object store URL ending
        in ``/`` names a prefix and reads every ``.vortex`` file below it. The files of a directory,
        prefix or pattern are read in sorted order. Every file must have the same dtype.
    store :
        An object store created from the `vortex.store` package. When given, every source is a
        path within it. By default the store is inferred from each source's URL.
    concurrency : :class:`int` | None
        The maximum number of splits each read fetches and decodes ahead of the consumer, unless the
        read sets its own. Query engines such as DuckDB and Polars read the dataset without setting
        one, so this is how to tune their reads. By default, four splits per worker thread.

    Examples
    --------

    Write a file per day, then count the failures across all of them with DuckDB:

    >>> import duckdb
    >>> import pyarrow as pa
    >>> import vortex as vx
    >>> for day in range(3):
    ...     vx.io.write(pa.table({"day": [day] * 4, "failure": [0, 1, 1, 0]}), f"day-{day}.vortex")
    >>> failures = vx.open_dataset("day-*.vortex")
    >>> failures.file_count
    3
    >>> duckdb.sql(
    ...     "SELECT day, count(*) AS n FROM failures WHERE failure = 1 GROUP BY day ORDER BY day"
    ... ).fetchall()
    [(0, 2), (1, 2), (2, 2)]
    """
    sources = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    return VortexDataset(
        _dataset.dataset_from_sources(
            [os.fspath(source) for source in sources], store=store, concurrency=concurrency
        )
    )


@final
class VortexFragment(pyarrow.dataset.Fragment):
    """Fragment of data from a :class:`.VortexDataset`."""

    def __init__(
        self,
        dataset: VortexDataset,
        _row_range: tuple[int, int],
    ) -> None:
        self._dataset = dataset
        self._row_range = _row_range

    @property
    @override
    def physical_schema(self) -> pyarrow.Schema:
        """Return the physical schema of this Fragment. This schema can be
        different from the dataset read schema."""
        return self._dataset.schema

    @property
    @override
    def partition_expression(self) -> pyarrow.dataset.Expression:
        """An Expression which evaluates to true for all data viewed by this
        Fragment."""
        raise NotImplementedError

    @override
    def scanner(
        self,
        schema: pyarrow.Schema | None = None,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.dataset.Scanner:
        """See :class:`vortex.dataset.VortexDataset.scanner`"""
        if schema:
            raise ValueError("schema is not supported")
        return self._dataset.scanner(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            cache_metadata=cache_metadata,
            memory_pool=memory_pool,
            _row_range=self._row_range,
        )

    @override
    def to_batches(
        self,
        schema: pyarrow.Schema | None = None,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> Iterator[pyarrow.RecordBatch]:
        """See :class:`vortex.dataset.VortexDataset.to_batches`"""
        if schema:
            raise ValueError("schema is not supported")
        return self._dataset.to_batches(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            cache_metadata=cache_metadata,
            memory_pool=memory_pool,
            _row_range=self._row_range,
        )

    @override
    def to_table(
        self,
        schema: pyarrow.Schema | None = None,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """See :class:`vortex.dataset.VortexDataset.to_table`"""
        if schema:
            raise ValueError("schema is not supported")
        return self._dataset.to_table(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            cache_metadata=cache_metadata,
            memory_pool=memory_pool,
            _row_range=self._row_range,
        )

    @override
    def take(  # ty: ignore[invalid-method-override]
        self,
        indices: pyarrow.Array[
            pyarrow.Int8Scalar
            | pyarrow.Int16Scalar
            | pyarrow.Int32Scalar
            | pyarrow.Int64Scalar
            | pyarrow.UInt8Scalar
            | pyarrow.UInt16Scalar
            | pyarrow.UInt32Scalar
            | pyarrow.UInt64Scalar
        ],
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """See :class:`vortex.dataset.VortexDataset.take`

        Warnings
        --------

        The indices are indices into *the file*, not indices into this fragment of the file.

        """
        return self._dataset.take(
            indices=indices,
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            cache_metadata=cache_metadata,
            memory_pool=memory_pool,
            _row_range=self._row_range,
        )

    @override
    def head(
        self,
        num_rows: int,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """See :class:`vortex.dataset.VortexDataset.head`"""
        return self._dataset.head(
            num_rows=num_rows,
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            cache_metadata=cache_metadata,
            memory_pool=memory_pool,
            _row_range=self._row_range,
        )

    # regarding the ignore: https://github.com/zen-xu/pyarrow-stubs/pull/258
    @override
    def count_rows(  # ty: ignore[invalid-method-override]
        self,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> int:
        """See :class:`vortex.dataset.VortexDataset.count_rows`"""
        return self._dataset.count_rows(
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            cache_metadata=cache_metadata,
            memory_pool=memory_pool,
            _row_range=self._row_range,
        )


@final
class VortexScanner(pyarrow.dataset.Scanner):
    """A PyArrow Dataset Scanner that reads from a Vortex Array.

    Parameters
    ----------
    dataset : VortexDataset
        The dataset to scan.
    columns : list of str
        The columns to keep, identified by name.
    filter : :class:`.pyarrow.dataset.Expression`
        Keep only rows for which this expression evaluates to ``True``. Any rows for which
        this expression evaluates to ``Null`` is removed.
    batch_size : int
        The maximum number of rows per batch.
    batch_readahead : int
        Not implemented.
    fragment_readahead : int
        Not implemented.
    fragment_scan_options : :class:`.pyarrow.dataset.FragmentScanOptions`
        Not implemented.
    use_threads : bool
        If ``True``, temporarily use available parallelism. If ``False``,
        temporarily disable Vortex background workers.
    memory_pool : :class:`.pyarrow.MemoryPool` | None
        Not implemented.
    concurrency : int | None
        The maximum number of splits fetched and decoded ahead of the consumer, across all the
        files of the dataset. Defaults to the dataset's :attr:`VortexDataset.concurrency`.

    Returns
    -------
    table : :class:`.pyarrow.Table`

    """

    def __init__(
        self,
        dataset: VortexDataset,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
        _row_range: tuple[int, int] | None = None,
        *,
        concurrency: int | None = None,
    ) -> None:
        self._dataset = dataset
        self._columns = columns
        self._filter = filter
        self._batch_size = batch_size
        self._batch_readahead = batch_readahead
        self._fragment_readahead = fragment_readahead
        self._fragment_scan_options = fragment_scan_options
        self._use_threads = use_threads
        self._cache_metadata = cache_metadata
        self._memory_pool = memory_pool
        self._row_range = _row_range
        self._concurrency = concurrency

    @property
    def schema(self) -> pyarrow.Schema:
        return self._dataset.schema

    @property
    @override
    def dataset_schema(self) -> pyarrow.Schema:
        return self._dataset.schema

    @property
    @override
    def projected_schema(self) -> pyarrow.Schema:
        if self._columns:
            fields: list[pa.Field[pa.DataType]] = [self._dataset.schema.field(c) for c in self._columns]
            return pyarrow.schema(fields)
        return self._dataset.schema

    @override
    def count_rows(self) -> int:
        return self._dataset.count_rows(
            self._filter,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._fragment_scan_options,
            self._use_threads,
            self._cache_metadata,
            self._memory_pool,
            self._row_range,
            concurrency=self._concurrency,
        )

    @override
    def head(self, num_rows: int) -> pyarrow.Table:
        """Load the first `num_rows` of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to read.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        return self._dataset.head(
            num_rows,
            self._columns,
            self._filter,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._fragment_scan_options,
            self._use_threads,
            self._cache_metadata,
            self._memory_pool,
            self._row_range,
            concurrency=self._concurrency,
        )

    @override
    def scan_batches(self) -> Iterator[pyarrow.dataset.TaggedRecordBatch]:  # ty: ignore[invalid-method-override]
        """Not implemented."""
        raise NotImplementedError("scan batches")

    @override
    def to_batches(self) -> Iterator[pyarrow.RecordBatch]:
        """Construct an iterator of :class:`.pyarrow.RecordBatch`.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        return self._dataset.to_batches(
            self._columns,
            self._filter,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._fragment_scan_options,
            self._use_threads,
            self._cache_metadata,
            self._memory_pool,
            self._row_range,
            concurrency=self._concurrency,
        )

    @override
    def to_reader(self) -> pyarrow.RecordBatchReader:
        """Construct a :class:`.pyarrow.RecordBatchReader`.


        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        return self._dataset.to_record_batch_reader(
            self._columns,
            self._filter,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._fragment_scan_options,
            self._use_threads,
            self._cache_metadata,
            self._memory_pool,
            self._row_range,
            concurrency=self._concurrency,
        )

    @override
    def to_table(self) -> pyarrow.Table:
        """Construct an Arrow :class:`.pyarrow.Table`.


        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        return self._dataset.to_table(
            self._columns,
            self._filter,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._fragment_scan_options,
            self._use_threads,
            self._cache_metadata,
            self._memory_pool,
            self._row_range,
            concurrency=self._concurrency,
        )
