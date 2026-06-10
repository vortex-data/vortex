# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import final

import pyarrow as pa
import pyarrow.compute
import pyarrow.dataset
from typing_extensions import override

from ._lib import dataset as _dataset  # pyright: ignore[reportMissingModuleSource]
from ._lib import file as _file  # pyright: ignore[reportMissingModuleSource]
from ._lib.runtime import set_worker_threads as _set_worker_threads  # pyright: ignore[reportMissingModuleSource]
from ._lib.runtime import worker_threads as _worker_threads  # pyright: ignore[reportMissingModuleSource]
from .arrays import array
from .arrow.expression import ensure_vortex_expression
from .expr import Expr, and_


@contextmanager
def _temporary_worker_threads(use_threads: bool) -> Iterator[None]:
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

    def __init__(self, dataset: _dataset.VortexDataset, *, filters: list[Expr] | None = None):
        self._dataset = dataset
        self._filters: list[Expr] = filters or []

    @staticmethod
    def from_url(url: str):
        return VortexDataset(_dataset.dataset_from_url(url))

    @staticmethod
    def from_path(path: str):
        return VortexDataset(_file.open(path).to_dataset())

    @property
    @override
    def schema(self) -> pyarrow.Schema:
        return self._dataset.schema()

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
                row_filter=self._filter_expression(filter), split_by=batch_size, row_range=_row_range
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

        with _temporary_worker_threads(use_threads):
            return (
                self._dataset.to_array(
                    columns=columns,
                    row_filter=self._filter_expression(filter),
                    row_range=_row_range,
                )
                .slice(0, num_rows)
                .to_arrow_table()
            )

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
        )

    @override
    def sort_by(self, sorting: str | list[tuple[str, str]], **kwargs) -> pyarrow.dataset.InMemoryDataset:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportIncompatibleMethodOverride]
        """Not implemented."""
        raise NotImplementedError("sort_by")

    @override
    def take(  # pyright: ignore[reportIncompatibleMethodOverride]
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
            columns=columns, row_filter=self._filter_expression(filter), split_by=batch_size, row_range=_row_range
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
                columns=columns, row_filter=self._filter_expression(filter), row_range=_row_range
            ).to_arrow_table()


def from_url(url: str) -> VortexDataset:
    return VortexDataset(_dataset.dataset_from_url(url))


@final
class VortexFragment(pyarrow.dataset.Fragment):
    """Fragment of data from a :class:`.VortexDataset`."""

    def __init__(
        self,
        dataset: VortexDataset,
        _row_range: tuple[int, int],
    ):
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
    def take(  # pyright: ignore[reportIncompatibleMethodOverride]
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
    def count_rows(  # pyright: ignore[reportIncompatibleMethodOverride]
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

    Returns
    -------
    table : :class:`.pyarrow.Table`

    """

    def __init__(
        self,
        dataset: VortexDataset | VortexMultiDataset,
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
    ):
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

    @property
    def schema(self):
        return self._dataset.schema

    @property
    @override
    def dataset_schema(self) -> pyarrow.Schema:
        return self._dataset.schema

    @property
    @override
    def projected_schema(self) -> pyarrow.Schema:
        if self._columns:
            fields: list[pa.Field[pa.DataType]] = [
                self._dataset.schema.field(c)  # pyright: ignore[reportUnknownMemberType]
                for c in self._columns
            ]
            return pyarrow.schema(fields)
        return self._dataset.schema

    @override
    def count_rows(self):
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
        )

    @override
    def scan_batches(self) -> Iterator[pyarrow.dataset.TaggedRecordBatch]:  # pyright: ignore[reportIncompatibleMethodOverride]
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
        )


@final
class VortexMultiDataset(pyarrow.dataset.Dataset):
    """Read multiple Vortex files sharing one schema as a single :class:`pyarrow.dataset.Dataset`.

    Use :func:`vortex.dataset.dataset` to construct one from a list of paths or a directory.
    """

    def __init__(self, datasets: list[VortexDataset]):
        if not datasets:
            raise ValueError("VortexMultiDataset requires at least one dataset")
        schema = datasets[0].schema
        for child in datasets[1:]:
            if not child.schema.equals(schema):
                raise ValueError("all files in a Vortex dataset must share the same schema")
        self._children = datasets

    @property
    @override
    def schema(self) -> pyarrow.Schema:
        return self._children[0].schema

    def _projected_schema(self, columns: list[str] | None) -> pyarrow.Schema:
        if columns is None:
            return self.schema
        fields: list[pa.Field[pa.DataType]] = [
            self.schema.field(c)  # pyright: ignore[reportUnknownMemberType]
            for c in columns
        ]
        return pyarrow.schema(fields)

    @override
    def filter(self, expression: pyarrow.dataset.Expression | Expr) -> VortexMultiDataset:
        """A new Dataset with a filter condition applied to every file.

        Successively calling this method conjuncts all the filter expressions together.
        """
        return VortexMultiDataset([child.filter(expression) for child in self._children])

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
    ) -> int:
        """Count the number of rows across all files in this dataset."""
        if _row_range is not None:
            raise ValueError("_row_range is not supported on multi-file datasets")
        return sum(
            child.count_rows(
                filter,
                batch_size,
                batch_readahead,
                fragment_readahead,
                fragment_scan_options,
                use_threads,
                cache_metadata,
                memory_pool,
            )
            for child in self._children
        )

    @override
    def get_fragments(self, filter: pyarrow.dataset.Expression | Expr | None = None) -> Iterator[VortexFragment]:
        """A fragment for each split of each file in the Dataset."""
        for child in self._children:
            yield from child.get_fragments(filter)

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
    ) -> pyarrow.Table:
        """Load the first `num_rows` of the dataset, reading from as few files as possible.

        See :meth:`VortexDataset.head` for parameter documentation.
        """
        if _row_range is not None:
            raise ValueError("_row_range is not supported on multi-file datasets")
        tables: list[pyarrow.Table] = []
        remaining = num_rows
        for child in self._children:
            if remaining <= 0:
                break
            table = child.head(
                remaining,
                columns,
                filter,
                batch_size,
                batch_readahead,
                fragment_readahead,
                fragment_scan_options,
                use_threads,
                cache_metadata,
                memory_pool,
            )
            if len(table) > 0:
                tables.append(table)
                remaining -= len(table)
        if not tables:
            return self._projected_schema(columns).empty_table()
        return pyarrow.concat_tables(tables)

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
    ) -> pyarrow.dataset.Scanner:
        """Construct a :class:`.pyarrow.dataset.Scanner` over all files.

        See :meth:`VortexDataset.scanner` for parameter documentation.
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
        )

    @override
    def sort_by(self, sorting: str | list[tuple[str, str]], **kwargs) -> pyarrow.dataset.InMemoryDataset:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportIncompatibleMethodOverride]
        """Not implemented."""
        raise NotImplementedError("sort_by")

    @override
    def take(  # pyright: ignore[reportIncompatibleMethodOverride]
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
    ) -> pyarrow.Table:
        """Load a subset of rows identified by their absolute indices into the whole dataset.

        Indices are interpreted over the concatenation of all files, in order. The returned rows
        are in the same order as `indices`, which need not be sorted.

        See :meth:`VortexDataset.take` for parameter documentation.
        """
        if _row_range is not None:
            raise ValueError("_row_range is not supported on multi-file datasets")
        if filter is not None or any(child._filters for child in self._children):  # pyright: ignore[reportPrivateUsage]
            raise NotImplementedError("take with a filter is not supported on multi-file datasets")

        pc = pyarrow.compute
        indices64 = indices.cast(pa.int64())
        order = pc.sort_indices(indices64)  # pyright: ignore[reportUnknownMemberType]
        sorted_indices = indices64.take(order)

        total = 0
        tables: list[pyarrow.Table] = []
        for child in self._children:
            length = child.count_rows(use_threads=use_threads)
            in_child = pc.and_(  # pyright: ignore[reportUnknownMemberType]
                pc.greater_equal(sorted_indices, total),  # pyright: ignore[reportUnknownMemberType]
                pc.less(sorted_indices, total + length),  # pyright: ignore[reportUnknownMemberType]
            )
            local = pc.subtract(sorted_indices.filter(in_child), total)  # pyright: ignore[reportUnknownMemberType]
            if len(local) > 0:
                tables.append(
                    child.take(
                        local,
                        columns,
                        batch_size=batch_size,
                        batch_readahead=batch_readahead,
                        fragment_readahead=fragment_readahead,
                        fragment_scan_options=fragment_scan_options,
                        use_threads=use_threads,
                        cache_metadata=cache_metadata,
                        memory_pool=memory_pool,
                    )
                )
            total += length

        taken = sum(len(t) for t in tables)
        if taken != len(indices64):
            raise IndexError(f"indices out of bounds for dataset of {total} rows")
        if not tables:
            return self._projected_schema(columns).empty_table()

        combined = pyarrow.concat_tables(tables)
        inverse = pc.sort_indices(order)  # pyright: ignore[reportUnknownMemberType]
        return combined.take(inverse)

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
    ) -> pyarrow.RecordBatchReader:
        """Construct a :class:`.pyarrow.RecordBatchReader` over all files, in order.

        See :meth:`VortexDataset.to_record_batch_reader` for parameter documentation.
        """
        if _row_range is not None:
            raise ValueError("_row_range is not supported on multi-file datasets")

        def batches() -> Iterator[pyarrow.RecordBatch]:
            for child in self._children:
                yield from child.to_record_batch_reader(
                    columns,
                    filter,
                    batch_size,
                    batch_readahead,
                    fragment_readahead,
                    fragment_scan_options,
                    use_threads,
                    cache_metadata,
                    memory_pool,
                )

        return pyarrow.RecordBatchReader.from_batches(self._projected_schema(columns), batches())

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
    ) -> Iterator[pyarrow.RecordBatch]:
        """Construct an iterator of :class:`.pyarrow.RecordBatch` over all files, in order.

        See :meth:`VortexDataset.to_batches` for parameter documentation.
        """
        yield from self.to_record_batch_reader(
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
        )

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
    ) -> pyarrow.Table:
        """Construct an Arrow :class:`.pyarrow.Table` from all files, in order.

        See :meth:`VortexDataset.to_table` for parameter documentation.
        """
        if isinstance(columns, dict):
            raise ValueError(
                "VortexMultiDataset does not currently support a dict of expressions as the 'column' parameter."
            )
        return self.to_record_batch_reader(
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
        ).read_all()


def dataset(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> VortexDataset | VortexMultiDataset:
    """Open one or more Vortex files as a :class:`pyarrow.dataset.Dataset`.

    Parameters
    ----------
    source : :class:`str`, :class:`os.PathLike`, or a sequence of them
        A path or URL to a Vortex file, a local directory (which is searched recursively for
        ``*.vortex`` files), or a sequence of paths and URLs. All files must share the same
        schema. Partition key discovery (e.g. Hive-style paths) is not currently supported.

    Returns
    -------
    :class:`VortexDataset` for a single file, :class:`VortexMultiDataset` otherwise.

    Examples
    --------

    Open every Vortex file in a directory and read them as one table:

    >>> import vortex as vx
    >>> ds = vx.dataset.dataset("data/") # doctest: +SKIP
    >>> ds.to_table() # doctest: +SKIP
    """
    if isinstance(source, str | os.PathLike):
        path = os.fspath(source)
        if "://" in path:
            return VortexDataset.from_url(path)
        if os.path.isdir(path):
            files = sorted(str(file) for file in Path(path).rglob("*.vortex"))
            if not files:
                raise ValueError(f"no .vortex files found under directory {path!r}")
            return dataset(files)
        return VortexDataset.from_path(path)

    children: list[VortexDataset] = []
    for child_source in source:
        child = dataset(child_source)
        if isinstance(child, VortexMultiDataset):
            children.extend(child._children)  # pyright: ignore[reportPrivateUsage]
        else:
            children.append(child)
    if not children:
        raise ValueError("expected at least one path")
    if len(children) == 1:
        return children[0]
    return VortexMultiDataset(children)
