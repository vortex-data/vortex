# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import warnings
from collections.abc import Iterator
from functools import reduce
from typing import final

import pyarrow as pa
import pyarrow.dataset
from typing_extensions import override

from ._lib import dataset as _dataset  # pyright: ignore[reportMissingModuleSource]
from ._lib import file as _file  # pyright: ignore[reportMissingModuleSource]
from .arrays import array
from .arrow.expression import ensure_vortex_expression
from .expr import Expr, and_


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

    # regarding the ignore: https://github.com/zen-xu/pyarrow-stubs/pull/258
    @override
    def count_rows(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> int:
        """Count the number of rows in this dataset."""
        if batch_readahead is not None:
            raise ValueError("batch_readahead not supported")
        if fragment_readahead is not None:
            raise ValueError("fragment_readahead not supported")
        if fragment_scan_options is not None:
            raise ValueError("fragment_scan_options not supported")
        if use_threads:
            warnings.warn("Vortex does not support threading. Ignoring use_threads=True")
        if cache_metadata:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata=True")
        del memory_pool
        return self._dataset.count_rows(row_filter=self._filter_expression(filter), split_by=batch_size)

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
        yield VortexFragment(self)

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
        use_threads: bool | None = None,
        cache_metadata: bool = False,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """Load the first `num_rows` of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evalutes to ``True``. Any rows for which
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
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool`, optional
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
        if use_threads:
            warnings.warn("Vortex does not support threading. Ignoring use_threads=True")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        if cache_metadata:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata=True")
        del memory_pool

        return (
            self._dataset.to_array(columns=columns, row_filter=self._filter_expression(filter))
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
        use_threads: bool | None = None,
        cache_metadata: bool = False,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.dataset.Scanner:
        """Construct a :class:`.pyarrow.dataset.Scanner`.

        Parameters
        ----------
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evalutes to ``True``. Any rows for which
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
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool`, optional
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
        use_threads: bool | None = None,
        cache_metadata: bool = False,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """Load a subset of rows identified by their absolute indices.

        Parameters
        ----------
        indices : :class:`.pyarrow.Array`
            A numeric array of absolute indices into `self` indicating which rows to keep.
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evalutes to ``True``. Any rows for which
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
            Not implemented.
        cache_metadata : bool
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool`, optional
            Not implemented.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        return self._dataset.to_array(
            columns=columns,
            row_filter=self._filter_expression(filter),
            indices=array(indices.cast(pa.uint64())),
        ).to_arrow_table()

    def to_record_batch_reader(
        self,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        cache_metadata: bool | None = None,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.RecordBatchReader:
        """Construct a :class:`.pyarrow.RecordBatchReader`.

        Parameters
        ----------
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evalutes to ``True``. Any rows for which
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
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool`, optional
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
        if use_threads:
            warnings.warn("Vortex does not support threading. Ignoring use_threads=True")
        if cache_metadata:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata=True")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        del memory_pool
        return self._dataset.to_record_batch_reader(
            columns=columns, row_filter=self._filter_expression(filter), split_by=batch_size
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
        use_threads: bool | None = None,
        cache_metadata: bool = False,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> Iterator[pyarrow.RecordBatch]:
        """Construct an iterator of :class:`.pyarrow.RecordBatch`.

        Parameters
        ----------
        columns : list of str
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evalutes to ``True``. Any rows for which
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
            Not implemented.
        cache_metadata : bool
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool`, optional
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
        )
        while True:
            try:
                yield record_batch_reader.read_next_batch()
            except StopIteration:
                return

    @override
    def to_table(
        self,
        columns: list[str] | dict[str, pyarrow.dataset.Expression] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        cache_metadata: bool = True,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """Construct an Arrow :class:`.pyarrow.Table`.

        Parameters
        ----------
        columns : list of str, dict[str, :class:`.pyarrow.dataset.Expression`], optional
            The columns to keep, identified by name.
        filter : :class:`.pyarrow.dataset.Expression`
            Keep only rows for which this expression evalutes to ``True``. Any rows for which
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
            Not implemented.
        memory_pool : :class:`.pyarrow.MemoryPool`, optional
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
        if use_threads:
            warnings.warn("Vortex does not support threading. Ignoring use_threads=True")
        if cache_metadata:
            warnings.warn("Vortex does not support cache_metadata. Ignoring cache_metadata=True")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        del memory_pool

        if isinstance(columns, dict):
            raise ValueError(
                "VortexDataset does not currently support a dict of expressions as the 'column' parameter."
            )

        return self._dataset.to_array(columns=columns, row_filter=self._filter_expression(filter)).to_arrow_table()


def from_url(url: str) -> VortexDataset:
    return VortexDataset(_dataset.dataset_from_url(url))


@final
class VortexFragment(pyarrow.dataset.Fragment):
    """Fragment of data from a :class:`.VortexDataset`."""

    def __init__(self, dataset: VortexDataset):
        self._dataset = dataset

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
        use_threads: bool | None = None,
        cache_metadata: bool = True,
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
        use_threads: bool | None = None,
        cache_metadata: bool = True,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> Iterator[pyarrow.RecordBatch]:
        """See :class:`vortex.dataset.VortexDataset.scanner`"""
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
        )

    @override
    def to_table(
        self,
        columns: list[str] | None = None,
        filter: pyarrow.dataset.Expression | Expr | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pyarrow.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        cache_metadata: bool = True,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """See :class:`vortex.dataset.VortexDataset.scanner`"""
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
        use_threads: bool | None = None,
        cache_metadata: bool = True,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """See :class:`vortex.dataset.VortexDataset.scanner`"""
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
        use_threads: bool | None = None,
        cache_metadata: bool = True,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> pyarrow.Table:
        """See :class:`vortex.dataset.VortexDataset.scanner`"""
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
        use_threads: bool | None = None,
        cache_metadata: bool = True,
        memory_pool: pyarrow.MemoryPool | None = None,
    ) -> int:
        """See :class:`vortex.dataset.VortexDataset.scanner`"""
        return self._dataset.count_rows(
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            cache_metadata=cache_metadata,
            memory_pool=memory_pool,
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
        Keep only rows for which this expression evalutes to ``True``. Any rows for which
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
        Not implemented.
    memory_pool : :class:`.pyarrow.MemoryPool`, optional
        Not implemented.

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
        use_threads: bool | None = None,
        cache_metadata: bool = False,
        memory_pool: pyarrow.MemoryPool | None = None,
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

    @property
    def schema(self):
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
        )
