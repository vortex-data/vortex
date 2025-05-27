import warnings
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset

import vortex as vx
from vortex._lib import dataset as _dataset
from vortex.arrow.expression import arrow_to_vortex as arrow_to_vortex_expr


class VortexDataset(pyarrow.dataset.Dataset):
    """Read Vortex files with row filter and column selection pushdown.

    This class implements the :class:`.pyarrow.dataset.Dataset` interface which enables its use with
    Polars, DuckDB, Pandas and others.

    """

    def __init__(self, dataset):
        self._dataset = dataset

    @staticmethod
    def from_url(url: str):
        return VortexDataset(_dataset.dataset_from_url(url))

    @staticmethod
    def from_path(path: str):
        return vx.open(path).to_dataset()

    @property
    def schema(self) -> pa.Schema:
        return self._dataset.schema()

    def count_rows(
        self,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ) -> int:
        """Not implemented."""
        raise NotImplementedError("count_rows")

    def filter(self, expression: pc.Expression) -> "VortexDataset":
        """Not implemented."""
        raise NotImplementedError("filter")

    def get_fragments(self, filter: pc.Expression | None = None) -> Iterator[pa.dataset.Fragment]:
        """Not implemented."""
        raise NotImplementedError("get_fragments")

    def head(
        self,
        num_rows: int,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ) -> pa.Table:
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
        memory_pool : :class:`.pyarrow.MemoryPool`
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
        if use_threads is True:
            warnings.warn("Vortex does not support threading. Ignoring use_threads=True")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        del memory_pool
        if filter is not None:
            filter = arrow_to_vortex_expr(filter, self.schema)
        return self._dataset.to_array(columns=columns, row_filter=filter).slice(0, num_rows).to_arrow_table()

    def join(
        self,
        right_dataset,
        keys,
        right_keys=None,
        join_type=None,
        left_suffix=None,
        right_suffix=None,
        coalesce_keys=True,
        use_threads: bool | None = None,
    ) -> pa.dataset.InMemoryDataset:
        """Not implemented."""
        raise NotImplementedError("join")

    def join_asof(self, right_dataset, on, by, tolerance, right_on=None, right_by=None) -> pa.dataset.InMemoryDataset:
        """Not implemented."""
        raise NotImplementedError("join_asof")

    def replace_schema(self, schema: pa.Schema):
        """Not implemented."""
        raise NotImplementedError("replace_schema")

    def scanner(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ) -> pa.dataset.Scanner:
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
        memory_pool : :class:`.pyarrow.MemoryPool`
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
            memory_pool,
        )

    def sort_by(self, sorting, **kwargs) -> pa.dataset.InMemoryDataset:
        """Not implemented."""
        raise NotImplementedError("sort_by")

    def take(
        self,
        indices: pa.Array | Any,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ) -> pa.Table:
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
        memory_pool : :class:`.pyarrow.MemoryPool`
            Not implemented.

        Returns
        -------
        table : :class:`.pyarrow.Table`

        """
        return self._dataset.to_array(
            columns=columns, row_filter=filter, indices=vx.array(indices.cast(pa.uint64()))
        ).to_arrow_table()

    def to_record_batch_reader(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ) -> pa.RecordBatchReader:
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
        memory_pool : :class:`.pyarrow.MemoryPool`
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
        if use_threads is True:
            warnings.warn("Vortex does not support threading. Ignoring use_threads=True")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        del memory_pool
        if filter is not None:
            filter = arrow_to_vortex_expr(filter, self.schema)
        return self._dataset.to_record_batch_reader(columns=columns, row_filter=filter)

    def to_batches(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ) -> Iterator[pa.RecordBatch]:
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
        memory_pool : :class:`.pyarrow.MemoryPool`
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
            memory_pool,
        )
        while True:
            try:
                yield record_batch_reader.read_next_batch()
            except StopIteration:
                return

    def to_table(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ) -> pa.Table:
        """Construct an Arrow :class:`.pyarrow.Table`.

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
        memory_pool : :class:`.pyarrow.MemoryPool`
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
        if use_threads is True:
            warnings.warn("Vortex does not support threading. Ignoring use_threads=True")
        if columns is not None and len(columns) == 0:
            raise ValueError("empty projections are not currently supported")
        del memory_pool
        if filter is not None:
            filter = arrow_to_vortex_expr(filter, self.schema)
        return self._dataset.to_array(columns=columns, row_filter=filter).to_arrow_table()


def from_path(path: str) -> VortexDataset:
    return VortexDataset(_dataset.dataset_from_path(path))


def from_url(url: str) -> VortexDataset:
    return VortexDataset(_dataset.dataset_from_url(url))


class VortexScanner(pa.dataset.Scanner):
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
    memory_pool : :class:`.pyarrow.MemoryPool`
        Not implemented.

    Returns
    -------
    table : :class:`.pyarrow.Table`

    """

    def __init__(
        self,
        dataset: VortexDataset,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pa.dataset.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool = None,
    ):
        self._dataset = dataset
        self._columns = columns
        self._filter = filter
        self._batch_size = batch_size
        self._batch_readahead = batch_readahead
        self._fragment_readahead = fragment_readahead
        self._fragment_scan_options = fragment_scan_options
        self._use_threads = use_threads
        self._memory_pool = memory_pool

    @property
    def schema(self):
        return self._datset.schema

    def count_rows(self):
        return self._dataset.count_rows(
            self._filter,
            self._batch_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._fragment_scan_options,
            self._use_threads,
            self._memory_pool,
        )

    def head(self, num_rows: int) -> pa.Table:
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
            self._memory_pool,
        )

    def scan_batches(self) -> Iterator[pa.dataset.TaggedRecordBatch]:
        """Not implemented."""
        raise NotImplementedError("scan batches")

    def to_batches(self) -> Iterator[pa.RecordBatch]:
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
            self._memory_pool,
        )

    def to_reader(self) -> pa.RecordBatchReader:
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
            self._memory_pool,
        )

    def to_table(self) -> pa.Table:
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
            self._memory_pool,
        )
