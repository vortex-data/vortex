# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar, final
from urllib.parse import quote

from pyarrow.fs import FileSystem, FSSpecHandler, PyFileSystem
from ray.data import Datasource, ReadTask
from ray.data._internal.util import RetryingPyFileSystem
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import BaseFileMetadataProvider, DefaultFileMetadataProvider
from ray.data.datasource.path_util import (
    _resolve_paths_and_filesystem,
)
from typing_extensions import override

from ..dataset import _open_exact_files
from ..expr import Expr as VortexExpr

if TYPE_CHECKING:
    import pandas
    import pyarrow.compute as pc

T = TypeVar("T")


def partition(k: int, ls: list[T]) -> list[list[T]]:
    assert k > 0
    n = len(ls)
    out: list[list[T]] = []
    start = 0

    for i in range(k):
        # (n // k) * k  ===  n + (n % k)
        #
        # We add that extra length to the leading sub-lists.
        part_len = (n // k) + 1 if i < n % k else (n // k)
        out.append(ls[start : start + part_len])
        start += part_len
    return out


def _url_scheme(fs: FileSystem) -> str | None:
    if isinstance(fs, PyFileSystem) and isinstance(fs.handler, FSSpecHandler):
        protocol = getattr(fs.handler.fs, "protocol", "")
        name = protocol if isinstance(protocol, str) else str(protocol[0])
    else:
        name = fs.type_name
    if name in ("file", "local"):
        return None
    return "gs" if name == "gcs" else name


@final
class VortexDatasource(Datasource):
    """Read a folder of Vortex files as a row-wise-partitioned table.

    Each read task reads its files as one :class:`vortex.dataset.VortexDataset`, so Ray reads with
    the same column selection, filter pushdown and cross-file concurrency as DuckDB, Polars and
    PyArrow.
    """

    def __init__(
        self,
        *,
        url: str,
        columns: list[str] | None = None,
        filter: pc.Expression | VortexExpr | None = None,
        batch_size: int | None = None,
        meta_provider: BaseFileMetadataProvider = DefaultFileMetadataProvider(),
    ) -> None:
        super().__init__()
        self._columns = columns
        self._filter = filter

        urls, fs = _resolve_paths_and_filesystem(url, None)
        fs = RetryingPyFileSystem.wrap(fs, retryable_errors=DataContext.get_current().retried_io_errors)

        paths_and_sizes = list(
            meta_provider.expand_paths(
                urls,
                fs,
                None,
                ignore_missing_paths=False,
            )
        )
        paths = [path for path, _ in paths_and_sizes]
        scheme = _url_scheme(fs.unwrap())
        if scheme is not None:
            paths = [path if "://" in path else f"{scheme}://{quote(path)}" for path in paths]
        self._paths: list[str] = paths
        self._batch_size = batch_size

    @override
    def estimate_inmemory_data_size(self) -> int | None:
        """Return an estimate of the in-memory data size, or None if unknown.

        Note that the in-memory data size may be larger than the on-disk data size.
        """
        return None

    @override
    def get_read_tasks(
        self, parallelism: int, per_task_row_limit: int | None = None, data_context: DataContext | None = None
    ) -> list[ReadTask]:
        """Execute the read and return read tasks.

        Args:
            parallelism: The requested read parallelism. The number of read
                tasks should equal to this value if possible.
            per_task_row_limit: The per-task row limit for the read tasks.

        Returns:
            A list of read tasks that can be executed to read blocks from the
            datasource in parallel.
        """
        return [
            _read_task(
                paths,
                self._columns,
                self._filter,
                self._batch_size,
                per_task_row_limit,
            )
            for paths in partition(parallelism, self._paths)
            if len(paths) > 0
        ]

    @property
    @override
    def supports_distributed_reads(self) -> bool:
        """If ``False``, only launch read tasks on the driver's node."""
        return True


def _read_task(
    paths: list[str],
    columns: list[str] | None,
    filter: pc.Expression | VortexExpr | None,
    batch_size: int | None,
    per_task_row_limit: int | None = None,
) -> ReadTask:
    if not paths:
        raise ValueError("no paths specified")

    dataset = _open_exact_files(paths)
    metadata = BlockMetadata(
        num_rows=None if filter is not None else dataset.count_rows(),
        size_bytes=None,
        exec_stats=None,
        input_files=tuple(paths),
    )
    output_schema = dataset.scanner(columns=columns).projected_schema

    def read() -> Iterable[pandas.DataFrame]:
        # A VortexDataset cannot be pickled, so the task opens its files again on the worker.
        reader = _open_exact_files(paths).to_record_batch_reader(
            columns=columns, filter=filter, batch_size=batch_size, limit=per_task_row_limit
        )

        for rb in reader:
            # We would prefer to generate Arrow, but we run into this issue: https://github.com/apache/arrow/issues/47279
            #
            # yield pa.Table.from_batches([rb])
            #
            yield rb.to_pandas()

    return ReadTask(read, metadata, output_schema)
