# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar, final

from ray.data import Datasource, ReadTask
from ray.data.block import BlockMetadata
from ray.data.datasource import BaseFileMetadataProvider, DefaultFileMetadataProvider
from ray.data.datasource.path_util import (
    _resolve_paths_and_filesystem,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]
)
from typing_extensions import override

from .. import open as vx_open
from ..arrow.expression import ensure_vortex_expression
from ..expr import Expr as VortexExpr
from ..type_aliases import IntoProjection

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


@final
class VortexDatasource(Datasource):
    """Read a folder of Vortex files as a row-wise-partitioned table."""

    def __init__(
        self,
        *,
        url: str,
        columns: IntoProjection = None,
        filter: pc.Expression | VortexExpr | None = None,
        batch_size: int | None = None,
        meta_provider: BaseFileMetadataProvider = DefaultFileMetadataProvider(),  # pyright: ignore[reportCallInDefaultInitializer]
    ):
        self._columns = columns
        self._filter = filter

        urls, fs = _resolve_paths_and_filesystem(url, None)  # pyright: ignore[reportUnknownVariableType]
        paths_and_sizes = list(
            meta_provider.expand_paths(
                urls,
                fs,  # pyright: ignore[reportUnknownArgumentType]
                None,
                ignore_missing_paths=False,
            )
        )
        self._paths: list[str] = [path for path, _ in paths_and_sizes]
        self._batch_size = batch_size

    @override
    def estimate_inmemory_data_size(self) -> int | None:
        """Return an estimate of the in-memory data size, or None if unknown.

        Note that the in-memory data size may be larger than the on-disk data size.
        """
        return None

    @override
    def get_read_tasks(self, parallelism: int) -> list[ReadTask]:
        """Execute the read and return read tasks.

        Args:
            parallelism: The requested read parallelism. The number of read
                tasks should equal to this value if possible.

        Returns:
            A list of read tasks that can be executed to read blocks from the
            datasource in parallel.
        """
        return [
            _read_task(paths, self._columns, self._filter, self._batch_size)
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
    columns: IntoProjection,
    filter: pc.Expression | VortexExpr | None,
    batch_size: int | None,
) -> ReadTask:
    if not paths:
        raise ValueError("no paths specified")

    files = [vx_open(path) for path in paths]
    schemas = [f.dtype.to_arrow_schema() for f in files]
    schema = schemas[0]
    assert all(s == schema for s in schemas[1:])

    num_rows = sum(len(f) for f in files)

    metadata = BlockMetadata(
        num_rows=num_rows,
        size_bytes=None,
        exec_stats=None,
        input_files=paths,
    )

    def read() -> Iterable[pandas.DataFrame]:
        # If we could serialize a PyVortexFile and a PyExpr, we could set those up earlier.

        vx_filter = ensure_vortex_expression(filter, schema=schema)
        for path in paths:
            f = vx_open(path)
            for rb in f.to_arrow(columns, expr=vx_filter, batch_size=batch_size):
                # We would prefer to generate Arrow, but we run into this issue: https://github.com/apache/arrow/issues/47279
                #
                # yield pa.Table.from_batches([rb])
                #
                yield rb.to_pandas()  # pyright: ignore[reportUnknownMemberType]

    return ReadTask(read, metadata, schema)
