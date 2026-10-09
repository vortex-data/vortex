#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import IO, final

import polars as pl
import pyarrow as pa
from typing_extensions import override
from vortex.io import ReadAt, ReadBytesAt

from vortex.type_aliases import IntoProjection

from . import CosStore, HfStore
from .arrays import Array
from .dataset import VortexDataset
from .dtype import DType
from .expr import Expr
from .iter import ArrayIterator
from .scan import RepeatedScan
from .store import ObjectStore

@final
class Footer:
    @property
    def row_count(self) -> int: ...

@final
class VortexFile:
    def __len__(self) -> int: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def path(self) -> str: ...
    @property
    def footer(self) -> Footer: ...
    @override
    def __reduce__(self) -> tuple[object, tuple[str, bool]]: ...
    def scan(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
    ) -> ArrayIterator: ...
    def prepare(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
    ) -> RepeatedScan: ...
    def to_arrow(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        limit: int | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
        schema: pa.Schema | None = None,
    ) -> pa.RecordBatchReader: ...
    def to_dataset(self) -> VortexDataset: ...
    def to_polars(self) -> pl.LazyFrame: ...
    def splits(self) -> list[tuple[int, int]]: ...

@final
class SegmentCache:
    def __init__(self, max_bytes: int) -> None: ...
    @property
    def size_bytes(self) -> int: ...
    @property
    def entry_count(self) -> int: ...
    def clear(self) -> None: ...

def open(
    path: str,
    *,
    store: ObjectStore | CosStore | HfStore | None = None,
    footer: Footer | None = None,
    without_segment_cache: bool = False,
    segment_cache: SegmentCache | None = None,
    cache_key: str | None = None,
) -> VortexFile: ...
def open_readable(
    reader: ReadBytesAt | ReadAt | IO[bytes],
    *,
    footer: Footer | None = None,
    concurrency: int | None = None,
    without_segment_cache: bool = False,
    segment_cache: SegmentCache | None = None,
    cache_key: str | None = None,
) -> VortexFile: ...
