#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow

from .arrays import Array
from .expr import Expr
from .store import ObjectStore

class VortexDataset:
    @property
    def concurrency(self) -> int | None: ...
    @property
    def file_count(self) -> int: ...
    def to_array(
        self,
        columns: list[str] | list[int] | None = None,
        row_filter: Expr | None = None,
        indices: Array | None = None,
        row_range: tuple[int, int] | None = None,
        concurrency: int | None = None,
    ) -> Array: ...
    def to_record_batch_reader(
        self,
        columns: list[str] | list[int] | None = None,
        row_filter: Expr | None = None,
        indices: Array | None = None,
        split_by: int | None = None,
        row_range: tuple[int, int] | None = None,
        concurrency: int | None = None,
    ) -> pyarrow.RecordBatchReader: ...
    def count_rows(
        self,
        row_filter: Expr | None = None,
        split_by: int | None = None,
        row_range: tuple[int, int] | None = None,
        concurrency: int | None = None,
    ) -> int: ...
    def schema(self) -> pyarrow.Schema: ...
    def splits(self) -> list[tuple[int, int]]: ...

def dataset_from_url(url: str) -> VortexDataset: ...
def dataset_from_sources(
    sources: list[str],
    *,
    store: ObjectStore | None = None,
    concurrency: int | None = None,
) -> VortexDataset: ...
