#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow

from .arrays import Array
from .expr import Expr

class VortexDataset:
    def to_array(
        self, columns: list[str] | list[int] | None = None, row_filter: Expr | None = None, indices: Array | None = None
    ) -> Array: ...
    def to_record_batch_reader(
        self,
        columns: list[str] | list[int] | None = None,
        row_filter: Expr | None = None,
        indices: Array | None = None,
        split_by: int | None = None,
    ) -> pyarrow.RecordBatchReader: ...
    def count_rows(self, row_filter: Expr | None = None, split_by: int | None = None) -> int: ...
    def schema(self) -> pyarrow.Schema: ...

def dataset_from_url(url: str) -> VortexDataset: ...
