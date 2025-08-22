#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import final

import polars as pl
import pyarrow as pa

from vortex.type_aliases import IntoProjection

from .arrays import Array
from .dataset import VortexDataset
from .dtype import DType
from .expr import Expr
from .iter import ArrayIterator

@final
class VortexFile:
    def __len__(self) -> int: ...
    @property
    def dtype(self) -> DType: ...
    def scan(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        indices: Array | None = None,
        batch_size: int | None = None,
    ) -> ArrayIterator: ...
    def to_arrow(
        self,
        projection: IntoProjection = None,
        *,
        expr: Expr | None = None,
        batch_size: int | None = None,
    ) -> pa.RecordBatchReader: ...
    def to_dataset(self) -> VortexDataset: ...
    def to_polars(self) -> pl.LazyFrame: ...

def open(path: str) -> VortexFile: ...
