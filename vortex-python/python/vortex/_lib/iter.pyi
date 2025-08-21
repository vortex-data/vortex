#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from collections.abc import Iterator
from typing import final

import pyarrow as pa

from .dtype import DType
from .arrays import Array

@final
class ArrayIterator:
    @property
    def dtype(self) -> DType: ...
    def read_all(self) -> Array: ...
    def __iter__(self) -> ArrayIterator: ...
    def __next__(self) -> Array: ...
    def to_arrow(self) -> pa.RecordBatchReader: ...
    @staticmethod
    def from_iter(dtype: DType, iter: Iterator[Array]) -> ArrayIterator: ...
