#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from collections.abc import Sequence
from typing import final

import pyarrow as pa

from .arrays import Array
from .dtype import DType

@final
class ArrayParts:
    @staticmethod
    def parse(data: bytes) -> ArrayParts: ...
    @property
    def metadata(self) -> bytes | None: ...
    @property
    def nbuffers(self) -> int: ...
    @property
    def buffers(self) -> list[pa.Buffer]: ...
    @property
    def nchildren(self) -> int: ...
    @property
    def children(self) -> list[ArrayParts]: ...
    def decode(self, ctx: ArrayContext, dtype: DType, len: int) -> pa.Array[pa.Scalar[pa.DataType]]: ...

@final
class ArrayContext:
    def __len__(self) -> int: ...

def decode_ipc_array(array_bytes: bytes, dtype_bytes: bytes) -> Array: ...
def decode_ipc_array_buffers(
    array_buffers: Sequence[bytes | memoryview], dtype_buffers: Sequence[bytes | memoryview]
) -> Array: ...
