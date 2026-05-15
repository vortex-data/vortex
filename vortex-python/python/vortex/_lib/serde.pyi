#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from collections.abc import Sequence
from typing import final

import pyarrow as pa

from .arrays import Array
from .dtype import DType

@final
class SerializedArray:
    @staticmethod
    def parse(data: bytes) -> SerializedArray: ...
    @property
    def metadata(self) -> bytes | None: ...
    @property
    def nbuffers(self) -> int: ...
    @property
    def buffers(self) -> list[pa.Buffer]: ...
    @property
    def nchildren(self) -> int: ...
    @property
    def children(self) -> list[SerializedArray]: ...
    def decode(self, ctx: ArrayContext, dtype: DType, len: int) -> Array: ...

@final
class ArrayContext:
    def __len__(self) -> int: ...

def encode_ipc_array_buffers(
    array: Array,
) -> tuple[list[bytes], list[bytes]]: ...
def decode_ipc_array(array_bytes: bytes, dtype_bytes: bytes) -> Array: ...
def decode_ipc_array_buffers(
    array_buffers: Sequence[bytes | memoryview],
    dtype_buffers: Sequence[bytes | memoryview],
) -> Array: ...
