#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import final

import pyarrow as pa

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
