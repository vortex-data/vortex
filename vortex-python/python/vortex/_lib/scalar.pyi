#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TypeAlias, final
from uuid import UUID

from .dtype import DType

ScalarPyType: TypeAlias = (
    None
    | int
    | float
    | str
    | Decimal
    | bytes
    | date
    | time
    | datetime
    | timedelta
    | UUID
    | list[ScalarPyType]
    | dict[str, ScalarPyType]
)

def scalar(value: object, *, dtype: DType | None = None) -> Scalar: ...

class Scalar:
    @property
    def dtype(self) -> DType: ...
    def as_py(self) -> ScalarPyType: ...

@final
class NullScalar:
    def as_py(self) -> None: ...

@final
class BoolScalar:
    def as_py(self) -> bool | None: ...

@final
class PrimitiveScalar:
    def as_py(self) -> int | float | None: ...

@final
class DecimalScalar:
    def as_py(self) -> Decimal | None: ...

@final
class Utf8Scalar:
    def as_py(self) -> str | None: ...

@final
class BinaryScalar:
    def as_py(self) -> bytes | None: ...

@final
class StructScalar:
    def as_py(self) -> dict[str, ScalarPyType] | None: ...
    def field(self, name: str) -> Scalar: ...

@final
class ListScalar:
    def as_py(self) -> list[ScalarPyType] | None: ...
    def element(self, idx: int) -> Scalar: ...

@final
class ExtensionScalar:
    def as_py(self) -> ScalarPyType: ...
    def storage(self) -> ScalarPyType: ...
