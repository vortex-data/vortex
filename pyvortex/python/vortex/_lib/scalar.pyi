from decimal import Decimal
from typing import Any, final

import vortex as vx

def scalar(value: Any, *, dtype: vx.DType | None = None) -> Scalar: ...

class Scalar:
    @property
    def dtype(self) -> vx.DType: ...
    def as_py(self) -> Any: ...

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
    def as_py(self) -> bool | None: ...
    def field(self, name: str) -> Scalar: ...

@final
class ListScalar:
    def as_py(self) -> bool | None: ...
    def element(self, idx: int) -> Scalar: ...

@final
class ExtensionScalar:
    def as_py(self) -> Any: ...
    def storage(self) -> Scalar: ...
