from typing import Any, final

import vortex as vx

def scalar(value: Any, *, dtype: vx.DType | None = None) -> Scalar: ...

class Scalar:
    def as_py(self) -> Any: ...

@final
class BoolScalar:
    def as_py(self) -> bool | None: ...

class Buffer:
    def into_python(self, *, recursive=False) -> bytes: ...

class BufferString:
    def into_python(self, *, recursive=False) -> str: ...

class VortexList:
    def into_python(self, *, recursive=False) -> list: ...

class VortexStruct:
    def into_python(self, *, recursive=False) -> dict: ...
