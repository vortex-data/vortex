from typing import Any, final

import pyarrow as pa

import vortex as vx

def _encode(obj: Any) -> pa.Array: ...
@final
class BoolArray(vx.Array):
    def __new__(cls, array: vx.Array) -> BoolArray: ...
    def true_count(self) -> int: ...
