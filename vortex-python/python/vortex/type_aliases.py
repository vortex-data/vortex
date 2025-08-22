from typing import TypeAlias

import pyarrow as pa

from ._lib.arrays import Array  # pyright: ignore[reportMissingModuleSource]
from ._lib.expr import Expr  # pyright: ignore[reportMissingModuleSource]
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]


IntoProjection: TypeAlias = Expr | list[str] | None
IntoArrayIterator: TypeAlias = Array | ArrayIterator | pa.Table | pa.RecordBatchReader
