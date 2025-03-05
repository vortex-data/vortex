import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds

import vortex as vx
import vortex.expr as ve

class VortexFile:
    def __len__(self) -> int: ...
    @property
    def dtype(self) -> vx.DType: ...
    def to_arrow(
        self,
        columns: list[str] | None = None,
        *,
        expr: ve.Expr | None = None,
        batch_size: int | None = None,
    ) -> pa.RecordBatchReader: ...
    def to_dataset(self) -> pds.Dataset: ...
    def to_polars(self) -> pl.LazyFrame: ...

def open(path: str) -> vx.VortexFile: ...
