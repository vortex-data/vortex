from collections.abc import Iterator
import pyarrow as pa
from vortex._lib import file as _file

VortexFile = _file.VortexFile


def _VortexFile_to_polars(self: VortexFile):
    """Read the Vortex file as a pl.LazyFrame, supporting column pruning and predicate pushdown."""
    import polars as pl
    from polars.io.plugins import register_io_source

    schema: pa.Schema = self.dtype.to_arrow_schema()

    def _io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        if predicate is not None:
            predicate

        self.dtype.

    schema = pl.Schema(self.dtype.to_arrow_schema())

    register_io_source(
        _io_source,
    )


VortexFile.to_polars = _VortexFile_to_polars
