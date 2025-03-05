from collections.abc import Iterator

import pyarrow as pa

from vortex._lib import expr as _expr
from vortex._lib import file as _file

VortexFile = _file.VortexFile


def _to_polars(self: VortexFile):
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
            predicate = _expr._expr_from_polars(predicate)

        for batch in self.to_arrow(
            columns=with_columns,
            expr=predicate,
            batch_size=batch_size,
        ):
            yield pl.DataFrame._from_arrow(batch, rechunk=False)

    return register_io_source(_io_source, schema=schema)


VortexFile.to_polars = _to_polars
