from collections.abc import Iterator
from typing import TypeAlias

import pyarrow as pa

import vortex as vx
import vortex.expr as ve
from vortex._lib import file as _file

VortexFile = _file.VortexFile
IntoProjection: TypeAlias = ve.Expr | list[str] | None
IntoArrayIterator: TypeAlias = vx.Array | vx.ArrayIterator


def _to_polars(self: VortexFile):
    """Read the Vortex file as a pl.LazyFrame, supporting column pruning and predicate pushdown."""
    import polars as pl
    from polars.io.plugins import register_io_source

    from vortex.polars_ import polars_to_vortex

    schema: pa.Schema = self.dtype.to_arrow_schema()

    def _io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        if predicate is not None:
            predicate = polars_to_vortex(predicate)

        reader = self.to_arrow(projection=with_columns, expr=predicate)

        for batch in reader:
            batch = pl.DataFrame._from_arrow(batch, rechunk=False)
            # TODO(ngates): set sortedness on DataFrame based on stats?
            yield batch

        # Make sure we always yield at least one empty DataFrame
        yield pl.DataFrame._from_arrow(
            data=pa.RecordBatch.from_arrays(
                [pa.array([], type=field.type) for field in reader.schema],
                schema=reader.schema,
            ),
        )

    return register_io_source(_io_source, schema=schema)


VortexFile.to_polars = _to_polars
