from collections.abc import Iterator

from vortex._lib import file as _file

VortexFile = _file.VortexFile


def _VortexFile_to_polars(self: VortexFile):
    """Read the Vortex file as a pl.LazyFrame, supporting column pruning and predicate pushdown."""
    import polars as pl
    from polars.io.plugins import register_io_source

    def _io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        pass

    schema = pl.Schema(self.dtype.to_arrow())

    register_io_source(
        _io_source,
    )


VortexFile.to_polars = _VortexFile_to_polars
