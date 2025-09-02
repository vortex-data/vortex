import duckdb
import os
import polars as pl
import pyarrow as pa
import pytest
import vortex as vx
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]
from vortex.expr import column


@pytest.fixture(scope="session")
def vxf(tmpdir_factory) -> vx.VortexFile:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    if not os.path.exists(fname):  # pyright: ignore[reportUnknownArgumentType]
        a = vx.array(pa.table({"x": list(range(1_000_000))}))
        vx.io.write(a, str(fname))  # pyright: ignore[reportUnknownArgumentType]
    return vx.open(str(fname))  # pyright: ignore[reportUnknownArgumentType]


def test_scan(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan()))


def test_scan_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan(indices=vx.array([50_000]))))


def test_scan_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan(expr=column("x") >= 500_000)))


def test_repeated_scan(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    rscan = vxf.to_repeated_scan()
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in rscan.execute()))


def test_repeated_scan_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    rscan = vxf.to_repeated_scan()
    benchmark(lambda: rscan.scalar_at(50_000))


def test_repeated_scan_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    rscan = vxf.to_repeated_scan(expr=column("x") > 500_000)
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in rscan.execute()))


def test_polars(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.collect().to_arrow())


def test_polars_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.slice(50_000, 50_001).collect().to_arrow())


def test_polars_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.filter(pl.col("x") >= pl.lit(500000).cast(pl.Int64)).collect().to_arrow())


def test_polars_streaming(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.collect(engine="streaming").to_arrow())


def test_polars_streaming_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.slice(50_000, 50_001).collect(engine="streaming").to_arrow())


def test_polars_streaming_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.filter(pl.col("x") >= pl.lit(500000).cast(pl.Int64)).collect(engine="streaming").to_arrow())


def test_duckdb(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    conn = duckdb.connect(database=":memory:")  # pyright: ignore[reportUnknownMemberType]
    ds = vxf.to_dataset()
    _ = conn.register("ds", ds)
    benchmark(lambda: conn.sql("select ds.x from ds").to_arrow_table())


def test_duckdb_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    conn = duckdb.connect(database=":memory:")  # pyright: ignore[reportUnknownMemberType]
    ds = vxf.to_dataset()
    _ = conn.register("ds", ds)
    benchmark(lambda: conn.sql("select ds.x from ds offset 50000 limit 1").to_arrow_table())


def test_duckdb_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    conn = duckdb.connect(database=":memory:")  # pyright: ignore[reportUnknownMemberType]
    ds = vxf.to_dataset()
    _ = conn.register("ds", ds)
    benchmark(lambda: conn.sql("select ds.x from ds where x >= 500000").to_arrow_table())
