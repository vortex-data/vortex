# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import duckdb
import polars as pl
import pyarrow as pa
import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

import vortex as vx
from vortex.expr import column


@pytest.mark.benchmark(group="filter", disable_gc=True)
def test_scan_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan(expr=column("x") >= 50_000)))


@pytest.mark.benchmark(group="filter", disable_gc=True)
def test_repeated_scan_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    rscan = vxf.to_repeated_scan(expr=column("x") > 50_000)
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in rscan.execute()))


@pytest.mark.benchmark(group="filter", disable_gc=True)
def test_polars_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.filter(pl.col("x") >= pl.lit(50_000).cast(pl.Int64)).collect().to_arrow())


@pytest.mark.benchmark(group="filter", disable_gc=True)
def test_polars_streaming_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.filter(pl.col("x") >= pl.lit(50_000).cast(pl.Int64)).collect(engine="streaming").to_arrow())


@pytest.mark.benchmark(group="filter", disable_gc=True)
def test_duckdb_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    conn = duckdb.connect(database=":memory:")
    ds = vxf.to_dataset()
    _ = conn.register("ds", ds)
    benchmark(lambda: conn.sql("select ds.x from ds where x >= 50000").to_arrow_table())
