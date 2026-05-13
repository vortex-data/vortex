# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import duckdb
import pyarrow as pa
import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

import vortex as vx


@pytest.mark.benchmark(group="scalar_at", disable_gc=True)
def test_scan_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan(indices=vx.array([50_000]))))


@pytest.mark.benchmark(group="scalar_at", disable_gc=True)
def test_repeated_scan_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    rscan = vxf.to_repeated_scan()
    benchmark(lambda: rscan.scalar_at(50_000))


@pytest.mark.benchmark(group="scalar_at", disable_gc=True)
def test_polars_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.slice(50_000, 50_001).collect().to_arrow())


@pytest.mark.benchmark(group="scalar_at", disable_gc=True)
def test_polars_streaming_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.slice(50_000, 50_001).collect(engine="streaming").to_arrow())


@pytest.mark.benchmark(group="scalar_at", disable_gc=True)
def test_duckdb_scalar_at(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    conn = duckdb.connect(database=":memory:")
    ds = vxf.to_dataset()
    _ = conn.register("ds", ds)
    benchmark(lambda: conn.sql("select ds.x from ds offset 50000 limit 1").to_arrow_table())
