# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import duckdb
import pyarrow as pa
import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

import vortex as vx


@pytest.mark.benchmark(group="scan", disable_gc=True)
def test_scan(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan()))


@pytest.mark.benchmark(group="scan", disable_gc=True)
def test_repeated_scan(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    rscan = vxf.to_repeated_scan()
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in rscan.execute()))


@pytest.mark.benchmark(group="scan", disable_gc=True)
def test_polars(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.collect().to_arrow())


@pytest.mark.benchmark(group="scan", disable_gc=True)
def test_polars_streaming(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    lf = vxf.to_polars()
    benchmark(lambda: lf.collect(engine="streaming").to_arrow())


@pytest.mark.benchmark(group="scan", disable_gc=True)
def test_duckdb(benchmark: BenchmarkFixture, vxf: vx.VortexFile):
    conn = duckdb.connect(database=":memory:")  # pyright: ignore[reportUnknownMemberType]
    ds = vxf.to_dataset()
    _ = conn.register("ds", ds)
    benchmark(lambda: conn.sql("select ds.x from ds").to_arrow_table())
