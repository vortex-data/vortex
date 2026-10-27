# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import vortex as vx
from vortex.expr import column


@pytest.mark.benchmark(group="filter")
def test_scan_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan(expr=column("x") >= 50_000)))


@pytest.mark.benchmark(group="filter")
def test_repeated_scan_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    rscan = vxf.to_repeated_scan(expr=column("x") > 50_000)
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in rscan.execute()))


@pytest.mark.benchmark(group="filter")
def test_scan_filter_to_arrow_array(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    benchmark(lambda: vxf.scan(expr=column("x") >= 50_000).to_arrow_array())
