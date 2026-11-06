# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import vortex as vx
from vortex.expr import column


@pytest.fixture
def filter_vxf(request: pytest.FixtureRequest, vxf: vx.VortexFile, codspeed_simulation: bool) -> vx.VortexFile:
    assert isinstance(request.node, pytest.Function)
    if codspeed_simulation and request.node.callspec.params["vxf"] == "int":
        pytest.skip("System calls vary inside these filtered scans under simulation.")
    return vxf


@pytest.mark.benchmark(group="filter")
def test_scan_filter(benchmark: BenchmarkFixture, filter_vxf: vx.VortexFile) -> None:
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in filter_vxf.scan(expr=column("x") >= 50_000)))


@pytest.mark.benchmark(group="filter")
def test_repeated_scan_filter(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    rscan = vxf.to_repeated_scan(expr=column("x") > 50_000)
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in rscan.execute()))


@pytest.mark.benchmark(group="filter")
def test_scan_filter_to_arrow_array(benchmark: BenchmarkFixture, filter_vxf: vx.VortexFile) -> None:
    benchmark(lambda: filter_vxf.scan(expr=column("x") >= 50_000).to_arrow_array())
