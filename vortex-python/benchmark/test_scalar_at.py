# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os

import pyarrow as pa
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import vortex as vx


@pytest.fixture
def scalar_vxf(request: pytest.FixtureRequest, vxf: vx.VortexFile) -> vx.VortexFile:
    enabled = request.config.getoption("codspeed", default=False) or "CODSPEED_ENV" in os.environ
    mode = request.config.getoption("codspeed_mode", default=None)
    if mode is None and "CODSPEED_ENV" in os.environ:
        mode = os.environ.get("CODSPEED_RUNNER_MODE", "simulation")
    assert isinstance(request.node, pytest.Function)
    if enabled and mode == "simulation" and request.node.callspec.params["vxf"] == "int":
        pytest.skip("Reactor polling varies inside these scalar scans under simulation.")
    return vxf


@pytest.mark.benchmark(group="scalar_at")
def test_scan_scalar_at(benchmark: BenchmarkFixture, scalar_vxf: vx.VortexFile) -> None:
    indices = vx.array([50_000])
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in scalar_vxf.scan(indices=indices)))


@pytest.mark.benchmark(group="scalar_at")
def test_repeated_scan_scalar_at(benchmark: BenchmarkFixture, scalar_vxf: vx.VortexFile) -> None:
    rscan = scalar_vxf.to_repeated_scan()
    benchmark(lambda: rscan.scalar_at(50_000))
