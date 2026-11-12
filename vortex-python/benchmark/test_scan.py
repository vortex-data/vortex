# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import vortex as vx


@pytest.mark.benchmark(group="scan")
def test_scan(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in vxf.scan()))


@pytest.mark.benchmark(group="scan")
def test_repeated_scan(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    rscan = vxf.to_repeated_scan()
    benchmark(lambda: pa.concat_tables(x.to_arrow_table() for x in rscan.execute()))


@pytest.mark.benchmark(group="scan")
def test_scan_to_arrow_array(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    benchmark(lambda: vxf.scan().to_arrow_array())


@pytest.mark.benchmark(group="scan")
def test_scan_read_all_to_arrow_array(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    benchmark(lambda: vxf.scan().read_all().to_arrow_array())


@pytest.mark.benchmark(group="scan")
def test_repeated_scan_to_arrow_array(benchmark: BenchmarkFixture, vxf: vx.VortexFile) -> None:
    rscan = vxf.to_repeated_scan()
    benchmark(lambda: rscan.execute().to_arrow_array())
