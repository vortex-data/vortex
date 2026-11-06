# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
from pathlib import Path

import pytest
from pytest_codspeed.instruments import MeasurementMode
from pytest_codspeed.plugin import get_plugin

import vortex as vx
from benchmark import data


@pytest.fixture
def codspeed_simulation(pytestconfig: pytest.Config) -> bool:
    # Use the resolved mode, including the runner's legacy "instrumentation" alias.
    plugin = get_plugin(pytestconfig)
    return plugin is not None and plugin.is_codspeed_enabled and plugin.mode == MeasurementMode.Simulation


@pytest.fixture(scope="session", params=list(data.COLUMN_SETS))
def vxf(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> vx.VortexFile:
    directory = os.environ.get("VORTEX_BENCH_DATA_DIR")
    path = data.write(Path(directory) if directory else tmp_path_factory.getbasetemp() / "data", request.param)

    # Measure scan and decode CPU work without dispatching file reads to IO threads.
    return vx.open(str(path), memory_map=True)
