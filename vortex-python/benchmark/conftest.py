# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
from pathlib import Path

import pytest

import vortex as vx
from benchmark import data


@pytest.fixture(scope="session", params=list(data.COLUMN_SETS))
def vxf(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> vx.VortexFile:
    directory = os.environ.get("VORTEX_BENCH_DATA_DIR")
    path = data.write(Path(directory) if directory else tmp_path_factory.getbasetemp() / "data", request.param)

    # Measure scan and decode CPU work without dispatching file reads to IO threads.
    return vx.open(str(path), memory_map=True)
