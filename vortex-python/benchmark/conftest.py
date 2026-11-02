# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
from pathlib import Path

import pytest

import vortex as vx
from benchmark import data


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    if not config.getoption("codspeed", default=False) and "CODSPEED_ENV" not in os.environ:
        return
    mode = config.getoption("codspeed_mode", default=None)
    if mode is None and "CODSPEED_ENV" in os.environ:
        mode = os.environ.get("CODSPEED_RUNNER_MODE", "simulation")
    if mode != "simulation":
        return

    excluded = {"test_scan_scalar_at[int]", "test_repeated_scan_scalar_at[int]"}
    scalar_at_path = Path(__file__).with_name("test_scalar_at.py")
    for item in items:
        if item.path == scalar_at_path and item.name in excluded:
            item.add_marker(
                pytest.mark.skip(reason="Async reactor polling varies under simulation. Run locally instead.")
            )


@pytest.fixture(scope="session", params=list(data.COLUMN_SETS))
def vxf(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> vx.VortexFile:
    directory = os.environ.get("VORTEX_BENCH_DATA_DIR")
    path = data.write(Path(directory) if directory else tmp_path_factory.getbasetemp() / "data", request.param)

    # Measure scan and decode CPU work without dispatching file reads to IO threads.
    return vx.open(str(path), memory_map=True)
