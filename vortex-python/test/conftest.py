# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import logging
import math
import os
import pathlib
import pytest
import subprocess

import vortex as vx

logging.basicConfig(level=logging.DEBUG)


def pytest_sessionstart():
    """Pytest plugin to trigger maturin builds before running tests."""
    if os.environ.get("CI") is None:
        # Running maturin develop --skip-install builds a "linux" wheel which PyPI rejects
        # (https://peps.python.org/pep-0513/#rationale). When testing an already built wheel, we
        # neither want to rebuild nor pollute the target/wheels directory with a wheel that PyPI
        # will reject.
        working_dir = pathlib.Path(__file__).parent.parent
        _ = subprocess.check_call(["maturin", "develop", "--skip-install"], cwd=working_dir)


def record(x: int, columns: list[str] | set[str] | None = None) -> dict[str, int | str | float]:
    return {
        k: v
        for k, v in {"index": x, "string": str(x), "bool": x % 2 == 0, "float": math.sqrt(x)}.items()
        if columns is None or k in columns
    }


@pytest.fixture(scope="session")
def vxf(tmpdir_factory: pytest.TempPathFactory) -> vx.VortexFile:
    import pyarrow as pa

    fname = tmpdir_factory.mktemp("data") / "foo.vortex"

    if not os.path.exists(fname):
        a = pa.array([record(x) for x in range(1_000_000)])
        arr = vx.compress(vx.array(a))
        vx.io.write(arr, str(fname))
    return vx.open(str(fname), without_segment_cache=True)
