# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The Vortex runtime must be rebuilt in forked children, or every read there hangs.

Every scenario in `child_scenarios.py` runs against each `multiprocessing` start method available
on the platform. `fork` is the hazard being guarded against; `spawn` and `forkserver` start their
children from a fresh interpreter, so they mainly prove that the pickling those methods require
(Expr, Array, VortexFile) still works and that a child built from scratch reads just as well.

Each combination runs in a **fresh interpreter** that imports nothing but Vortex, and only then
creates children. That keeps the tests measuring Vortex's own behaviour: several third-party
libraries (Ray, DuckDB, CUDA drivers) install their own `atfork` handlers or hold locks that make
`fork` unsafe process-wide, so forking straight from the pytest process would be testing them as
much as us.

A regression shows up as a timeout, since the failure mode being guarded against is a hang.
"""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pytest

import vortex as vx

from .child_scenarios import ROWS, SCENARIOS

TIMEOUT_SECONDS = 120
SCRIPT = Path(__file__).with_name("child_scenarios.py")
AVAILABLE_START_METHODS = frozenset(multiprocessing.get_all_start_methods())


SCENARIO_CASES: list[tuple[str, str]] = [
    (name, start_method) for name in sorted(SCENARIOS) for start_method in SCENARIOS[name].start_methods
]


@pytest.fixture(scope="module")
def people(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("fork") / "people.vortex"
    vx.io.write(vx.array(pa.array([{"name": f"person-{i}", "age": i} for i in range(ROWS)])), str(path))
    return path


@pytest.mark.parametrize(("name", "start_method"), SCENARIO_CASES)
def test_child_process_scenario(name: str, start_method: str, people: Path) -> None:
    if start_method not in AVAILABLE_START_METHODS:
        pytest.skip(f"the {start_method!r} start method is not available on {sys.platform}")

    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), name, start_method, str(people)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"scenario {name!r} ({start_method}) did not finish within {TIMEOUT_SECONDS}s (likely a deadlock)")

    if proc.returncode != 0 or not proc.stdout.strip().endswith("OK"):
        output = f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        pytest.fail(f"scenario {name!r} ({start_method}) failed (exit {proc.returncode})\n{output}")
