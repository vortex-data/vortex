# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Regression test for https://github.com/vortex-data/vortex/issues/7760.

Importing vortex after pyarrow.dataset must not corrupt pyarrow's runtime.
This test runs in a subprocess because import-order bugs and allocator
conflicts only manifest in a fresh process with specific load ordering.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "script",
    [
        # Issue #7760: pyarrow.dataset before vortex
        "import pyarrow.dataset; import vortex; import pyarrow as pa; pa.array([1])",
        # Reverse order should also be fine
        "import vortex; import pyarrow.dataset; import pyarrow as pa; pa.array([1])",
    ],
    ids=["pyarrow_first", "vortex_first"],
)
def test_import_order_no_crash(script):
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Process crashed (exit code {result.returncode}).\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
