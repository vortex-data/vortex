# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark input files.

Run ``python -m benchmark.data <dir>`` to write the files before a measured run, then set
``VORTEX_BENCH_DATA_DIR=<dir>``. Under CodSpeed simulation all of the process runs on Valgrind,
so data written ahead of time keeps the slow setup out of the run.
"""

import random
import sys
from array import array
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

import vortex as vx

LENGTH = 100_000

COLUMN_SETS: dict[str, tuple[str, ...]] = {
    "int": ("x",),
    "int_str": ("x", "y"),
    "int_float": ("x", "z"),
    "int_str_float": ("x", "y", "z"),
}


def _hex_strings(length: int) -> pa.StringArray:
    # 32 random hex characters for each row, built from buffers without a Python loop per row.
    width = 32
    chars = random.Random(0).randbytes(length * width // 2).hex().encode()
    offsets = array("i", range(0, length * width + 1, width))

    return pa.StringArray.from_buffers(length, pa.py_buffer(offsets), pa.py_buffer(chars))


def table(columns: tuple[str, ...], length: int = LENGTH) -> pa.Table:
    x = pa.array(range(length), type=pa.int64())
    arrays: list[pa.Array[Any] | pa.ChunkedArray[Any]] = [x]

    if "y" in columns:
        arrays.append(_hex_strings(length))
    if "z" in columns:
        arrays.append(pc.sqrt(x))

    return pa.table(arrays, names=list(columns))


def write(directory: Path, name: str) -> Path:
    """Write the file for ``name`` into ``directory``, unless it is already there."""
    path = directory / f"{name}.vortex"

    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        vx.io.write(table(COLUMN_SETS[name]), str(path))

    return path


if __name__ == "__main__":
    for name in COLUMN_SETS:
        print(write(Path(sys.argv[1]), name))
