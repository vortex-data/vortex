# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Simple benchmarks
"""

from __future__ import annotations

import io
from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vortex as vx


@pytest.fixture(params=[10, 100])
def vortex_array(request):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    rows: list[dict[str, list[int | None]]] = []
    for _ in range(1_000):
        r: dict[str, list[int | None]] = {}
        for col in range(request.param):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            # Create large arrays of length 100 for each column.
            r[f"col{col}"] = [1, 2, None, 4] * 25
        rows.append(r)
    return vx.array(rows)


@pytest.fixture(params=[10, 100])
def arrow_array(request):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    rows: list[dict[str, list[int | None]]] = []
    for _ in range(1_000):
        r: dict[str, list[int | None]] = {}
        for col in range(request.param):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            # Create large arrays of length 100 for each column.
            r[f"col{col}"] = [1, 2, None, 4] * 25
        rows.append(r)
    return pa.Table.from_pylist(rows)


def test_compress_vortex(
    benchmark: Callable[[Callable[[], None]], None],
    vortex_array: vx.Array,
):
    def compress():
        _ = vx.compress(vortex_array)

    benchmark(compress)


def test_compress_parquet(
    benchmark: Callable[[Callable[[], None]], None], arrow_array: pa.Array[pa.Scalar[pa.DataType]]
):
    def compress():
        # write to bytes in memory.
        bout = io.BytesIO()
        pq.write_table(arrow_array, bout)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

    benchmark(compress)
