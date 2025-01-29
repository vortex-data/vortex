"""
Simple benchmarks
"""

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vortex as vx


@pytest.fixture(params=[10, 100])
def vortex_array(request):
    rows = []
    for row in range(1_000):
        r = {}
        for col in range(request.param):
            # Create large arrays of length 100 for each column.
            r[f"col{col}"] = [1, 2, None, 4] * 25
        rows.append(r)
    return vx.array(rows)


@pytest.fixture(params=[10, 100])
def arrow_array(request):
    rows = []
    for row in range(1_000):
        r = {}
        for col in range(request.param):
            # Create large arrays of length 100 for each column.
            r[f"col{col}"] = [1, 2, None, 4] * 25
        rows.append(r)
    return pa.Table.from_pylist(rows)


def test_compress_vortex(benchmark, vortex_array):
    def compress():
        vx.compress(vortex_array)

    benchmark(compress)


def test_compress_parquet(benchmark, arrow_array):
    def compress():
        # write to bytes in memory.
        bout = io.BytesIO()
        pq.write_table(arrow_array, bout)

    benchmark(compress)
