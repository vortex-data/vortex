# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os.path
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vortex


def test_primitive_compress():
    a = pa.array([0, 0, 0, 0, 9, 9, 9, 9, 1, 5])
    arr_compressed = vortex.compress(vortex.array(a))
    assert not isinstance(arr_compressed, vortex.PrimitiveArray)
    assert arr_compressed.nbytes < a.nbytes


def test_for_compress():
    a = pa.array(np.arange(10_000) + 10_000_000)
    arr_compressed = vortex.compress(vortex.array(a))
    assert not isinstance(arr_compressed, vortex.PrimitiveArray)


def test_arrange_encode():
    a = vortex.array(pa.array(np.arange(10_000), type=pa.uint32()))
    compressed = vortex.compress(a)
    assert isinstance(compressed, vortex.FastLanesDeltaArray | vortex.SequenceArray)
    assert compressed.nbytes < a.nbytes


def test_zigzag_encode():
    a = vortex.array(pa.array([-1, -1, 0, -1, 1, -1]))
    zarr = vortex.ZigZagArray.encode(a)
    assert isinstance(zarr, vortex.ZigZagArray)
    # TODO(ngates): support decoding once we have decompressor.


def test_chunked_encode():
    chunked = pa.chunked_array([pa.array([0, 1, 2]), pa.array([3, 4, 5])])
    encoded = vortex.array(chunked)
    arrow = encoded.to_arrow_array()
    assert isinstance(arrow, pa.ChunkedArray)
    assert arrow.combine_chunks() == pa.array([0, 1, 2, 3, 4, 5])


def test_table_encode():
    table = pa.table(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        {  # pyright: ignore[reportArgumentType]
            "number": pa.chunked_array([pa.array([0, 1, 2]), pa.array([3, 4, 5])]),
            "string": pa.chunked_array(
                [pa.array(["a", "b", "c"], type=pa.string_view()), pa.array(["d", "e", "f"], type=pa.string_view())]
            ),
        }
    )
    assert isinstance(table, pa.Table)

    encoded = vortex.array(table)
    arrow = encoded.to_arrow_array()
    assert isinstance(arrow, pa.ChunkedArray)
    assert arrow.combine_chunks() == pa.StructArray.from_arrays(  # pyright: ignore[reportUnknownMemberType]
        [pa.array([0, 1, 2, 3, 4, 5]), pa.array(["a", "b", "c", "d", "e", "f"], type=pa.string_view())],
        names=["number", "string"],
    )


@pytest.mark.skip(reason="We have no way to guarantee that the bench-vortex data has been downloaded.")
def test_taxi():
    curdir = Path(os.path.dirname(__file__)).parent.parent
    table = pq.read_table(curdir / "bench-vortex/data/yellow-tripdata-2023-11.parquet")  # pyright: ignore[reportUnknownMemberType]
    compressed = vortex.compress(vortex.array(table[:100]))
    decompressed = compressed.to_arrow_array()
    assert len(decompressed) == 100
    # hard to test because of string_view
    # assert pc.equal(decompressed, table[:100].to_struct_array()), (decompressed, table[:100].to_struct_array())
