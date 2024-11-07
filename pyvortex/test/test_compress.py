import os.path
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vortex


@pytest.mark.xfail(reason="Not yet implemented")
def test_primitive_compress():
    a = pa.array([0, 0, 0, 0, 9, 9, 9, 9, 1, 5])
    arr_compressed = vortex.compress(vortex.array(a))
    assert not isinstance(arr_compressed, vortex.encoding.PrimitiveArray)
    assert arr_compressed.nbytes < a.nbytes


@pytest.mark.xfail(reason="Not yet implemented")
def test_for_compress():
    a = pa.array(np.arange(10_000) + 10_000_000)
    arr_compressed = vortex.compress(vortex.array(a))
    assert not isinstance(arr_compressed, vortex.encoding.PrimitiveArray)


@pytest.mark.xfail(reason="Not yet implemented")
def test_bool_compress():
    a = vortex.array(pa.array([False] * 10_000 + [True] * 10_000))
    arr_compressed = vortex.compress(a)
    assert len(arr_compressed) == 20_000
    assert isinstance(arr_compressed, vortex.encoding.RoaringBoolArray)
    assert arr_compressed.nbytes < a.nbytes


@pytest.mark.xfail(reason="Not yet implemented")
def test_roaring_bool_encode():
    a = vortex.array(pa.array([True] * 10_000))
    rarr = vortex.encoding.RoaringBoolArray.encode(a)
    assert isinstance(rarr, vortex.encoding.RoaringBoolArray)
    assert rarr.nbytes < a.nbytes


@pytest.mark.xfail(reason="Not yet implemented")
def test_arange_encode():
    a = vortex.array(pa.array(np.arange(10_000), type=pa.uint32()))
    compressed = vortex.compress(a)
    assert isinstance(compressed, vortex.encoding.DeltaArray) or isinstance(compressed, vortex.encoding.RoaringIntArray)
    assert compressed.nbytes < a.nbytes


@pytest.mark.xfail(reason="Not yet implemented")
def test_zigzag_encode():
    a = vortex.array(pa.array([-1, -1, 0, -1, 1, -1]))
    zarr = vortex.encoding.ZigZagArray.encode(a)
    assert isinstance(zarr, vortex.encoding.ZigZagArray)
    # TODO(ngates): support decoding once we have decompressor.


def test_chunked_encode():
    chunked = pa.chunked_array([pa.array([0, 1, 2]), pa.array([3, 4, 5])])
    encoded = vortex.array(chunked)
    assert encoded.to_arrow_array().combine_chunks() == pa.array([0, 1, 2, 3, 4, 5])


def test_table_encode():
    table = pa.table(
        {
            "number": pa.chunked_array([pa.array([0, 1, 2]), pa.array([3, 4, 5])]),
            "string": pa.chunked_array(
                [pa.array(["a", "b", "c"], type=pa.string_view()), pa.array(["d", "e", "f"], type=pa.string_view())]
            ),
        }
    )
    encoded = vortex.array(table)
    assert encoded.to_arrow_array().combine_chunks() == pa.StructArray.from_arrays(
        [pa.array([0, 1, 2, 3, 4, 5]), pa.array(["a", "b", "c", "d", "e", "f"], type=pa.string_view())],
        names=["number", "string"],
    )


@pytest.mark.xfail(reason="Not yet implemented")
def test_taxi():
    curdir = Path(os.path.dirname(__file__)).parent.parent
    table = pq.read_table(curdir / "bench-vortex/data/yellow-tripdata-2023-11.parquet")
    compressed = vortex.compress(vortex.array(table[:100]))
    decompressed = compressed.to_arrow_array()
    assert not decompressed
