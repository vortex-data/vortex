import pyarrow as pa

import vortex


def test_primitive_array_round_trip():
    a = pa.array([0, 1, 2, 3])
    arr = vortex.array(a)
    assert arr.to_arrow_array() == a


def test_array_with_nulls():
    a = pa.array([b"123", None], type=pa.string_view())
    arr = vortex.array(a)
    assert arr.to_arrow_array() == a


def test_varbin_array_round_trip():
    a = pa.array(["a", "b", "c"], type=pa.string_view())
    arr = vortex.array(a)
    assert arr.to_arrow_array() == a


def test_varbin_array_take():
    a = vortex.array(pa.array(["a", "b", "c", "d"]))
    assert a.take(vortex.array(pa.array([0, 2]))).to_arrow_array() == pa.array(
        ["a", "c"],
        type=pa.string_view(),
    )


def test_empty_array():
    a = pa.array([], type=pa.uint8())
    primitive = vortex.array(a)
    assert primitive.to_arrow_array().type == pa.uint8()
