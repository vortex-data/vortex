# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pytest

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
    a = vortex.array(pa.array(["a", "b", "c", "d"], type=pa.string_view()))
    assert a.take(vortex.array(pa.array([0, 2]))).to_arrow_array() == pa.array(
        ["a", "c"],
        type=pa.string_view(),
    )


def test_empty_array():
    a = pa.array([], type=pa.uint8())
    primitive = vortex.array(a)
    assert primitive.to_arrow_array().type == pa.uint8()


@pytest.mark.xfail(raises=IndexError)
def test_scalar_at_out_of_bounds():
    a = vortex.array([10, 42, 999, 1992])
    _s = a.scalar_at(10)


def test_getitem_int():
    a = vortex.array([10, 42, 999, 1992])
    assert a[2].as_py() == 999
    assert a[-1].as_py() == 1992
    with pytest.raises(IndexError):
        a[4]
    with pytest.raises(IndexError):
        a[-5]
    with pytest.raises(TypeError):
        a["nope"]  # pyright: ignore[reportArgumentType, reportCallIssue, reportUnusedExpression]


def test_getitem_slice():
    a = vortex.array([10, 42, 999, 1992])
    assert a[1:3].to_arrow_array() == pa.array([42, 999])
    assert a[:-1].to_arrow_array() == pa.array([10, 42, 999])
    assert len(a[3:1]) == 0
    with pytest.raises(ValueError):
        a[::2]


def test_null_count():
    assert vortex.array([1, None, 3, None]).null_count == 2
    assert vortex.array([1, 2, 3]).null_count == 0


def test_cast():
    a = vortex.array([1, 2, 3]).cast(vortex.float_(64))
    assert a.to_arrow_array() == pa.array([1.0, 2.0, 3.0])


def test_fill_null():
    a = vortex.array([1, None, 3]).fill_null(0)
    assert a.null_count == 0
    assert a.to_arrow_array() == pa.array([1, 0, 3])


def test_is_null():
    a = vortex.array([1, None, 3])
    assert a.is_null().to_arrow_array() == pa.array([False, True, False])
    assert a.is_not_null().to_arrow_array() == pa.array([True, False, True])


def test_arrow_c_array():
    a = vortex.array([1, None, 3])
    assert pa.array(a) == pa.array([1, None, 3])  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_arrow_c_array_chunked():
    chunked = pa.chunked_array([[1, 2], [3]])
    a = vortex.array(chunked)
    assert pa.array(a) == pa.array([1, 2, 3])  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_arrow_c_stream():
    a = vortex.array([1, None, 3])
    assert pa.chunked_array(a) == pa.chunked_array([[1, None, 3]])  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_array_iterator_arrow_c_stream():
    a = vortex.array([{"x": 1}, {"x": 2}])
    iterator = vortex.ArrayIterator.from_iter(a.dtype, iter([a]))
    assert pa.table(iterator) == pa.table({"x": [1, 2]})


def test_decimal_exports():
    assert issubclass(vortex.DecimalArray, vortex.Array)
    assert vortex.DecimalScalar is not None
