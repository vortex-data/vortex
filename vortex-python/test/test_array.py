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


@pytest.mark.parametrize(
    "arrow_type",
    [
        pa.duration("us"),
        pa.month_day_nano_interval(),
        pa.binary(3),
    ],
)
def test_unsupported_arrow_type_raises_value_error(arrow_type: pa.DataType):
    # Regression test for https://github.com/vortex-data/vortex/issues/8346:
    # unsupported Arrow types must surface as a clean ValueError, not a PanicException.
    table = pa.table({"c0": pa.array([], type=arrow_type)})
    with pytest.raises(ValueError):
        _ = vortex.array(table)
