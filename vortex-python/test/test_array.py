# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pytest

import vortex


def test_primitive_array_round_trip(session: vortex.Session):
    a = pa.array([0, 1, 2, 3])
    arr = vortex.array(a)
    assert arr.to_arrow_array(session=session) == a


def test_array_with_nulls(session: vortex.Session):
    a = pa.array([b"123", None], type=pa.string_view())
    arr = vortex.array(a)
    assert arr.to_arrow_array(session=session) == a


def test_varbin_array_round_trip(session: vortex.Session):
    a = pa.array(["a", "b", "c"], type=pa.string_view())
    arr = vortex.array(a)
    assert arr.to_arrow_array(session=session) == a


def test_varbin_array_take(session: vortex.Session):
    a = vortex.array(pa.array(["a", "b", "c", "d"], type=pa.string_view()))
    assert a.take(vortex.array(pa.array([0, 2]))).to_arrow_array(session=session) == pa.array(
        ["a", "c"],
        type=pa.string_view(),
    )


def test_empty_array(session: vortex.Session):
    a = pa.array([], type=pa.uint8())
    primitive = vortex.array(a)
    assert primitive.to_arrow_array(session=session).type == pa.uint8()


@pytest.mark.xfail(raises=IndexError)
def test_scalar_at_out_of_bounds(session: vortex.Session):
    a = vortex.array([10, 42, 999, 1992])
    _s = a.scalar_at(10, session=session)
