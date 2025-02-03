import pytest

import vortex as vx


@pytest.mark.parametrize(
    "value,scalar_cls",
    [
        (None, vx.NullScalar),
        (True, vx.BoolScalar),
        (False, vx.BoolScalar),
        (0, vx.PrimitiveScalar),
        (-1, vx.PrimitiveScalar),
        (1.0, vx.PrimitiveScalar),
        ("hello", vx.Utf8Scalar),
        (b"hello", vx.BinaryScalar),
        ({}, vx.StructScalar),
        ({"a": 0, "b": "foo"}, vx.StructScalar),
        ([], vx.ListScalar),
        ([0, 1], vx.ListScalar),
    ],
)
def test_round_trip(value, scalar_cls: type[vx.Scalar]):
    scalar = vx.scalar(value)
    assert isinstance(scalar, scalar_cls)
    assert scalar.as_py() == value


def test_f16():
    scalar = vx.scalar(1.0, dtype=vx.float_(16))
    assert scalar.dtype == vx.float_(16)
    assert scalar.as_py() == 1.0
