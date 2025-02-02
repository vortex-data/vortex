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
        ({"a": 0, "b": "foo"}, vx.StructScalar),
        ([0, 1], vx.ListScalar),
    ],
)
def test_round_trip(value, scalar_cls: type[vx.Scalar]):
    scalar = vx.scalar(value)
    assert isinstance(scalar, scalar_cls)
    assert scalar.as_py() == value
