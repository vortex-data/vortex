# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import decimal

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
def test_round_trip(
    value: bool | int | float | bytes | str | list[int] | dict[str, str] | None, scalar_cls: type[vx.Scalar]
) -> None:
    scalar = vx.scalar(value)
    assert isinstance(scalar, scalar_cls)
    assert scalar.as_py() == value


def test_f16() -> None:
    scalar = vx.scalar(1.0, dtype=vx.float_(16))
    assert scalar.dtype == vx.float_(16)
    assert scalar.as_py() == 1.0


@pytest.mark.parametrize(
    "precision,scale,stored,expected",
    [
        (10, 2, 12345, "123.45"),
        # A negative stored value used to render as "-123.-45", which Decimal refuses.
        (10, 2, -12345, "-123.45"),
        # Truncating division put the whole part at 0 and left the sign on the fraction.
        (10, 2, -5, "-0.05"),
        (10, 2, 5, "0.05"),
        # The stored value picks the narrowest storage that holds it, so a small value at an
        # everyday scale reached `10i8.pow(3)`, which overflows an i8.
        (10, 3, 5, "0.005"),
        (10, 5, 5, "0.00005"),
        (10, 3, -5, "-0.005"),
        # Scale 0 has no fractional digits, so the exponent has to be 0 and not -1.
        (10, 0, -7, "-7"),
        # A negative scale means trailing zeros before the point, so the exponent is positive.
        (5, -5, 1, "1E+5"),
        (5, -5, -1, "-1E+5"),
        (1, -128, 1, "1E+128"),
        # Above scale 38 no storage width holds the factor, i128 included.
        (76, 39, 1, "1E-39"),
        (76, 76, -1, "-1E-76"),
    ],
)
def test_decimal_round_trip(precision: int, scale: int, stored: int, expected: str) -> None:
    scalar = vx.scalar(stored, dtype=vx.decimal(precision=precision, scale=scale))

    value = scalar.as_py()
    assert isinstance(value, decimal.Decimal)
    # Compare the string form, not just the numeric value: Decimal("-7") == Decimal("-7.0"), so an
    # equality check alone would not pin the exponent to the dtype's scale.
    assert str(value) == expected
    assert value.as_tuple().exponent == -scale


def test_decimal_ignores_context_precision() -> None:
    """A wide decimal must survive conversion whatever the ambient context precision is."""
    digits = "9" * 38
    with decimal.localcontext() as ctx:
        ctx.prec = 3
        scalar = vx.scalar(int(digits), dtype=vx.decimal(precision=38, scale=19))
        value = scalar.as_py()
    assert isinstance(value, decimal.Decimal)
    assert len(value.as_tuple().digits) == 38
    assert str(value) == f"{digits[:19]}.{digits[19:]}"
