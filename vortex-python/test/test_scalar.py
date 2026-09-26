# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import datetime
import decimal
import uuid
import zoneinfo
from typing import Literal

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


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("ns", 3_723_500_000_000),
        ("us", 3_723_500_000),
        ("ms", 3_723_500),
    ],
)
def test_time_scalar(unit: Literal["ns", "us", "ms"], expected: int) -> None:
    scalar = vx.scalar(datetime.time(1, 2, 3, 500_000), dtype=vx.time(unit))
    assert isinstance(scalar, vx.ExtensionScalar)
    assert scalar.dtype == vx.time(unit)
    assert scalar.as_py() == expected


def test_time_scalar_defaults_to_microseconds() -> None:
    scalar = vx.scalar(datetime.time(0, 0, 1, 7))
    assert scalar.dtype == vx.time("us")
    assert scalar.as_py() == 1_000_007


def test_time_scalar_rejects_precision_loss() -> None:
    with pytest.raises(ValueError, match="without losing precision"):
        _ = vx.scalar(datetime.time(0, 0, 1, 500_000), dtype=vx.time("s"))


def test_time_scalar_rejects_timezone() -> None:
    with pytest.raises(ValueError, match="Timezone-aware"):
        _ = vx.scalar(datetime.time(12, tzinfo=datetime.UTC))


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("days", 1),
        ("ms", 86_400_000),
    ],
)
def test_date_scalar(unit: Literal["days", "ms"], expected: int) -> None:
    scalar = vx.scalar(datetime.date(1970, 1, 2), dtype=vx.date(unit))
    assert isinstance(scalar, vx.ExtensionScalar)
    assert scalar.dtype == vx.date(unit)
    assert scalar.as_py() == expected


def test_date_scalar_defaults_to_days() -> None:
    scalar = vx.scalar(datetime.date(1969, 12, 31))
    assert scalar.dtype == vx.date("days")
    assert scalar.as_py() == -1


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("s", 86_401),
        ("ms", 86_401_000),
        ("us", 86_401_000_000),
        ("ns", 86_401_000_000_000),
    ],
)
def test_timestamp_scalar(unit: Literal["s", "ms", "us", "ns"], expected: int) -> None:
    scalar = vx.scalar(datetime.datetime(1970, 1, 2, 0, 0, 1), dtype=vx.timestamp(unit))
    assert scalar.dtype == vx.timestamp(unit)
    assert scalar.as_py() == expected


def test_timestamp_scalar_defaults_to_microseconds() -> None:
    scalar = vx.scalar(datetime.datetime(1970, 1, 1, 0, 0, 0, 5))
    assert scalar.dtype == vx.timestamp("us")
    assert scalar.as_py() == 5


def test_timestamp_scalar_stores_aware_values_as_utc() -> None:
    try:
        zone = zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:
        pytest.skip("tz database unavailable")
    # 1970-01-01 is EST, five hours behind UTC.
    scalar = vx.scalar(datetime.datetime(1970, 1, 1, tzinfo=zone))
    assert scalar.dtype == vx.timestamp("us", tz="America/New_York")
    assert scalar.as_py() == 5 * 3_600_000_000


def test_timestamp_scalar_utc() -> None:
    scalar = vx.scalar(datetime.datetime(1970, 1, 1, 0, 0, 1, tzinfo=datetime.UTC))
    assert scalar.dtype == vx.timestamp("us", tz="UTC")
    assert scalar.as_py() == 1_000_000


def test_timestamp_scalar_rejects_fixed_offset_timezone() -> None:
    zone = datetime.timezone(datetime.timedelta(hours=2))
    with pytest.raises(ValueError, match="Unsupported timezone"):
        _ = vx.scalar(datetime.datetime(1970, 1, 1, tzinfo=zone))


def test_timestamp_scalar_rejects_mismatched_timezone() -> None:
    with pytest.raises(ValueError, match="naive datetime"):
        _ = vx.scalar(datetime.datetime(1970, 1, 1), dtype=vx.timestamp("us", tz="UTC"))
    with pytest.raises(ValueError, match="timezone-aware datetime"):
        _ = vx.scalar(datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC), dtype=vx.timestamp("us"))


def test_timestamp_scalar_rejects_precision_loss() -> None:
    with pytest.raises(ValueError, match="without losing precision"):
        _ = vx.scalar(datetime.datetime(1970, 1, 1, 0, 0, 0, 1), dtype=vx.timestamp("s"))


@pytest.mark.parametrize(
    "value,precision,scale",
    [
        ("1.25", 3, 2),
        ("-0.001", 3, 3),
        ("12E+2", 4, 0),
        ("0", 1, 0),
        ("9" * 38, 38, 0),
    ],
)
def test_decimal_scalar_infers_precision_and_scale(value: str, precision: int, scale: int) -> None:
    scalar = vx.scalar(decimal.Decimal(value))
    assert scalar.dtype == vx.decimal(precision=precision, scale=scale)
    assert scalar.as_py() == decimal.Decimal(value)


def test_decimal_scalar_rescales_to_dtype() -> None:
    scalar = vx.scalar(decimal.Decimal("1.5"), dtype=vx.decimal(precision=10, scale=3))
    assert scalar.dtype == vx.decimal(precision=10, scale=3)
    assert scalar.as_py() == decimal.Decimal("1.500")


def test_decimal_scalar_rejects_precision_loss() -> None:
    with pytest.raises(ValueError, match="without losing precision"):
        _ = vx.scalar(decimal.Decimal("1.25"), dtype=vx.decimal(precision=10, scale=1))


def test_decimal_scalar_rejects_overflowing_precision() -> None:
    with pytest.raises(ValueError, match="does not fit in precision"):
        _ = vx.scalar(decimal.Decimal("123.4"), dtype=vx.decimal(precision=3, scale=1))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_decimal_scalar_rejects_non_finite(value: str) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _ = vx.scalar(decimal.Decimal(value))


def test_uuid_scalar() -> None:
    value = uuid.UUID("12345678-1234-5678-1234-567812345678")
    scalar = vx.scalar(value)
    assert isinstance(scalar, vx.ExtensionScalar)
    assert "vortex.uuid" in repr(scalar.dtype)
    assert scalar.as_py() == list(value.bytes)


def test_list_scalar_with_dtype() -> None:
    scalar = vx.scalar([1, 2], dtype=vx.list_(vx.int_(32)))
    assert isinstance(scalar, vx.ListScalar)
    assert scalar.dtype == vx.list_(vx.int_(32))
    assert scalar.as_py() == [1, 2]


def test_struct_scalar_with_dtype() -> None:
    dtype = vx.struct({"a": vx.int_(32), "b": vx.list_(vx.int_(8))})
    scalar = vx.scalar({"a": 1, "b": [2]}, dtype=dtype)
    assert scalar.dtype == dtype
    assert scalar.as_py() == {"a": 1, "b": [2]}


def test_timestamp_scalar_pytz_zone() -> None:
    pytz = pytest.importorskip("pytz")
    value = pytz.timezone("America/New_York").localize(datetime.datetime(1970, 1, 1))
    scalar = vx.scalar(value)
    assert scalar.dtype == vx.timestamp("us", tz="America/New_York")
    assert scalar.as_py() == 5 * 3_600_000_000


def test_timestamp_scalar_keeps_pandas_nanoseconds() -> None:
    pd = pytest.importorskip("pandas")
    value = pd.Timestamp("1970-01-01 00:00:00.000001001")
    assert vx.scalar(value, dtype=vx.timestamp("ns")).as_py() == 1_001
    with pytest.raises(ValueError, match="without losing precision"):
        _ = vx.scalar(value)


def test_decimal_scalar_rejects_unrepresentable_scale() -> None:
    with pytest.raises(ValueError, match="cannot be represented as a Vortex decimal"):
        _ = vx.scalar(decimal.Decimal("1E-100"))
