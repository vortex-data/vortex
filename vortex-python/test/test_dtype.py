# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest

import vortex as vx


def test_factories() -> None:
    assert str(vx.null()) == "null"
    assert str(vx.bool_()) == "bool"
    assert str(vx.bool_(nullable=True)) == "bool?"
    assert str(vx.int_()) == "i64"
    assert str(vx.int_(32)) == "i32"
    assert str(vx.int_(32, nullable=True)) == "i32?"
    assert str(vx.uint(32)) == "u32"
    assert str(vx.float_(16)) == "f16"
    assert str(vx.struct({"a": vx.int_(nullable=True)})) == "{a=i64?}"


def test_timestamp_rejects_unknown_timezone():
    with pytest.raises(RuntimeError, match="Not/A/Timezone"):
        vx.timestamp("s", tz="Not/A/Timezone")


def test_fixed_size_binary():
    dtype = vx.fixed_size_binary(16, nullable=True)
    assert isinstance(dtype, vx.FixedSizeBinaryDType)
    assert dtype.byte_width == 16
    assert str(dtype) == "fixed_size_binary[16]?"
    assert repr(vx.fixed_size_binary(4)) == "fixed_size_binary(4, nullable=False)"


def test_fixed_size_binary_arrow_round_trip():
    import pyarrow as pa

    arrow = pa.array([b"ab", None, b"cd"], type=pa.binary(2))
    array = vx.array(arrow)
    assert array.dtype == vx.fixed_size_binary(2, nullable=True)
    assert array.scalar_at(0).as_py() == b"ab"
    assert array.scalar_at(1).as_py() is None
    assert array.to_arrow_array() == arrow
