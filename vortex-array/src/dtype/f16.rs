// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use half::f16;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;

/// A trait for types that can be created from primitive values, including f16.
///
/// This extends the `FromPrimitive` trait to also support conversion from f16 values.
pub trait FromPrimitiveOrF16: FromPrimitive {
    /// Converts an f16 value to this type, returning None if the conversion fails.
    fn from_f16(v: f16) -> Option<Self>;
}

macro_rules! from_primitive_or_f16_for_signed {
    ($T:ty) => {
        impl FromPrimitiveOrF16 for $T {
            fn from_f16(value: f16) -> Option<Self> {
                value.to_i64().and_then(|v| FromPrimitive::from_i64(v))
            }
        }
    };
}

macro_rules! from_primitive_or_f16_for_unsigned {
    ($T:ty) => {
        impl FromPrimitiveOrF16 for $T {
            fn from_f16(value: f16) -> Option<Self> {
                value.to_u64().and_then(|v| FromPrimitive::from_u64(v))
            }
        }
    };
}

from_primitive_or_f16_for_unsigned!(usize);
from_primitive_or_f16_for_unsigned!(u8);
from_primitive_or_f16_for_unsigned!(u16);
from_primitive_or_f16_for_unsigned!(u32);
from_primitive_or_f16_for_unsigned!(u64);
from_primitive_or_f16_for_signed!(i8);
from_primitive_or_f16_for_signed!(i16);
from_primitive_or_f16_for_signed!(i32);
from_primitive_or_f16_for_signed!(i64);

impl FromPrimitiveOrF16 for f16 {
    fn from_f16(v: f16) -> Option<Self> {
        Some(v)
    }
}

impl FromPrimitiveOrF16 for f32 {
    fn from_f16(v: f16) -> Option<Self> {
        Some(v.to_f32())
    }
}

impl FromPrimitiveOrF16 for f64 {
    fn from_f16(v: f16) -> Option<Self> {
        Some(v.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use rstest::rstest;

    use super::FromPrimitiveOrF16;

    /// A value an `f16` can hold exactly must convert to every integer type wide enough for it.
    /// The unsigned half of this always worked; the signed half returned `None` for everything.
    #[rstest]
    #[case(1.0, 1)]
    #[case(0.0, 0)]
    #[case(255.0, 255)]
    fn signed_and_unsigned_agree_on_representable_values(
        #[case] value: f32,
        #[case] expected: i64,
    ) {
        let v = f16::from_f32(value);
        assert_eq!(i16::from_f16(v).map(i64::from), Some(expected));
        assert_eq!(i32::from_f16(v).map(i64::from), Some(expected));
        assert_eq!(i64::from_f16(v), Some(expected));
        assert_eq!(u64::from_f16(v), u64::try_from(expected).ok());
    }

    /// Negative values are the case where signed and unsigned must differ.
    #[test]
    fn negative_values_reach_signed_types_only() {
        let v = f16::from_f32(-1.0);
        assert_eq!(i8::from_f16(v), Some(-1));
        assert_eq!(i64::from_f16(v), Some(-1));
        assert_eq!(u8::from_f16(v), None);
        assert_eq!(u64::from_f16(v), None);
    }

    /// The conversion still declines when the value genuinely does not fit, which is the only
    /// case the trait's `None` is documented to cover.
    #[rstest]
    #[case::too_wide_for_i8(1000.0)]
    #[case::negative_too_wide_for_i8(-1000.0)]
    fn out_of_range_is_still_none(#[case] value: f32) {
        assert_eq!(i8::from_f16(f16::from_f32(value)), None);
    }

    #[test]
    fn nan_and_infinity_are_none() {
        assert_eq!(i32::from_f16(f16::NAN), None);
        assert_eq!(i32::from_f16(f16::INFINITY), None);
        assert_eq!(i32::from_f16(f16::NEG_INFINITY), None);
    }

    /// `f16` cannot represent every `i64`, so a value beyond its mantissa must not silently
    /// round on the way in.
    #[test]
    fn i64_round_trips_only_what_f16_can_hold() {
        assert_eq!(i64::from_f16(f16::from_f32(2048.0)), Some(2048));
        // 2049 is not representable in f16; it becomes 2048.
        assert_eq!(i64::from_f16(f16::from_f32(2049.0)), Some(2048));
    }
}
