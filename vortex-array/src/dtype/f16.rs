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
            fn from_f16(_: f16) -> Option<Self> {
                None
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
