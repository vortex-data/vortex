// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use half::f16;
use num_traits::FromPrimitive;

/// A trait for types that can be created from primitive values, including f16.
///
/// This extends the `FromPrimitive` trait to also support conversion from f16 values.
pub trait FromPrimitiveOrF16: FromPrimitive {
    /// Converts an f16 value to this type, returning None if the conversion fails.
    fn from_f16(v: f16) -> Option<Self>;
}

macro_rules! from_primitive_or_f16_for_non_floating_point {
    ($T:ty) => {
        impl FromPrimitiveOrF16 for $T {
            fn from_f16(_: f16) -> Option<Self> {
                None
            }
        }
    };
}

from_primitive_or_f16_for_non_floating_point!(usize);
from_primitive_or_f16_for_non_floating_point!(u8);
from_primitive_or_f16_for_non_floating_point!(u16);
from_primitive_or_f16_for_non_floating_point!(u32);
from_primitive_or_f16_for_non_floating_point!(u64);
from_primitive_or_f16_for_non_floating_point!(i8);
from_primitive_or_f16_for_non_floating_point!(i16);
from_primitive_or_f16_for_non_floating_point!(i32);
from_primitive_or_f16_for_non_floating_point!(i64);

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
