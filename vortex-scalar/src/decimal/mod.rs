// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display};

use crate::{BigCast, i256};

mod scalar;
pub use scalar::*;

mod value;
pub use value::{DecimalValue, DecimalValueType};

/// Type of decimal scalar values.
///
/// This trait is implemented by native integer types that can be used to store decimal values.
pub trait NativeDecimalType:
    Copy + Eq + Ord + Default + Send + Sync + BigCast + Debug + Display + 'static
{
    /// The decimal value type corresponding to this native type.
    const VALUES_TYPE: DecimalValueType;

    /// Attempts to convert a decimal value to this native type.
    fn maybe_from(decimal_type: DecimalValue) -> Option<Self>;
}

mod macros;
use macros::impl_native_decimal_type;

impl_native_decimal_type!(i8, I8);
impl_native_decimal_type!(i16, I16);
impl_native_decimal_type!(i32, I32);
impl_native_decimal_type!(i64, I64);
impl_native_decimal_type!(i128, I128);
impl_native_decimal_type!(i256, I256);

#[cfg(test)]
mod tests;
