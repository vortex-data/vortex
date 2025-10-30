// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display};
use std::panic::RefUnwindSafe;

use paste::paste;

use crate::{BigCast, i256};

/// Type of the decimal values.
///
/// This is used for other crates to understand the different underlying representations possible
/// for decimals.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Hash, prost::Enumeration)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[repr(u8)]
pub enum DecimalType {
    /// 8-bit decimal value type.
    I8 = 0,
    /// 16-bit decimal value type.
    I16 = 1,
    /// 32-bit decimal value type.
    I32 = 2,
    /// 64-bit decimal value type.
    I64 = 3,
    /// 128-bit decimal value type.
    I128 = 4,
    /// 256-bit decimal value type.
    I256 = 5,
}

/// Type of decimal scalar values.
///
/// This trait is implemented by native integer types that can be used to store decimal values.
pub trait NativeDecimalType:
    Send
    + Sync
    + Clone
    + Copy
    + Debug
    + Display
    + Default
    + RefUnwindSafe
    + Eq
    + Ord
    + BigCast
    + 'static
{
    /// The decimal value type corresponding to this native type.
    const DECIMAL_TYPE: DecimalType;

    /// Downcast the provided object to a type-specific instance.
    fn downcast<V: DecimalTypeDowncast>(visitor: V) -> V::Output<Self>;

    /// Upcast a type-specific instance to a generic instance.
    fn upcast<V: DecimalTypeUpcast>(input: V::Input<Self>) -> V;
}

/// Trait for downcasting decimal values to native integer types.
pub trait DecimalTypeDowncast {
    /// The output type for downcasting.
    type Output<T: NativeDecimalType>;

    /// Downcast to i8.
    fn into_i8(self) -> Self::Output<i8>;
    /// Downcast to i16.
    fn into_i16(self) -> Self::Output<i16>;
    /// Downcast to i32.
    fn into_i32(self) -> Self::Output<i32>;
    /// Downcast to i64.
    fn into_i64(self) -> Self::Output<i64>;
    /// Downcast to i128.
    fn into_i128(self) -> Self::Output<i128>;
    /// Downcast to i256.
    fn into_i256(self) -> Self::Output<i256>;
}

/// Trait for upcasting native integer types to decimal values.
pub trait DecimalTypeUpcast {
    /// The input type for upcasting.
    type Input<T: NativeDecimalType>;

    /// Upcast from i8.
    fn from_i8(input: Self::Input<i8>) -> Self;
    /// Upcast from i16.
    fn from_i16(input: Self::Input<i16>) -> Self;
    /// Upcast from i32.
    fn from_i32(input: Self::Input<i32>) -> Self;
    /// Upcast from i64.
    fn from_i64(input: Self::Input<i64>) -> Self;
    /// Upcast from i128.
    fn from_i128(input: Self::Input<i128>) -> Self;
    /// Upcast from i256.
    fn from_i256(input: Self::Input<i256>) -> Self;
}

macro_rules! impl_decimal {
    ($T:ty, $UPPER:ident) => {
        paste! {
            impl NativeDecimalType for $T {
                const DECIMAL_TYPE: DecimalType = DecimalType::$UPPER;

                #[inline]
                fn downcast<V: DecimalTypeDowncast>(visitor: V) -> V::Output<Self> {
                    paste::paste! { visitor.[<into_ $T>]() }
                }

                #[inline]
                fn upcast<V: DecimalTypeUpcast>(input: V::Input<Self>) -> V {
                    paste::paste! { V::[<from_ $T>](input) }
                }
            }
        }
    };
}

impl_decimal!(i8, I8);
impl_decimal!(i16, I16);
impl_decimal!(i32, I32);
impl_decimal!(i64, I64);
impl_decimal!(i128, I128);
impl_decimal!(i256, I256);
