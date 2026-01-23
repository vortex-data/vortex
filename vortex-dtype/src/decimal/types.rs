// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::panic::RefUnwindSafe;

use num_traits::ConstOne;
use num_traits::ConstZero;
use paste::paste;
use vortex_error::VortexError;
use vortex_error::vortex_bail;

use crate::BigCast;
use crate::DecimalDType;
use crate::PType;
use crate::decimal::max_precision::MAX_DECIMAL256_FOR_EACH_PRECISION;
use crate::decimal::max_precision::MIN_DECIMAL256_FOR_EACH_PRECISION;
use crate::i256;

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

impl DecimalType {
    /// Maps a `DecimalDType` (precision) into the smallest `DecimalType` that can represent it.
    pub fn smallest_decimal_value_type(decimal_dtype: &DecimalDType) -> DecimalType {
        match decimal_dtype.precision() {
            1..=2 => DecimalType::I8,
            3..=4 => DecimalType::I16,
            5..=9 => DecimalType::I32,
            10..=18 => DecimalType::I64,
            19..=38 => DecimalType::I128,
            39..=76 => DecimalType::I256,
            0 => unreachable!("precision must be greater than 0"),
            p => unreachable!("precision larger than 76 is invalid found precision {p}"),
        }
    }

    /// Returns the size in bytes of the underlying native type for this decimal type.
    pub fn byte_width(&self) -> usize {
        match self {
            DecimalType::I8 => size_of::<i8>(),
            DecimalType::I16 => size_of::<i16>(),
            DecimalType::I32 => size_of::<i32>(),
            DecimalType::I64 => size_of::<i64>(),
            DecimalType::I128 => size_of::<i128>(),
            DecimalType::I256 => size_of::<i256>(),
        }
    }

    /// True if `Self` can represent every value of the type `DecimalDType`.
    pub fn is_compatible_decimal_value_type(self, dtype: DecimalDType) -> bool {
        self >= Self::smallest_decimal_value_type(&dtype)
    }
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

    /// The maximum precision supported by this decimal type.
    const MAX_PRECISION: u8;
    /// The maximum scale supported by this decimal type.
    const MAX_SCALE: i8;

    /// The minimum value for each precision supported by this decimal type.
    /// This is an array of length `MAX_PRECISION + 1` where the `i`th element is the minimum value
    /// for a precision of `i` (including precision 0).
    const MIN_BY_PRECISION: &'static [Self];
    /// The maximum value for each precision supported by this decimal type.
    /// similar to `MIN_BY_PRECISION`.
    const MAX_BY_PRECISION: &'static [Self];

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

                const MAX_PRECISION: u8 = match DecimalType::$UPPER {
                    DecimalType::I8 => 2,
                    DecimalType::I16 => 4,
                    DecimalType::I32 => 9,
                    DecimalType::I64 => 18,
                    DecimalType::I128 => 38,
                    DecimalType::I256 => 76,
                };
                const MAX_SCALE: i8 = Self::MAX_PRECISION as i8;

                const MIN_BY_PRECISION: &'static [Self] = &{
                    let mut mins = [$T::ZERO; Self::MAX_PRECISION as usize + 1];
                    let mut p = $T::ONE;
                    let mut i = 0;
                    while i < Self::MAX_PRECISION as usize {
                        p = p * 10;
                        mins[i + 1] = -(p - 1);
                        i += 1;
                    }
                    mins
                };

                const MAX_BY_PRECISION: &'static [Self] = &{
                    let mut maxs = [$T::ZERO; Self::MAX_PRECISION as usize + 1];
                    let mut p = $T::ONE;
                    let mut i = 0;
                    while i < Self::MAX_PRECISION as usize {
                        p = p * 10;
                        maxs[i + 1] = p - 1;
                        i += 1;
                    }
                    maxs
                };

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

impl NativeDecimalType for i256 {
    const DECIMAL_TYPE: DecimalType = DecimalType::I256;
    const MAX_PRECISION: u8 = 76;
    const MAX_SCALE: i8 = 76;
    const MIN_BY_PRECISION: &'static [Self] = &MIN_DECIMAL256_FOR_EACH_PRECISION;
    const MAX_BY_PRECISION: &'static [Self] = &MAX_DECIMAL256_FOR_EACH_PRECISION;

    fn downcast<V: DecimalTypeDowncast>(visitor: V) -> V::Output<Self> {
        visitor.into_i256()
    }

    fn upcast<V: DecimalTypeUpcast>(input: V::Input<Self>) -> V {
        V::from_i256(input)
    }
}

impl TryFrom<PType> for DecimalType {
    type Error = VortexError;

    fn try_from(value: PType) -> Result<Self, Self::Error> {
        Ok(match value {
            PType::I8 => DecimalType::I8,
            PType::I16 => DecimalType::I8,
            PType::I32 => DecimalType::I8,
            PType::I64 => DecimalType::I8,
            p @ (PType::U8
            | PType::U16
            | PType::U32
            | PType::U64
            | PType::F16
            | PType::F32
            | PType::F64) => vortex_bail!("cannot convert ptype {p} to DecimalType"),
        })
    }
}
