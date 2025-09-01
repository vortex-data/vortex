// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Additional trait implementations for decimal types to ensure consistency.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;

use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexError, VortexExpect, vortex_err};

use crate::{
    DecimalScalar, InnerScalarValue, NativeDecimalType, Scalar, ScalarValue, ToPrimitive, i256,
    match_each_decimal_value,
};

/// Type of the decimal values.
///
/// This is used for other crates to understand the different underlying representations possible
/// for decimals.
#[derive(Clone, Copy, Debug, prost::Enumeration, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
#[non_exhaustive]
pub enum DecimalValueType {
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

impl Scalar {
    /// Creates a new decimal scalar with the given value, precision, scale, and nullability.
    pub fn decimal(
        value: DecimalValue,
        decimal_type: DecimalDType,
        nullability: Nullability,
    ) -> Self {
        Self::new(
            DType::Decimal(decimal_type, nullability),
            ScalarValue(InnerScalarValue::Decimal(value)),
        )
    }
}

/// A decimal value that can be stored in various integer widths.
///
/// This enum represents decimal values with different storage sizes,
/// from 8-bit to 256-bit integers.
#[derive(Debug, Clone, Copy)]
pub enum DecimalValue {
    /// 8-bit signed decimal value.
    I8(i8),
    /// 16-bit signed decimal value.
    I16(i16),
    /// 32-bit signed decimal value.
    I32(i32),
    /// 64-bit signed decimal value.
    I64(i64),
    /// 128-bit signed decimal value.
    I128(i128),
    /// 256-bit signed decimal value.
    I256(i256),
}

impl DecimalValue {
    /// Cast `self` to T using the respective `ToPrimitive` method.
    /// If the value cannot be represented by `T`, `None` is returned.
    pub fn cast<T: NativeDecimalType>(&self) -> Option<T> {
        match_each_decimal_value!(self, |value| { T::from(*value) })
    }
}

// Comparisons between DecimalValue types should upcast to i256 and operate in the upcast space.
// Decimal values can take on any signed scalar type, but so long as their values are the same
// they are considered the same.
// DecimalScalar handles ensuring that both values being compared have the same precision/scale.
impl PartialEq for DecimalValue {
    fn eq(&self, other: &Self) -> bool {
        let self_upcast = match_each_decimal_value!(self, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });
        let other_upcast = match_each_decimal_value!(other, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });

        self_upcast == other_upcast
    }
}

impl Eq for DecimalValue {}

impl PartialOrd for DecimalValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_upcast = match_each_decimal_value!(self, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });
        let other_upcast = match_each_decimal_value!(other, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });

        self_upcast.partial_cmp(&other_upcast)
    }
}

// Hashing works in the upcast space similar to the other comparison and equality operators.
impl Hash for DecimalValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let self_upcast = match_each_decimal_value!(self, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });
        self_upcast.hash(state);
    }
}

use super::macros::{decimal_scalar_pack, decimal_scalar_unpack};

decimal_scalar_unpack!(i8, I8);
decimal_scalar_unpack!(i16, I16);
decimal_scalar_unpack!(i32, I32);
decimal_scalar_unpack!(i64, I64);
decimal_scalar_unpack!(i128, I128);
decimal_scalar_unpack!(i256, I256);

decimal_scalar_pack!(i8, i8, I8);
decimal_scalar_pack!(i16, i16, I16);
decimal_scalar_pack!(i32, i32, I32);
decimal_scalar_pack!(i64, i64, I64);
decimal_scalar_pack!(i128, i128, I128);
decimal_scalar_pack!(i256, i256, I256);

decimal_scalar_pack!(u8, i16, I16);
decimal_scalar_pack!(u16, i32, I32);
decimal_scalar_pack!(u32, i64, I64);
decimal_scalar_pack!(u64, i128, I128);

impl From<DecimalValue> for ScalarValue {
    fn from(value: DecimalValue) -> Self {
        Self(InnerScalarValue::Decimal(value))
    }
}

// Add From<DecimalValue> for Scalar to match other types
impl From<DecimalValue> for Scalar {
    fn from(value: DecimalValue) -> Self {
        // Default to a reasonable precision and scale
        // This matches how primitive types work - they get a default nullability
        let dtype = match &value {
            DecimalValue::I8(_) => DecimalDType::new(3, 0),
            DecimalValue::I16(_) => DecimalDType::new(5, 0),
            DecimalValue::I32(_) => DecimalDType::new(10, 0),
            DecimalValue::I64(_) => DecimalDType::new(19, 0),
            DecimalValue::I128(_) => DecimalDType::new(38, 0),
            DecimalValue::I256(_) => DecimalDType::new(76, 0),
        };
        Scalar::decimal(value, dtype, Nullability::NonNullable)
    }
}

// Add TryFrom<&Scalar> for DecimalValue
impl TryFrom<&Scalar> for DecimalValue {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        let decimal_scalar = DecimalScalar::try_from(scalar)?;
        decimal_scalar
            .decimal_value()
            .as_ref()
            .cloned()
            .ok_or_else(|| vortex_err!("Cannot extract DecimalValue from null decimal"))
    }
}

// Add TryFrom<Scalar> for DecimalValue (delegates to &Scalar)
impl TryFrom<Scalar> for DecimalValue {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        DecimalValue::try_from(&scalar)
    }
}

// Add TryFrom<&Scalar> for Option<DecimalValue>
impl TryFrom<&Scalar> for Option<DecimalValue> {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        let decimal_scalar = DecimalScalar::try_from(scalar)?;
        Ok(decimal_scalar.decimal_value())
    }
}

// Add TryFrom<Scalar> for Option<DecimalValue> (delegates to &Scalar)
impl TryFrom<Scalar> for Option<DecimalValue> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Option::<DecimalValue>::try_from(&scalar)
    }
}

impl fmt::Display for DecimalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecimalValue::I8(v8) => write!(f, "decimal8({v8})"),
            DecimalValue::I16(v16) => write!(f, "decimal16({v16})"),
            DecimalValue::I32(v32) => write!(f, "decimal32({v32})"),
            DecimalValue::I64(v32) => write!(f, "decimal64({v32})"),
            DecimalValue::I128(v128) => write!(f, "decimal128({v128})"),
            DecimalValue::I256(v256) => write!(f, "decimal256({v256})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::DType;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;

    #[test]
    fn test_decimal_value_from_scalar() {
        let value = DecimalValue::I32(12345);
        let scalar = Scalar::from(value);

        // Test extraction
        let extracted: DecimalValue = DecimalValue::try_from(&scalar).unwrap();
        assert_eq!(extracted, value);

        // Test owned extraction
        let extracted_owned: DecimalValue = DecimalValue::try_from(scalar).unwrap();
        assert_eq!(extracted_owned, value);
    }

    #[test]
    fn test_decimal_value_option_from_scalar() {
        // Non-null case
        let value = DecimalValue::I64(999999);
        let scalar = Scalar::from(value);

        let extracted: Option<DecimalValue> = Option::try_from(&scalar).unwrap();
        assert_eq!(extracted, Some(value));

        // Null case
        let null_scalar = Scalar::null(DType::Decimal(
            DecimalDType::new(10, 2),
            Nullability::Nullable,
        ));

        let extracted_null: Option<DecimalValue> = Option::try_from(&null_scalar).unwrap();
        assert_eq!(extracted_null, None);
    }

    #[test]
    fn test_decimal_value_from_conversion() {
        // Test that From<DecimalValue> creates reasonable defaults
        let values = vec![
            DecimalValue::I8(127),
            DecimalValue::I16(32767),
            DecimalValue::I32(1000000),
            DecimalValue::I64(1000000000000),
            DecimalValue::I128(123456789012345678901234567890),
            DecimalValue::I256(i256::from_i128(987654321)),
        ];

        for value in values {
            let scalar = Scalar::from(value);
            assert!(!scalar.is_null());

            // Verify we can extract it back
            let extracted: DecimalValue = DecimalValue::try_from(&scalar).unwrap();
            assert_eq!(extracted, value);
        }
    }

    #[rstest]
    #[case(DecimalValue::I8(100), DecimalValue::I8(100))]
    #[case(DecimalValue::I16(0), DecimalValue::I256(i256::ZERO))]
    #[case(DecimalValue::I8(100), DecimalValue::I128(100))]
    fn test_decimal_value_eq(#[case] left: DecimalValue, #[case] right: DecimalValue) {
        assert_eq!(left, right);
    }

    #[rstest]
    #[case(DecimalValue::I128(10), DecimalValue::I8(11))]
    #[case(DecimalValue::I256(i256::ZERO), DecimalValue::I16(10))]
    #[case(DecimalValue::I128(-1_000), DecimalValue::I8(1))]
    fn test_decimal_value_cmp(#[case] lower: DecimalValue, #[case] upper: DecimalValue) {
        assert!(lower < upper, "expected {lower} < {upper}");
    }

    #[test]
    fn test_hash() {
        let mut set = HashSet::new();
        set.insert(DecimalValue::I8(100));
        set.insert(DecimalValue::I16(100));
        set.insert(DecimalValue::I32(100));
        set.insert(DecimalValue::I64(100));
        set.insert(DecimalValue::I128(100));
        set.insert(DecimalValue::I256(i256::from_i128(100)));
        assert_eq!(set.len(), 1);
    }
}
