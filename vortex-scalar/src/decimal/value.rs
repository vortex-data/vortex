// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Additional trait implementations for decimal types to ensure consistency.

use core::cmp::Ordering;
use core::hash::Hash;

use vortex_dtype::{DecimalDType, Nullability};
use vortex_error::{VortexError, VortexExpect, vortex_err};

use crate::{
    DecimalScalar, InnerScalarValue, NativeDecimalType, Scalar, ScalarValue, ToPrimitive, i256,
};

/// Matches over each decimal value variant, binding the inner value to a variable.
///
/// # Example
///
/// ```ignore
/// match_each_decimal_value!(value, |v| {
///     println!("Value: {}", v);
/// });
/// ```
#[macro_export] // Used in `vortex-array`.
macro_rules! match_each_decimal_value {
    ($self:expr, | $value:ident | $body:block) => {{
        match $self {
            DecimalValue::I8(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I16(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I32(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I64(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I128(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I256(v) => {
                let $value = v;
                $body
            }
        }
    }};
}

/// Macro to match over each decimal value type, binding the corresponding native type (from
/// `DecimalValueType`)
#[macro_export] // Used in `vortex-array`.
macro_rules! match_each_decimal_value_type {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::{DecimalValueType, i256};
        match $self {
            DecimalValueType::I8 => {
                type $enc = i8;
                $body
            }
            DecimalValueType::I16 => {
                type $enc = i16;
                $body
            }
            DecimalValueType::I32 => {
                type $enc = i32;
                $body
            }
            DecimalValueType::I64 => {
                type $enc = i64;
                $body
            }
            DecimalValueType::I128 => {
                type $enc = i128;
                $body
            }
            DecimalValueType::I256 => {
                type $enc = i256;
                $body
            }
            ty => unreachable!("unknown decimal value type {:?}", ty),
        }
    }};
}

/// Type of the decimal values.
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
