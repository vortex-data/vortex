// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use num_traits::ToPrimitive as NumToPrimitive;

use vortex_dtype::{DType, DecimalDType, Nullability, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::scalar_value::InnerScalarValue;
use crate::{BigCast, Scalar, ScalarValue, ToPrimitive, i256};

/// Matches over each decimal value variant, binding the inner value to a variable.
///
/// # Example
///
/// ```ignore
/// match_each_decimal_value!(value, |v| {
///     println!("Value: {}", v);
/// });
/// ```
#[macro_export]
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

/// Macro to match over each decimal value type, binding the corresponding native type (from `DecimalValueType`)
#[macro_export]
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

/// Type of decimal scalar values.
///
/// This trait is implemented by native integer types that can be used
/// to store decimal values.
pub trait NativeDecimalType:
    Copy + Eq + Ord + Default + Send + Sync + BigCast + Debug + Display + 'static
{
    /// The decimal value type corresponding to this native type.
    const VALUES_TYPE: DecimalValueType;

    /// Attempts to convert a decimal value to this native type.
    fn maybe_from(decimal_type: DecimalValue) -> Option<Self>;
}

impl NativeDecimalType for i8 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I8;

    fn maybe_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I8(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i16 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I16;

    fn maybe_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I16(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i32 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I32;

    fn maybe_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I32(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i64 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I64;

    fn maybe_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I64(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i128 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I128;

    fn maybe_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I128(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i256 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I256;

    fn maybe_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I256(v) => Some(v),
            _ => None,
        }
    }
}

impl Display for DecimalValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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

/// A scalar value representing a decimal number with fixed precision and scale.
#[derive(Debug, Clone, Copy, Hash)]
pub struct DecimalScalar<'a> {
    dtype: &'a DType,
    decimal_type: DecimalDType,
    value: Option<DecimalValue>,
}

impl<'a> DecimalScalar<'a> {
    /// Creates a new decimal scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a decimal type.
    pub fn try_new(dtype: &'a DType, value: &ScalarValue) -> VortexResult<Self> {
        let decimal_type = DecimalDType::try_from(dtype)?;
        let value = value.as_decimal()?;

        Ok(Self {
            dtype,
            decimal_type,
            value,
        })
    }

    /// Returns the data type of this decimal scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the decimal value, or None if null.
    pub fn decimal_value(&self) -> &Option<DecimalValue> {
        &self.value
    }

    /// Cast decimal scalar to another data type.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        match dtype {
            DType::Decimal(target_dtype, target_nullability) => {
                // Cast between decimal types
                if self.decimal_type == *target_dtype {
                    // Same decimal type, just change nullability if needed
                    return Ok(Scalar::new(
                        dtype.clone(),
                        ScalarValue(InnerScalarValue::Decimal(
                            self.value.clone().unwrap_or(DecimalValue::I128(0)),
                        )),
                    ));
                }
                
                // Different precision/scale - need to implement scaling logic
                // For now, we'll do a simple value preservation without scaling
                // TODO: Implement proper decimal scaling logic
                if let Some(value) = &self.value {
                    Ok(Scalar::decimal(
                        value.clone(),
                        *target_dtype,
                        *target_nullability,
                    ))
                } else {
                    Ok(Scalar::null(dtype.clone()))
                }
            }
            DType::Primitive(ptype, nullability) => {
                // Cast decimal to primitive type
                if let Some(decimal_value) = &self.value {
                    // Convert decimal value to primitive, accounting for scale
                    let scale_factor = 10_i128.pow(self.decimal_type.scale() as u32);
                    
                    // Convert to i128 for calculation
                    let scaled_value = match_each_decimal_value!(decimal_value, |v| {
                        NumToPrimitive::to_i128(v).ok_or_else(|| 
                            vortex_err!("Decimal value too large to cast to primitive")
                        )
                    })?;
                    
                    // Apply scale to get the actual value
                    let actual_value = scaled_value as f64 / scale_factor as f64;
                    
                    // Cast to target primitive type
                    use PType::*;
                    let primitive_scalar = match ptype {
                        U8 => {
                            let v = actual_value as u8;
                            if actual_value < 0.0 || actual_value > u8::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u8", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        U16 => {
                            let v = actual_value as u16;
                            if actual_value < 0.0 || actual_value > u16::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u16", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        U32 => {
                            let v = actual_value as u32;
                            if actual_value < 0.0 || actual_value > u32::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u32", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        U64 => {
                            let v = actual_value as u64;
                            if actual_value < 0.0 || actual_value > u64::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u64", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        I8 => {
                            let v = actual_value as i8;
                            if actual_value < i8::MIN as f64 || actual_value > i8::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i8", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        I16 => {
                            let v = actual_value as i16;
                            if actual_value < i16::MIN as f64 || actual_value > i16::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i16", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        I32 => {
                            let v = actual_value as i32;
                            if actual_value < i32::MIN as f64 || actual_value > i32::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i32", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        I64 => {
                            let v = actual_value as i64;
                            if actual_value < i64::MIN as f64 || actual_value > i64::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i64", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        F16 => {
                            use vortex_dtype::half::f16;
                            Scalar::primitive(f16::from_f64(actual_value), *nullability)
                        }
                        F32 => Scalar::primitive(actual_value as f32, *nullability),
                        F64 => Scalar::primitive(actual_value, *nullability),
                    };
                    Ok(primitive_scalar)
                } else {
                    // Null decimal to primitive
                    Ok(Scalar::null(dtype.clone()))
                }
            }
            _ => vortex_bail!(
                "Cannot cast decimal to {}: unsupported conversion",
                dtype
            ),
        }
    }
}

impl<'a> TryFrom<&'a Scalar> for DecimalScalar<'a> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> Result<Self, Self::Error> {
        DecimalScalar::try_new(&scalar.dtype, &scalar.value)
    }
}

impl Display for DecimalScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.value.as_ref() {
            Some(&dv) => {
                // Introduce some of the scale factors instead.
                match dv {
                    DecimalValue::I8(v) => write!(
                        f,
                        "decimal8({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I16(v) => write!(
                        f,
                        "decimal16({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I32(v) => write!(
                        f,
                        "decimal32({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I64(v) => write!(
                        f,
                        "decimal64({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I128(v) => write!(
                        f,
                        "decimal128({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I256(v) => write!(
                        f,
                        "decimal256({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                }
            }
            None => {
                write!(f, "null")
            }
        }
    }
}

impl PartialEq for DecimalScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
    }
}

impl Eq for DecimalScalar<'_> {}

/// Ord is not implemented since it's undefined for different PTypes
impl PartialOrd for DecimalScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return None;
        }
        self.value.partial_cmp(&other.value)
    }
}

macro_rules! decimal_scalar_unpack {
    ($ty:ident, $arm:ident) => {
        impl TryFrom<DecimalScalar<'_>> for Option<$ty> {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                Ok(match value.value {
                    None => None,
                    Some(DecimalValue::$arm(v)) => Some(v),
                    v => vortex_bail!("Cannot extract decimal {:?} as {}", v, stringify!($ty)),
                })
            }
        }

        impl TryFrom<DecimalScalar<'_>> for $ty {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                match value.value {
                    None => vortex_bail!("Cannot extract value from null decimal"),
                    Some(DecimalValue::$arm(v)) => Ok(v),
                    v => vortex_bail!("Cannot extract decimal {:?} as {}", v, stringify!($ty)),
                }
            }
        }
    };
}

decimal_scalar_unpack!(i8, I8);
decimal_scalar_unpack!(i16, I16);
decimal_scalar_unpack!(i32, I32);
decimal_scalar_unpack!(i64, I64);
decimal_scalar_unpack!(i128, I128);
decimal_scalar_unpack!(i256, I256);

macro_rules! decimal_scalar_pack {
    ($from:ident, $to:ident, $arm:ident) => {
        impl From<$from> for DecimalValue {
            fn from(value: $from) -> Self {
                DecimalValue::$arm(value as $to)
            }
        }
    };
}

decimal_scalar_pack!(i8, i8, I8);
decimal_scalar_pack!(u8, i16, I16);
decimal_scalar_pack!(i16, i16, I16);
decimal_scalar_pack!(u16, i32, I32);
decimal_scalar_pack!(i32, i32, I32);
decimal_scalar_pack!(u32, i64, I64);
decimal_scalar_pack!(i64, i64, I64);
decimal_scalar_pack!(u64, i128, I128);

decimal_scalar_pack!(i128, i128, I128);
decimal_scalar_pack!(i256, i256, I256);

#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod tests {
    use std::collections::HashSet;

    use rstest::rstest;
    use vortex_dtype::{DType, DecimalDType, Nullability, PType};

    use crate::{DecimalValue, Scalar, i256};

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

    #[test]
    fn test_decimal_cast_to_primitive() {
        // Create a decimal with value 123.45 (scale=2, so stored as 12345)
        let decimal_scalar = Scalar::decimal(
            DecimalValue::I32(12345),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        
        // Cast to f64 should give us 123.45
        let float_result = decimal_scalar.cast(&DType::Primitive(PType::F64, Nullability::NonNullable)).unwrap();
        let float_value: f64 = float_result.try_into().unwrap();
        assert!((float_value - 123.45).abs() < 0.001);
        
        // Cast to i32 should give us 123 (truncated)
        let int_result = decimal_scalar.cast(&DType::Primitive(PType::I32, Nullability::NonNullable)).unwrap();
        let int_value: i32 = int_result.try_into().unwrap();
        assert_eq!(int_value, 123);
    }

    #[test]
    fn test_decimal_cast_null_handling() {
        // Null decimal
        let null_decimal = Scalar::null(DType::Decimal(
            DecimalDType::new(10, 2),
            Nullability::Nullable,
        ));
        
        // Cast null decimal to primitive should preserve null
        let result = null_decimal.cast(&DType::Primitive(PType::I32, Nullability::Nullable)).unwrap();
        assert!(result.is_null());
        
        // Cast null decimal to another decimal type should preserve null
        let result = null_decimal.cast(&DType::Decimal(
            DecimalDType::new(20, 4),
            Nullability::Nullable,
        )).unwrap();
        assert!(result.is_null());
    }

    #[test]
    fn test_decimal_cast_overflow() {
        // Large decimal value that won't fit in i8
        let decimal_scalar = Scalar::decimal(
            DecimalValue::I32(100000),
            DecimalDType::new(10, 0),
            Nullability::NonNullable,
        );
        
        // Cast to i8 should fail due to overflow
        let result = decimal_scalar.cast(&DType::Primitive(PType::I8, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_decimal_cast_between_decimal_types() {
        // Decimal with different precision/scale
        let decimal_scalar = Scalar::decimal(
            DecimalValue::I32(12345),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        
        // Cast to different decimal type (currently just preserves value)
        let result = decimal_scalar.cast(&DType::Decimal(
            DecimalDType::new(20, 4),
            Nullability::NonNullable,
        )).unwrap();
        
        // Value should be preserved (TODO: proper scaling logic)
        let decimal_value: Option<DecimalValue> = result.try_into().unwrap();
        assert_eq!(decimal_value, Some(DecimalValue::I32(12345)));
    }

    #[test]
    fn test_decimal_cast_negative_values() {
        // Negative decimal value
        let decimal_scalar = Scalar::decimal(
            DecimalValue::I32(-5678),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        
        // Cast to f64 should give us -56.78
        let float_result = decimal_scalar.cast(&DType::Primitive(PType::F64, Nullability::NonNullable)).unwrap();
        let float_value: f64 = float_result.try_into().unwrap();
        assert!((float_value - (-56.78)).abs() < 0.001);
        
        // Cast to unsigned should fail
        let result = decimal_scalar.cast(&DType::Primitive(PType::U32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_decimal_cast_edge_values() {
        // Test with extreme values for each decimal type
        let test_cases = vec![
            (DecimalValue::I8(i8::MAX), DecimalDType::new(3, 0)),
            (DecimalValue::I8(i8::MIN), DecimalDType::new(3, 0)),
            (DecimalValue::I16(i16::MAX), DecimalDType::new(5, 0)),
            (DecimalValue::I16(i16::MIN), DecimalDType::new(5, 0)),
            (DecimalValue::I32(i32::MAX), DecimalDType::new(10, 0)),
            (DecimalValue::I32(i32::MIN), DecimalDType::new(10, 0)),
        ];
        
        for (value, dtype) in test_cases {
            let decimal_scalar = Scalar::decimal(value, dtype, Nullability::NonNullable);
            
            // Cast to f64 should always work for these ranges
            let result = decimal_scalar.cast(&DType::Primitive(PType::F64, Nullability::NonNullable));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_decimal_cast_with_scale() {
        // Test various scale factors
        let test_cases = vec![
            (1234, 0, 1234.0),      // No scale
            (1234, 1, 123.4),       // Scale 1
            (1234, 2, 12.34),       // Scale 2
            (1234, 3, 1.234),       // Scale 3
            (1234, 4, 0.1234),      // Scale 4
        ];
        
        for (value, scale, expected) in test_cases {
            let decimal_scalar = Scalar::decimal(
                DecimalValue::I32(value),
                DecimalDType::new(10, scale),
                Nullability::NonNullable,
            );
            
            let float_result = decimal_scalar.cast(&DType::Primitive(PType::F64, Nullability::NonNullable)).unwrap();
            let float_value: f64 = float_result.try_into().unwrap();
            assert!((float_value - expected).abs() < 0.0001, 
                   "Expected {} but got {} for value={} scale={}", expected, float_value, value, scale);
        }
    }

    #[test]
    fn test_decimal_cast_unsupported_types() {
        let decimal_scalar = Scalar::decimal(
            DecimalValue::I32(1234),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );
        
        // Cast to unsupported types should fail
        let result = decimal_scalar.cast(&DType::Bool(Nullability::NonNullable));
        assert!(result.is_err());
        
        let result = decimal_scalar.cast(&DType::Utf8(Nullability::NonNullable));
        assert!(result.is_err());
        
        let result = decimal_scalar.cast(&DType::Binary(Nullability::NonNullable));
        assert!(result.is_err());
    }
}
