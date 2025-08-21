// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests;
mod value;

use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use num_traits::ToPrimitive as NumToPrimitive;
use vortex_dtype::{DType, DecimalDType, Nullability, PType};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};

pub use crate::decimal::value::{DecimalValue, DecimalValueType};
use crate::scalar_value::InnerScalarValue;
use crate::{BigCast, Scalar, ScalarValue, i256, match_each_decimal_value};

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
    pub fn decimal_value(&self) -> Option<DecimalValue> {
        self.value
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
                            self.value.unwrap_or(DecimalValue::I128(0)),
                        )),
                    ));
                }

                // Different precision/scale - need to implement scaling logic
                // For now, we'll do a simple value preservation without scaling
                // TODO: Implement proper decimal scaling logic
                if let Some(value) = &self.value {
                    Ok(Scalar::decimal(*value, *target_dtype, *target_nullability))
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
                        NumToPrimitive::to_i128(v).ok_or_else(|| {
                            vortex_err!("Decimal value too large to cast to primitive")
                        })
                    })?;

                    // Apply scale to get the actual value
                    let actual_value = scaled_value as f64 / scale_factor as f64;

                    // Cast to target primitive type
                    use PType::*;
                    #[allow(clippy::cast_possible_truncation)]
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
                "Cannot cast decimal to {dtype}: decimal scalars can only be cast to decimal or primitive numeric types"
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
    ($T:ident, $arm:ident) => {
        impl TryFrom<DecimalScalar<'_>> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                Ok(match value.value {
                    None => None,
                    Some(DecimalValue::$arm(v)) => Some(v),
                    v => vortex_bail!("Cannot extract decimal {:?} as {}", v, stringify!($T)),
                })
            }
        }

        impl TryFrom<DecimalScalar<'_>> for $T {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                match value.value {
                    None => vortex_bail!("Cannot extract value from null decimal"),
                    Some(DecimalValue::$arm(v)) => Ok(v),
                    v => vortex_bail!("Cannot extract decimal {:?} as {}", v, stringify!($T)),
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
