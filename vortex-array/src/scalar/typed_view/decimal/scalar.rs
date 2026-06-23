// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`DecimalScalar`] typed view implementation.

use std::cmp::Ordering;
use std::fmt;

use num_traits::ToPrimitive as NumToPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::PType;
use crate::match_each_decimal_value;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing a decimal number with fixed precision and scale.
#[derive(Debug, Clone, Copy, Hash)]
pub struct DecimalScalar<'a> {
    /// The data type of this scalar.
    dtype: &'a DType,
    /// The decimal type (precision and scale).
    decimal_type: DecimalDType,
    /// The decimal value, or [`None`] if null.
    decimal_value: Option<DecimalValue>,
}

impl<'a> DecimalScalar<'a> {
    /// Creates a new decimal scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a decimal type.
    pub fn try_new(dtype: &'a DType, value: Option<&ScalarValue>) -> VortexResult<Self> {
        let decimal_type = DecimalDType::try_from(dtype)?;
        let value = value.map(|v| *v.as_decimal());

        Ok(Self {
            dtype,
            decimal_type,
            decimal_value: value,
        })
    }

    /// Returns the data type of this decimal scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the decimal value, or None if null.
    pub fn decimal_value(&self) -> Option<DecimalValue> {
        self.decimal_value
    }

    /// Returns whether this decimal value is zero, or `None` if null.
    pub fn is_zero(&self) -> Option<bool> {
        self.decimal_value.map(|v| v.is_zero())
    }

    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        match dtype {
            DType::Decimal(target_dtype, target_nullability) => {
                if let Some(value) = &self.decimal_value {
                    let value = value.cast_decimal(self.decimal_type, *target_dtype)?;
                    Ok(Scalar::decimal(value, *target_dtype, *target_nullability))
                } else {
                    Ok(Scalar::null(dtype.clone()))
                }
            }
            DType::Primitive(ptype, nullability) => {
                // Cast decimal to primitive type
                if let Some(decimal_value) = &self.decimal_value {
                    // Convert decimal value to primitive, accounting for scale
                    let scale_factor = 10_i128.pow(self.decimal_type.scale() as u32);

                    // Convert to i128 for calculation
                    let scaled_value = match_each_decimal_value!(decimal_value, |v| {
                        NumToPrimitive::to_i128(v).ok_or_else(|| {
                            vortex_err!("Decimal value too large to cast to primitive")
                        })
                    })?;

                    // TODO(connor): A lot of questionable stuff happening here, it would be good to
                    // either formally prove this is all correct or use more checked methods.

                    // Apply scale to get the actual value.
                    let actual_value = scaled_value as f64 / scale_factor as f64;

                    // Cast to target primitive type. Note that the `as` keyword does **MORE** than
                    // a simple bitcast / memory transmuation.
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "truncation is intentional - range checks happen after"
                    )]
                    let primitive_scalar = match ptype {
                        PType::U8 => {
                            let v = actual_value as u8;
                            if actual_value < 0.0 || actual_value > u8::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u8", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::U16 => {
                            let v = actual_value as u16;
                            if actual_value < 0.0 || actual_value > u16::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u16", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::U32 => {
                            let v = actual_value as u32;
                            if actual_value < 0.0 || actual_value > u32::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u32", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::U64 => {
                            let v = actual_value as u64;
                            if actual_value < 0.0 || actual_value > u64::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for u64", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::I8 => {
                            let v = actual_value as i8;
                            if actual_value < i8::MIN as f64 || actual_value > i8::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i8", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::I16 => {
                            let v = actual_value as i16;
                            if actual_value < i16::MIN as f64 || actual_value > i16::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i16", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::I32 => {
                            let v = actual_value as i32;
                            if actual_value < i32::MIN as f64 || actual_value > i32::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i32", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::I64 => {
                            let v = actual_value as i64;
                            if actual_value < i64::MIN as f64 || actual_value > i64::MAX as f64 {
                                vortex_bail!("Decimal value {} out of range for i64", actual_value);
                            }
                            Scalar::primitive(v, *nullability)
                        }
                        PType::F16 => {
                            use crate::dtype::half::f16;
                            Scalar::primitive(f16::from_f64(actual_value), *nullability)
                        }
                        PType::F32 => Scalar::primitive(actual_value as f32, *nullability),
                        PType::F64 => Scalar::primitive(actual_value, *nullability),
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

    /// Apply the (checked) operator to self and other using SQL-style null semantics.
    ///
    /// If the operation overflows, None is returned.
    ///
    /// If the types are incompatible (ignoring nullability and precision/scale), an error is returned.
    ///
    /// If either value is null, the result is null.
    ///
    /// The result will have the same decimal type (precision/scale) as `self`, and the result
    /// is checked to ensure it fits within the precision constraints.
    pub fn checked_binary_numeric(
        &self,
        other: &DecimalScalar<'a>,
        op: NumericOperator,
    ) -> Option<DecimalScalar<'a>> {
        // We could have ops between different types but need to add rules for type inference.
        if self.decimal_type != other.decimal_type {
            vortex_panic!(
                "decimal types must match: {} vs {}",
                self.decimal_type,
                other.decimal_type
            );
        }

        // Use the more nullable dtype as the result type
        let result_dtype = if self.dtype.is_nullable() {
            self.dtype
        } else {
            other.dtype
        };

        // Handle null cases using SQL semantics
        let result_value = match (self.decimal_value, other.decimal_value) {
            (None, _) | (_, None) => None,
            (Some(lhs), Some(rhs)) => {
                // Perform the operation
                let operation_result = match op {
                    NumericOperator::Add => lhs.checked_add(&rhs),
                    NumericOperator::Sub => lhs.checked_sub(&rhs),
                    NumericOperator::Mul => lhs.checked_mul(&rhs),
                    NumericOperator::Div => lhs.checked_div(&rhs),
                }?;

                // Check if the result fits within the precision constraints
                if operation_result.fits_in_precision(self.decimal_type) {
                    Some(operation_result)
                } else {
                    // Result exceeds precision, return None (overflow)
                    return None;
                }
            }
        };

        Some(DecimalScalar {
            dtype: result_dtype,
            decimal_type: self.decimal_type,
            decimal_value: result_value,
        })
    }
}

impl PartialEq for DecimalScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.decimal_value == other.decimal_value
    }
}

impl Eq for DecimalScalar<'_> {}

/// Ord is not implemented since it's undefined for different PTypes
impl PartialOrd for DecimalScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return None;
        }
        self.decimal_value.partial_cmp(&other.decimal_value)
    }
}

impl fmt::Display for DecimalScalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(&decimal_value) = self.decimal_value.as_ref() else {
            return write!(f, "null");
        };

        // Introduce some of the scale factors instead.
        match decimal_value {
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
}
