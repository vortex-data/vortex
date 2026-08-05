// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`DecimalScalar`] typed view implementation.

use std::cmp::Ordering;
use std::fmt;

use num_traits::ToPrimitive as NumToPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::arithmetic::checked_decimal_numeric;
use super::arithmetic::decimal_numeric_result_dtype;
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
    /// Both operands must share the same decimal type `(p, s)`. The result type follows Arrow's
    /// decimal arithmetic rules, so it is generally *wider* than the operands, and the result is
    /// therefore an owned [`Scalar`] rather than a view:
    ///
    /// | operator | result precision | result scale |
    /// | -------- | ---------------- | ------------ |
    /// | Add, Sub | `p + 1`          | `s`          |
    /// | Mul      | `2p + 1`         | `2s`         |
    /// | Div      | `p + s + 4`      | `s + 4`      |
    ///
    /// Precision saturates at the maximum decimal precision. Mul is exact — the doubled scale
    /// leaves the raw product of the stored integers correctly scaled — while Div truncates
    /// toward zero.
    ///
    /// If either value is null, the result is null.
    ///
    /// `Ok(None)` means the operation overflowed the result precision or divided by zero. Note
    /// that the array kernels raise an error in that situation rather than yielding null.
    ///
    /// # Errors
    ///
    /// Returns an error if the operands have different decimal types, or if the operation has no
    /// valid result type: a Mul whose scale would exceed the maximum scale, or a Div whose
    /// precision would fall outside the legal range.
    pub fn checked_binary_numeric(
        &self,
        other: &DecimalScalar<'_>,
        op: NumericOperator,
    ) -> VortexResult<Option<Scalar>> {
        // We could have ops between different types but need to add rules for type inference.
        if self.decimal_type != other.decimal_type {
            vortex_bail!(
                "decimal types must match: {} vs {}",
                self.decimal_type,
                other.decimal_type
            );
        }

        let result_decimal_type = decimal_numeric_result_dtype(self.decimal_type, op)?;
        let nullability = self.dtype.nullability() | other.dtype.nullability();
        let result_dtype = DType::Decimal(result_decimal_type, nullability);

        // Handle null cases using SQL semantics.
        let (Some(lhs), Some(rhs)) = (self.decimal_value, other.decimal_value) else {
            return Ok(Some(Scalar::null(result_dtype.as_nullable())));
        };

        Ok(
            checked_decimal_numeric(lhs, rhs, self.decimal_type, result_decimal_type, op)
                .map(|value| Scalar::decimal(value, result_decimal_type, nullability)),
        )
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
