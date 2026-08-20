// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PrimitiveScalar`] typed view implementation.

use std::any::type_name;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Add;
use std::ops::Sub;

use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use super::pvalue::CoercePValue;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FromPrimitiveOrF16;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_native_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing a primitive type.
///
/// This type provides a view into a primitive scalar value of any primitive type
/// (integers, floats) with various bit widths.
#[derive(Debug, Clone, Copy, Hash)]
pub struct PrimitiveScalar<'a> {
    /// The data type of this scalar.
    dtype: &'a DType,
    /// The primitive type.
    ptype: PType,
    /// The primitive value, or [`None`] if null.
    pvalue: Option<PValue>,
}

impl Display for PrimitiveScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.pvalue {
            None => write!(f, "null"),
            Some(pv) => write!(f, "{pv}"),
        }
    }
}

impl PartialEq for PrimitiveScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.pvalue == other.pvalue
    }
}

impl Eq for PrimitiveScalar<'_> {}

/// Ord is not implemented since it's undefined for different PTypes
impl PartialOrd for PrimitiveScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return None;
        }
        self.pvalue.partial_cmp(&other.pvalue)
    }
}

impl<'a> PrimitiveScalar<'a> {
    /// Creates a new primitive scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a primitive type or if the value
    /// cannot be converted to the expected primitive type.
    pub fn try_new(dtype: &'a DType, value: Option<&ScalarValue>) -> VortexResult<Self> {
        let ptype = PType::try_from(dtype)?;

        // Read the serialized value into the correct PValue.
        // The serialized form may come back over the wire as e.g. any integer type.
        let pvalue = match value {
            None => None,
            Some(v) => {
                let pv = v.as_primitive();
                match_each_native_ptype!(ptype, |T| { Some(PValue::from(<T>::coerce(*pv)?)) })
            }
        };

        Ok(Self {
            dtype,
            ptype,
            pvalue,
        })
    }

    /// Returns the data type of this primitive scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the primitive type of this scalar.
    #[inline]
    pub fn ptype(&self) -> PType {
        self.ptype
    }

    /// Returns the primitive value, or None if null.
    #[inline]
    pub fn pvalue(&self) -> Option<PValue> {
        self.pvalue
    }

    // TODO(connor): This should probably be deprecated for `try_typed_value`.
    /// Returns the value as a specific native primitive type.
    ///
    /// Returns `None` if the scalar is null, otherwise returns `Some(value)` where
    /// value is the underlying primitive value cast to the requested type `T`.
    ///
    /// # Panics
    ///
    /// Panics if the primitive type of this scalar does not match the requested type.
    pub fn typed_value<T: NativePType>(&self) -> Option<T> {
        assert_eq!(
            self.ptype,
            T::PTYPE,
            "Attempting to read {} scalar as {}",
            self.ptype,
            T::PTYPE
        );

        self.pvalue.and_then(|pv| pv.cast::<T>().ok())
    }

    /// Returns the value as a specific native primitive type.
    ///
    /// Returns `Ok(None)` if the scalar is null, otherwise returns `Ok(Some(value))` where
    /// value is the underlying primitive value cast to the requested type `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the primitive type of this scalar does not match the requested type.
    pub fn try_typed_value<T: NativePType>(&self) -> VortexResult<Option<T>> {
        vortex_ensure!(
            self.ptype == T::PTYPE,
            "Attempting to read {} scalar as {}",
            self.ptype,
            T::PTYPE
        );

        if let Some(pv) = self.pvalue {
            pv.cast::<T>().map(Some)
        } else {
            Ok(None)
        }
    }

    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let pvalue = self
            .pvalue
            .vortex_expect("nullness handled in Scalar::cast");

        match dtype {
            DType::Primitive(ptype, nullability) => Ok(match_each_native_ptype!(*ptype, |Q| {
                Scalar::primitive(pvalue.cast::<Q>()?, *nullability)
            })),
            DType::Decimal(decimal_dtype, nullability) => Ok(Scalar::decimal(
                pvalue_to_decimal(pvalue, *decimal_dtype)?,
                *decimal_dtype,
                *nullability,
            )),
            _ => vortex_bail!("Cannot cast primitive scalar to {dtype}"),
        }
    }

    /// Returns true if the scalar is nan.
    pub fn is_nan(&self) -> bool {
        self.pvalue.as_ref().is_some_and(|p| p.is_nan())
    }

    /// Returns whether this decimal value is zero, or `None` if null.
    pub fn is_zero(&self) -> Option<bool> {
        self.pvalue.map(|v| v.is_zero())
    }

    /// Attempts to extract the primitive value as the given type.
    ///
    /// # Errors
    ///
    /// Panics if the cast fails due to overflow or type incompatibility. See also
    /// `as_opt` for the checked version that does not panic.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// # use vortex_array::dtype::{DType, PType};
    /// # use vortex_array::scalar::Scalar;
    /// let wide = Scalar::primitive(1000i32, false.into());
    ///
    /// // This succeeds
    /// let narrow = wide.as_primitive().as_::<i16>();
    /// assert_eq!(narrow, Some(1000i16));
    ///
    /// // This also succeeds
    /// let null = Scalar::null(DType::Primitive(PType::I16, true.into()));
    /// assert_eq!(null.as_primitive().as_::<i8>(), None);
    ///
    /// // This will panic, because 1000 does not fit in i8
    /// wide.as_primitive().as_::<i8>();
    /// ```
    pub fn as_<T: FromPrimitiveOrF16>(&self) -> Option<T> {
        self.as_opt::<T>().unwrap_or_else(|| {
            vortex_panic!(
                "cast {} to {}: value out of range",
                self.ptype,
                type_name::<T>()
            )
        })
    }

    /// Returns the inner value cast to the desired type.
    ///
    /// If the cast fails, `None` is returned. If the scalar represents a null, `Some(None)`
    /// is returned. Otherwise, `Some(Some(T))` is returned for a successful non-null conversion.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_array::dtype::{DType, PType};
    /// # use vortex_array::scalar::Scalar;
    ///
    /// // Non-null values
    /// let scalar = Scalar::primitive(100i32, false.into());
    /// let primitive = scalar.as_primitive();
    /// assert_eq!(primitive.as_opt::<i8>(), Some(Some(100i8)));
    ///
    /// // Null value
    /// let scalar = Scalar::null(DType::Primitive(PType::I32, true.into()));
    /// let primitive = scalar.as_primitive();
    /// assert_eq!(primitive.as_opt::<i8>(), Some(None));
    ///
    /// // Failed conversion: 1000 cannot fit in an i8
    /// let scalar = Scalar::primitive(1000i32, false.into());
    /// let primitive = scalar.as_primitive();
    /// assert_eq!(primitive.as_opt::<i8>(), None);
    /// ```
    pub fn as_opt<T: FromPrimitiveOrF16>(&self) -> Option<Option<T>> {
        if let Some(pv) = self.pvalue {
            match pv {
                PValue::U8(v) => T::from_u8(v),
                PValue::U16(v) => T::from_u16(v),
                PValue::U32(v) => T::from_u32(v),
                PValue::U64(v) => T::from_u64(v),
                PValue::I8(v) => T::from_i8(v),
                PValue::I16(v) => T::from_i16(v),
                PValue::I32(v) => T::from_i32(v),
                PValue::I64(v) => T::from_i64(v),
                PValue::F16(v) => T::from_f16(v),
                PValue::F32(v) => T::from_f32(v),
                PValue::F64(v) => T::from_f64(v),
            }
            .map(Some)
        } else {
            Some(None)
        }
    }
}

fn pvalue_to_decimal(pvalue: PValue, decimal_dtype: DecimalDType) -> VortexResult<DecimalValue> {
    let value = match pvalue {
        PValue::U8(v) => i256::from_i128(i128::from(v)),
        PValue::U16(v) => i256::from_i128(i128::from(v)),
        PValue::U32(v) => i256::from_i128(i128::from(v)),
        PValue::U64(v) => i256::from_i128(i128::from(v)),
        PValue::I8(v) => i256::from_i128(i128::from(v)),
        PValue::I16(v) => i256::from_i128(i128::from(v)),
        PValue::I32(v) => i256::from_i128(i128::from(v)),
        PValue::I64(v) => i256::from_i128(i128::from(v)),
        PValue::F16(_) | PValue::F32(_) | PValue::F64(_) => {
            vortex_bail!("Cannot cast floating primitive {pvalue} to decimal {decimal_dtype}")
        }
    };

    let scaled = DecimalValue::rescale_i256(value, 0, decimal_dtype.scale())?;
    DecimalValue::try_from_i256(scaled, decimal_dtype)
}

impl Sub for PrimitiveScalar<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(&rhs)
            .vortex_expect("PrimitiveScalar subtract: overflow or underflow")
    }
}

impl CheckedSub for PrimitiveScalar<'_> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        self.checked_binary_numeric(rhs, NumericOperator::Sub)
    }
}

impl Add for PrimitiveScalar<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(&rhs)
            .vortex_expect("PrimitiveScalar add: overflow or underflow")
    }
}

impl CheckedAdd for PrimitiveScalar<'_> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        self.checked_binary_numeric(rhs, NumericOperator::Add)
    }
}

impl<'a> PrimitiveScalar<'a> {
    /// Apply the (checked) operator to self and other using SQL-style null semantics.
    ///
    /// If the operation overflows, `None` is returned (for integral types only).
    ///
    /// Note: Floating-point operations cannot overflow in the traditional sense.
    /// Instead, they may return `Some(Inf)` or `Some(NaN)` for operations that
    /// would overflow or are undefined (e.g., `0.0 / 0.0`).
    ///
    /// If the types are incompatible (ignoring nullability), an error is returned.
    ///
    /// If either value is null, the result is null.
    pub fn checked_binary_numeric(
        &self,
        other: &PrimitiveScalar<'a>,
        op: NumericOperator,
    ) -> Option<PrimitiveScalar<'a>> {
        if !self.dtype().eq_ignore_nullability(other.dtype()) {
            vortex_panic!("types must match: {} {}", self.dtype(), other.dtype());
        }
        let result_dtype = if self.dtype().is_nullable() {
            self.dtype()
        } else {
            other.dtype()
        };
        let ptype = self.ptype();

        match_each_native_ptype!(
            self.ptype(),
            integral: |P| {
                self.checked_integral_numeric_operator::<P>(other, result_dtype, ptype, op)
            },
            floating: |P| {
                let lhs = self.typed_value::<P>();
                let rhs = other.typed_value::<P>();
                let value_or_null = match (lhs, rhs) {
                    (_, None) | (None, _) => None,
                    (Some(lhs), Some(rhs)) => match op {
                        NumericOperator::Add => Some(lhs + rhs),
                        NumericOperator::Sub => Some(lhs - rhs),
                        NumericOperator::Mul => Some(lhs * rhs),
                        NumericOperator::Div => Some(lhs / rhs),
                    }
                };
                Some(Self { dtype: result_dtype, ptype, pvalue: value_or_null.map(PValue::from) })
            }
        )
    }

    /// Applies a checked arithmetic operation between two integral primitive scalars.
    fn checked_integral_numeric_operator<
        P: NativePType
            + TryFrom<PValue, Error = VortexError>
            + CheckedSub
            + CheckedAdd
            + CheckedMul
            + CheckedDiv,
    >(
        &self,
        other: &PrimitiveScalar<'a>,
        result_dtype: &'a DType,
        ptype: PType,
        op: NumericOperator,
    ) -> Option<PrimitiveScalar<'a>>
    where
        PValue: From<P>,
    {
        let lhs = self.typed_value::<P>();
        let rhs = other.typed_value::<P>();
        let value_or_null_or_overflow = match (lhs, rhs) {
            (_, None) | (None, _) => Some(None),
            (Some(lhs), Some(rhs)) => match op {
                NumericOperator::Add => lhs.checked_add(&rhs).map(Some),
                NumericOperator::Sub => lhs.checked_sub(&rhs).map(Some),
                NumericOperator::Mul => lhs.checked_mul(&rhs).map(Some),
                NumericOperator::Div => lhs.checked_div(&rhs).map(Some),
            },
        };

        value_or_null_or_overflow.map(|value_or_null| Self {
            dtype: result_dtype,
            ptype,
            pvalue: value_or_null.map(PValue::from),
        })
    }
}
