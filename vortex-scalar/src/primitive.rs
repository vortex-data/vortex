// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Sub};

use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub};
use vortex_dtype::half::f16;
use vortex_dtype::{
    DType, FromPrimitiveOrF16, NativePType, Nullability, PType, match_each_native_ptype,
};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err, vortex_panic};

use crate::pvalue::{CoercePValue, PValue};
use crate::{InnerScalarValue, Scalar, ScalarValue};

/// A scalar value representing a primitive type.
///
/// This type provides a view into a primitive scalar value of any primitive type
/// (integers, floats) with various bit widths.
#[derive(Debug, Clone, Copy, Hash)]
pub struct PrimitiveScalar<'a> {
    dtype: &'a DType,
    ptype: PType,
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
    pub fn try_new(dtype: &'a DType, value: &ScalarValue) -> VortexResult<Self> {
        let ptype = PType::try_from(dtype)?;

        // Read the serialized value into the correct PValue.
        // The serialized form may come back over the wire as e.g. any integer type.
        let pvalue = match_each_native_ptype!(ptype, |T| {
            value
                .as_pvalue()?
                .map(|pv| VortexResult::Ok(PValue::from(<T>::coerce(pv)?)))
                .transpose()?
        });

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

        self.pvalue.map(|pv| pv.cast::<T>())
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let ptype = PType::try_from(dtype)?;
        let pvalue = self
            .pvalue
            .vortex_expect("nullness handled in Scalar::cast");
        Ok(match_each_native_ptype!(ptype, |Q| {
            Scalar::primitive(
                pvalue
                    .cast_opt::<Q>()
                    .ok_or_else(|| vortex_err!("Cannot cast {} to {}", self.ptype, dtype))?,
                dtype.nullability(),
            )
        }))
    }

    /// Returns true if the scalar is nan.
    pub fn is_nan(&self) -> bool {
        self.pvalue.as_ref().is_some_and(|p| p.is_nan())
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
    /// # use vortex_dtype::{DType, PType};
    /// # use vortex_scalar::Scalar;
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
    /// # use vortex_dtype::{DType, PType};
    /// # use vortex_scalar::Scalar;
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

impl<'a> TryFrom<&'a Scalar> for PrimitiveScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        Self::try_new(value.dtype(), value.value())
    }
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

impl Scalar {
    /// Creates a new primitive scalar from a native value.
    pub fn primitive<T: NativePType + Into<PValue>>(value: T, nullability: Nullability) -> Self {
        Self::primitive_value(value.into(), T::PTYPE, nullability)
    }

    /// Create a PrimitiveScalar from a PValue.
    ///
    /// Note that an explicit PType is passed since any compatible PValue may be used as the value
    /// for a primitive type.
    pub fn primitive_value(value: PValue, ptype: PType, nullability: Nullability) -> Self {
        Self::new(
            DType::Primitive(ptype, nullability),
            ScalarValue(InnerScalarValue::Primitive(value)),
        )
    }

    /// Reinterprets the bytes of this scalar as a different primitive type.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a primitive type or if the types have different byte widths.
    pub fn reinterpret_cast(&self, ptype: PType) -> Self {
        let primitive = PrimitiveScalar::try_from(self).unwrap_or_else(|e| {
            vortex_panic!(
                e,
                "Failed to reinterpret cast {} to {}",
                self.dtype(),
                ptype
            )
        });
        if primitive.ptype() == ptype {
            return self.clone();
        }

        assert_eq!(
            primitive.ptype().byte_width(),
            ptype.byte_width(),
            "can't reinterpret cast between integers of two different widths"
        );

        Scalar::new(
            DType::Primitive(ptype, self.dtype().nullability()),
            primitive
                .pvalue
                .map(|p| p.reinterpret_cast(ptype))
                .map(|x| ScalarValue(InnerScalarValue::Primitive(x)))
                .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null)),
        )
    }
}

macro_rules! primitive_scalar {
    ($T:ty) => {
        impl TryFrom<&Scalar> for $T {
            type Error = VortexError;

            fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
                <Option<$T>>::try_from(value)?
                    .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
            }
        }

        impl TryFrom<Scalar> for $T {
            type Error = VortexError;

            fn try_from(value: Scalar) -> Result<Self, Self::Error> {
                <$T>::try_from(&value)
            }
        }

        impl TryFrom<&Scalar> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
                Ok(PrimitiveScalar::try_from(value)?.typed_value::<$T>())
            }
        }

        impl TryFrom<Scalar> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: Scalar) -> Result<Self, Self::Error> {
                <Option<$T>>::try_from(&value)
            }
        }

        impl From<$T> for Scalar {
            fn from(value: $T) -> Self {
                Scalar::new(
                    DType::Primitive(<$T>::PTYPE, Nullability::NonNullable),
                    ScalarValue(InnerScalarValue::Primitive(value.into())),
                )
            }
        }

        impl From<$T> for ScalarValue {
            fn from(value: $T) -> Self {
                ScalarValue(InnerScalarValue::Primitive(value.into()))
            }
        }
    };
}

primitive_scalar!(u8);
primitive_scalar!(u16);
primitive_scalar!(u32);
primitive_scalar!(u64);
primitive_scalar!(i8);
primitive_scalar!(i16);
primitive_scalar!(i32);
primitive_scalar!(i64);
primitive_scalar!(f16);
primitive_scalar!(f32);
primitive_scalar!(f64);

/// Read a scalar as usize. For usize only, we implicitly cast for better ergonomics.
impl TryFrom<&Scalar> for usize {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
        let prim = PrimitiveScalar::try_from(value)?
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("cannot convert Null to usize"))?;
        Ok(usize::try_from(prim)?)
    }
}

impl TryFrom<&Scalar> for Option<usize> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
        Ok(PrimitiveScalar::try_from(value)?
            .as_::<u64>()
            .map(usize::try_from)
            .transpose()?)
    }
}

/// Read a scalar as usize. For usize only, we implicitly cast for better ergonomics.
impl From<usize> for Scalar {
    fn from(value: usize) -> Self {
        Scalar::primitive(value as u64, Nullability::NonNullable)
    }
}

impl From<PValue> for ScalarValue {
    fn from(value: PValue) -> Self {
        ScalarValue(InnerScalarValue::Primitive(value))
    }
}

/// Read a scalar as usize. For usize only, we implicitly cast for better ergonomics.
impl From<usize> for ScalarValue {
    fn from(value: usize) -> Self {
        ScalarValue(InnerScalarValue::Primitive((value as u64).into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Binary element-wise operations on two arrays or two scalars.
pub enum NumericOperator {
    /// Binary element-wise addition of two arrays or of two scalars.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    Add,
    /// Binary element-wise subtraction of two arrays or of two scalars.
    Sub,
    /// Same as [NumericOperator::Sub] but with the parameters flipped: `right - left`.
    RSub,
    /// Binary element-wise multiplication of two arrays or of two scalars.
    Mul,
    /// Binary element-wise division of two arrays or of two scalars.
    Div,
    /// Same as [NumericOperator::Div] but with the parameters flipped: `right / left`.
    RDiv,
    // Missing from arrow-rs:
    // Min,
    // Max,
    // Pow,
}

impl Display for NumericOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl NumericOperator {
    /// Returns the operator with swapped operands (e.g., Sub becomes RSub).
    pub fn swap(self) -> Self {
        match self {
            NumericOperator::Add => NumericOperator::Add,
            NumericOperator::Sub => NumericOperator::RSub,
            NumericOperator::RSub => NumericOperator::Sub,
            NumericOperator::Mul => NumericOperator::Mul,
            NumericOperator::Div => NumericOperator::RDiv,
            NumericOperator::RDiv => NumericOperator::Div,
        }
    }
}

impl<'a> PrimitiveScalar<'a> {
    /// Apply the (checked) operator to self and other using SQL-style null semantics.
    ///
    /// If the operation overflows, Ok(None) is returned.
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
                        NumericOperator::RSub => Some(rhs - lhs),
                        NumericOperator::Mul => Some(lhs * rhs),
                        NumericOperator::Div => Some(lhs / rhs),
                        NumericOperator::RDiv => Some(rhs / lhs),
                    }
                };
                Some(Self { dtype: result_dtype, ptype, pvalue: value_or_null.map(PValue::from) })
            }
        )
    }

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
                NumericOperator::RSub => rhs.checked_sub(&lhs).map(Some),
                NumericOperator::Mul => lhs.checked_mul(&rhs).map(Some),
                NumericOperator::Div => lhs.checked_div(&rhs).map(Some),
                NumericOperator::RDiv => rhs.checked_div(&lhs).map(Some),
            },
        };

        value_or_null_or_overflow.map(|value_or_null| Self {
            dtype: result_dtype,
            ptype,
            pvalue: value_or_null.map(PValue::from),
        })
    }
}

#[cfg(test)]
mod tests {
    use num_traits::CheckedSub;
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexExpect;

    use crate::{InnerScalarValue, PValue, PrimitiveScalar, ScalarValue};

    #[test]
    fn test_integer_subtract() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let p_scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(5))),
        )
        .unwrap();
        let p_scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(4))),
        )
        .unwrap();
        let pscalar_or_overflow = p_scalar1.checked_sub(&p_scalar2);
        let value_or_null_or_type_error = pscalar_or_overflow.unwrap().as_::<i32>();
        assert_eq!(value_or_null_or_type_error.unwrap(), 1);

        assert_eq!((p_scalar1 - p_scalar2).as_::<i32>().unwrap(), 1);
    }

    #[test]
    #[should_panic(expected = "PrimitiveScalar subtract: overflow or underflow")]
    fn test_integer_subtract_overflow() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let p_scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(i32::MIN))),
        )
        .unwrap();
        let p_scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(i32::MAX))),
        )
        .unwrap();
        let _ = p_scalar1 - p_scalar2;
    }

    #[test]
    fn test_float_subtract() {
        let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let p_scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::F32(1.99f32))),
        )
        .unwrap();
        let p_scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::F32(1.0f32))),
        )
        .unwrap();
        let pscalar_or_overflow = p_scalar1.checked_sub(&p_scalar2).unwrap();
        let value_or_null_or_type_error = pscalar_or_overflow.as_::<f32>();
        assert_eq!(value_or_null_or_type_error.unwrap(), 0.99f32);

        assert_eq!((p_scalar1 - p_scalar2).as_::<f32>().unwrap(), 0.99f32);
    }

    #[test]
    fn test_primitive_scalar_equality() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(42))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(42))),
        )
        .unwrap();
        let scalar3 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(43))),
        )
        .unwrap();

        assert_eq!(scalar1, scalar2);
        assert_ne!(scalar1, scalar3);
    }

    #[test]
    fn test_primitive_scalar_partial_ord() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(10))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(20))),
        )
        .unwrap();

        assert!(scalar1 < scalar2);
        assert!(scalar2 > scalar1);
        assert_eq!(
            scalar1.partial_cmp(&scalar1),
            Some(std::cmp::Ordering::Equal)
        );
    }

    #[test]
    fn test_primitive_scalar_null_handling() {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let null_scalar =
            PrimitiveScalar::try_new(&dtype, &ScalarValue(InnerScalarValue::Null)).unwrap();

        assert_eq!(null_scalar.pvalue(), None);
        assert_eq!(null_scalar.typed_value::<i32>(), None);
    }

    #[test]
    fn test_typed_value_correct_type() {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let scalar = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::F64(3.5))),
        )
        .unwrap();

        assert_eq!(scalar.typed_value::<f64>(), Some(3.5));
    }

    #[test]
    #[should_panic(expected = "Attempting to read")]
    fn test_typed_value_wrong_type() {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let scalar = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::F64(3.5))),
        )
        .unwrap();

        let _ = scalar.typed_value::<i32>();
    }

    #[rstest]
    #[case(PType::I8, 127i32, PType::I16, true)]
    #[case(PType::I8, 127i32, PType::I32, true)]
    #[case(PType::I8, 127i32, PType::I64, true)]
    #[case(PType::U8, 255i32, PType::U16, true)]
    #[case(PType::U8, 255i32, PType::U32, true)]
    #[case(PType::I32, 42i32, PType::F32, true)]
    #[case(PType::I32, 42i32, PType::F64, true)]
    // Overflow cases
    #[case(PType::I32, 300i32, PType::U8, false)]
    #[case(PType::I32, -1i32, PType::U32, false)]
    #[case(PType::I32, 256i32, PType::I8, false)]
    #[case(PType::U16, 65535i32, PType::I8, false)]
    fn test_primitive_cast(
        #[case] source_type: PType,
        #[case] source_value: i32,
        #[case] target_type: PType,
        #[case] should_succeed: bool,
    ) {
        let source_pvalue = match source_type {
            PType::I8 => PValue::I8(i8::try_from(source_value).vortex_expect("cannot cast")),
            PType::U8 => PValue::U8(u8::try_from(source_value).vortex_expect("cannot cast")),
            PType::U16 => PValue::U16(u16::try_from(source_value).vortex_expect("cannot cast")),
            PType::I32 => PValue::I32(source_value),
            _ => unreachable!("Test case uses unexpected source type"),
        };

        let dtype = DType::Primitive(source_type, Nullability::NonNullable);
        let scalar = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(source_pvalue)),
        )
        .unwrap();

        let target_dtype = DType::Primitive(target_type, Nullability::NonNullable);
        let result = scalar.cast(&target_dtype);

        if should_succeed {
            assert!(
                result.is_ok(),
                "Cast from {:?} to {:?} should succeed",
                source_type,
                target_type
            );
        } else {
            assert!(
                result.is_err(),
                "Cast from {:?} to {:?} should fail due to overflow",
                source_type,
                target_type
            );
        }
    }

    #[test]
    fn test_as_conversion_success() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(42))),
        )
        .unwrap();

        assert_eq!(scalar.as_::<i64>(), Some(42i64));
        assert_eq!(scalar.as_::<f64>(), Some(42.0));
    }

    #[test]
    fn test_as_conversion_overflow() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(-1))),
        )
        .unwrap();

        // Converting -1 to u32 should fail
        let result = scalar.as_opt::<u32>();
        assert!(result.is_none());
    }

    #[test]
    fn test_as_conversion_null() {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let scalar =
            PrimitiveScalar::try_new(&dtype, &ScalarValue(InnerScalarValue::Null)).unwrap();

        assert_eq!(scalar.as_::<i32>(), None);
        assert_eq!(scalar.as_::<f64>(), None);
    }

    #[test]
    fn test_numeric_operator_swap() {
        use crate::primitive::NumericOperator;

        assert_eq!(NumericOperator::Add.swap(), NumericOperator::Add);
        assert_eq!(NumericOperator::Sub.swap(), NumericOperator::RSub);
        assert_eq!(NumericOperator::RSub.swap(), NumericOperator::Sub);
        assert_eq!(NumericOperator::Mul.swap(), NumericOperator::Mul);
        assert_eq!(NumericOperator::Div.swap(), NumericOperator::RDiv);
        assert_eq!(NumericOperator::RDiv.swap(), NumericOperator::Div);
    }

    #[test]
    fn test_checked_binary_numeric_add() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(10))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(20))),
        )
        .unwrap();

        let result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::Add)
            .unwrap();
        assert_eq!(result.typed_value::<i32>(), Some(30));
    }

    #[test]
    fn test_checked_binary_numeric_overflow() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(i32::MAX))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(1))),
        )
        .unwrap();

        // Add should overflow and return None
        let result = scalar1.checked_binary_numeric(&scalar2, NumericOperator::Add);
        assert!(result.is_none());
    }

    #[test]
    fn test_checked_binary_numeric_with_null() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(10))),
        )
        .unwrap();
        let null_scalar =
            PrimitiveScalar::try_new(&dtype, &ScalarValue(InnerScalarValue::Null)).unwrap();

        // Operation with null should return null
        let result = scalar1
            .checked_binary_numeric(&null_scalar, NumericOperator::Add)
            .unwrap();
        assert_eq!(result.pvalue(), None);
    }

    #[test]
    fn test_checked_binary_numeric_mul() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(5))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(6))),
        )
        .unwrap();

        let result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::Mul)
            .unwrap();
        assert_eq!(result.typed_value::<i32>(), Some(30));
    }

    #[test]
    fn test_checked_binary_numeric_div() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(20))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(4))),
        )
        .unwrap();

        let result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::Div)
            .unwrap();
        assert_eq!(result.typed_value::<i32>(), Some(5));
    }

    #[test]
    fn test_checked_binary_numeric_rdiv() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(4))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(20))),
        )
        .unwrap();

        // RDiv means right / left, so 20 / 4 = 5
        let result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::RDiv)
            .unwrap();
        assert_eq!(result.typed_value::<i32>(), Some(5));
    }

    #[test]
    fn test_checked_binary_numeric_div_by_zero() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(10))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(0))),
        )
        .unwrap();

        // Division by zero should return None for integers
        let result = scalar1.checked_binary_numeric(&scalar2, NumericOperator::Div);
        assert!(result.is_none());
    }

    #[test]
    fn test_checked_binary_numeric_float_ops() {
        use crate::primitive::NumericOperator;

        let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let scalar1 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::F32(10.0))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::F32(2.5))),
        )
        .unwrap();

        // Test all operations with floats
        let add_result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::Add)
            .unwrap();
        assert_eq!(add_result.typed_value::<f32>(), Some(12.5));

        let sub_result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::Sub)
            .unwrap();
        assert_eq!(sub_result.typed_value::<f32>(), Some(7.5));

        let mul_result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::Mul)
            .unwrap();
        assert_eq!(mul_result.typed_value::<f32>(), Some(25.0));

        let div_result = scalar1
            .checked_binary_numeric(&scalar2, NumericOperator::Div)
            .unwrap();
        assert_eq!(div_result.typed_value::<f32>(), Some(4.0));
    }

    #[test]
    fn test_from_primitive_or_f16() {
        use vortex_dtype::half::f16;

        use crate::primitive::FromPrimitiveOrF16;

        // Test f16 to f32 conversion
        let f16_val = f16::from_f32(3.5);
        assert!(f32::from_f16(f16_val).is_some());

        // Test f16 to f64 conversion
        assert!(f64::from_f16(f16_val).is_some());

        // Test PValue::F16(f16) to integer conversion (should fail)
        assert!(i32::try_from(PValue::from(f16_val)).is_err());
        assert!(u32::try_from(PValue::from(f16_val)).is_err());
    }

    #[test]
    fn test_partial_ord_different_types() {
        let dtype1 = DType::Primitive(PType::I32, Nullability::NonNullable);
        let dtype2 = DType::Primitive(PType::F32, Nullability::NonNullable);

        let scalar1 = PrimitiveScalar::try_new(
            &dtype1,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(10))),
        )
        .unwrap();
        let scalar2 = PrimitiveScalar::try_new(
            &dtype2,
            &ScalarValue(InnerScalarValue::Primitive(PValue::F32(10.0))),
        )
        .unwrap();

        // Different types should not be comparable
        assert_eq!(scalar1.partial_cmp(&scalar2), None);
    }

    #[test]
    fn test_scalar_value_from_usize() {
        let value: ScalarValue = 42usize.into();
        assert!(matches!(
            value.0,
            InnerScalarValue::Primitive(PValue::U64(42))
        ));
    }

    #[test]
    fn test_getters() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scalar = PrimitiveScalar::try_new(
            &dtype,
            &ScalarValue(InnerScalarValue::Primitive(PValue::I32(42))),
        )
        .unwrap();

        assert_eq!(scalar.dtype(), &dtype);
        assert_eq!(scalar.ptype(), PType::I32);
        assert_eq!(scalar.pvalue(), Some(PValue::I32(42)));
    }
}
