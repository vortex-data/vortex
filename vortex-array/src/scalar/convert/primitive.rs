// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions for [`PrimitiveScalar`]s.

use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::scalar::PValue;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

// TODO(connor): Ideally we remove this.
impl From<PrimitiveScalar<'_>> for Scalar {
    fn from(ps: PrimitiveScalar<'_>) -> Self {
        // SAFETY: `PrimitiveScalar` is already a valid `Scalar`.
        unsafe {
            Scalar::new_unchecked(ps.dtype().clone(), ps.pvalue().map(ScalarValue::Primitive))
        }
    }
}

impl From<PValue> for ScalarValue {
    fn from(value: PValue) -> Self {
        ScalarValue::Primitive(value)
    }
}

/// Implements scalar conversions for a primitive type.
macro_rules! primitive_scalar {
    ($T:ty) => {
        ////////////////////////////////////////////////////////////////////////////////////////////
        // `Scalar` INTO $T conversions.
        ////////////////////////////////////////////////////////////////////////////////////////////

        /// Fallible conversion from a [`ScalarValue`] into an `T`.
        ///
        /// # Errors
        ///
        /// Returns an error if unable to convert the scalar value into the target type,
        impl TryFrom<&ScalarValue> for $T {
            type Error = VortexError;

            fn try_from(value: &ScalarValue) -> VortexResult<Self> {
                value.as_primitive().cast::<$T>()
            }
        }

        /// Fallible conversion from a [`Scalar`] into an `T`.
        ///
        /// # Errors
        ///
        /// Returns an error if unable to convert the scalar into the target type, or if the
        /// `Scalar` itself is null.
        impl TryFrom<&Scalar> for $T {
            type Error = VortexError;

            fn try_from(value: &Scalar) -> VortexResult<Self> {
                match value.value() {
                    Some(ScalarValue::Primitive(pv)) => pv.cast::<$T>(),
                    Some(_) => Err(vortex_err!(
                        "Expected primitive scalar, found {}",
                        value.dtype()
                    )),
                    None => Err(vortex_err!("Can't extract present value from null scalar")),
                }
            }
        }

        /// Fallible conversion from a [`Scalar`] into an `Option<T>`.
        ///
        /// # Errors
        ///
        /// Returns an error if the [`Scalar`] is not primitive, or if it is unable to convert the
        /// [`Scalar`] into the target type.
        impl TryFrom<&Scalar> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: &Scalar) -> VortexResult<Self> {
                let primitive_scalar = PrimitiveScalar::try_new(value.dtype(), value.value())?;
                primitive_scalar.try_typed_value::<$T>()
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        // `Scalar` FROM $T conversions.
        ////////////////////////////////////////////////////////////////////////////////////////////

        /// `Into<ScalarValue>` for T.
        impl From<$T> for ScalarValue {
            fn from(value: $T) -> Self {
                ScalarValue::Primitive(value.into())
            }
        }

        /// Non-nullable `Into<Scalar>` implementation for T.
        impl From<$T> for Scalar {
            fn from(value: $T) -> Self {
                Scalar::try_new(
                    DType::Primitive(<$T>::PTYPE, Nullability::NonNullable),
                    Some(ScalarValue::Primitive(value.into())),
                )
                .vortex_expect(
                    "somehow unable to construct a primitive `Scalar` from a native type",
                )
            }
        }

        /// Nullable `Into<Scalar>` implementation for T.
        impl From<Option<$T>> for Scalar {
            fn from(value: Option<$T>) -> Self {
                Scalar::try_new(
                    DType::Primitive(<$T>::PTYPE, Nullability::Nullable),
                    value.map(|value| ScalarValue::Primitive(value.into())),
                )
                .vortex_expect(
                    "somehow unable to construct a primitive `Scalar` from a native type",
                )
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

////////////////////////////////////////////////////////////////////////////////////////////
// `Scalar` <---> `usize` conversions.
////////////////////////////////////////////////////////////////////////////////////////////

// NB: We cast `usize` to `u64` (which should always succeed) for better ergonomics.

/// Fallible conversion from a [`ScalarValue`] into an `T`.
///
/// # Errors
///
/// Returns an error if unable to convert the scalar value into the target type,
impl TryFrom<&ScalarValue> for usize {
    type Error = VortexError;

    fn try_from(value: &ScalarValue) -> VortexResult<Self> {
        let val = value.as_primitive().cast::<u64>()?;
        Ok(usize::try_from(val)?)
    }
}

impl TryFrom<&Scalar> for usize {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
        let prim_scalar = value
            .as_primitive_opt()
            .ok_or_else(|| vortex_err!("Expected primitive scalar, found {}", value.dtype()))?;

        let prim = prim_scalar
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("cannot convert Null to usize"))?;

        Ok(usize::try_from(prim)?)
    }
}

impl TryFrom<&Scalar> for Option<usize> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
        let prim_scalar = value
            .as_primitive_opt()
            .ok_or_else(|| vortex_err!("Expected primitive scalar, found {}", value.dtype()))?;

        Ok(prim_scalar.as_::<u64>().map(usize::try_from).transpose()?)
    }
}

impl From<usize> for ScalarValue {
    fn from(value: usize) -> Self {
        ScalarValue::Primitive((value as u64).into())
    }
}

impl From<usize> for Scalar {
    fn from(value: usize) -> Self {
        Scalar::try_new(
            DType::Primitive(PType::U64, Nullability::NonNullable),
            Some(ScalarValue::Primitive((value as u64).into())),
        )
        .vortex_expect("somehow unable to construct a primitive `Scalar` from a native type")
    }
}

impl From<Option<usize>> for Scalar {
    fn from(value: Option<usize>) -> Self {
        Scalar::try_new(
            DType::Primitive(PType::U64, Nullability::Nullable),
            value.map(|value| ScalarValue::Primitive((value as u64).into())),
        )
        .vortex_expect("somehow unable to construct a primitive `Scalar` from a native type")
    }
}
