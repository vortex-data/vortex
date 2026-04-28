// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions from an assortment of types into scalars.

use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDType;
use crate::dtype::Nullability;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

// NB: Unfortunately, we cannot do a blanket implementation of `From<Option<T>> for Scalar` because
// we have no easy way of getting a `DType` from just a `T`. Maybe in the future it would be nice if
// we had that as a private trait impl...

/// Generates the three impl blocks for converting a type into [`ScalarValue`] and [`Scalar`]:
/// - `From<$ty> for ScalarValue`
/// - `From<$ty> for Scalar` (non-nullable)
/// - `From<Option<$ty>> for Scalar` (nullable)
macro_rules! impl_into_scalar {
    ($ty:ty, $variant:ident, | $value:ident | $convert:expr) => {
        impl From<$ty> for ScalarValue {
            fn from($value: $ty) -> Self {
                ScalarValue::$variant($convert)
            }
        }

        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Self::try_new(
                    DType::$variant(Nullability::NonNullable),
                    Some(ScalarValue::from(value)),
                )
                .vortex_expect("unable to construct a `Scalar`")
            }
        }

        impl From<Option<$ty>> for Scalar {
            fn from(value: Option<$ty>) -> Self {
                Self::try_new(
                    DType::$variant(Nullability::Nullable),
                    value.map(ScalarValue::from),
                )
                .vortex_expect("unable to construct a `Scalar`")
            }
        }
    };

    // Identity conversion (no transformation needed).
    ($ty:ty, $variant:ident) => {
        impl_into_scalar!($ty, $variant, |value| value);
    };
}

// Boolean scalar values can only be made from `bool`.
impl_into_scalar!(bool, Bool);

// Binary scalar values can be made from both `&[u8]` and `ByteBuffer`.
impl_into_scalar!(&[u8], Binary, |value| ByteBuffer::from(value.to_vec()));
impl_into_scalar!(ByteBuffer, Binary);

// UTF-8 scalar values can be made from `&str`, `String`, and `BufferString`.
impl_into_scalar!(&str, Utf8, |value| value.to_string().into());
impl_into_scalar!(String, Utf8, |value| value.into());
impl_into_scalar!(BufferString, Utf8);

////////////////////////////////////////////////////////////////////////////////////////////////////
// List (`Vec`) conversion into `Scalar`.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl<T> From<Vec<T>> for ScalarValue
where
    T: NativeDType,
    Scalar: From<T>,
{
    fn from(vec: Vec<T>) -> Self {
        ScalarValue::Tuple(
            vec.into_iter()
                .map(|elem| Scalar::from(elem).into_value())
                .collect(),
        )
    }
}

impl<T> From<Vec<T>> for Scalar
where
    T: NativeDType,
    Scalar: From<T>,
{
    fn from(vec: Vec<T>) -> Self {
        Scalar::try_new(
            DType::List(Arc::from(T::dtype()), Nullability::NonNullable),
            Some(ScalarValue::from(vec)),
        )
        .vortex_expect("unable to construct a list `Scalar` from `Vec<T>`")
    }
}

impl<T> From<Option<Vec<T>>> for Scalar
where
    T: NativeDType,
    Scalar: From<T>,
{
    fn from(vec: Option<Vec<T>>) -> Self {
        Scalar::try_new(
            DType::List(Arc::from(T::dtype()), Nullability::Nullable),
            vec.map(ScalarValue::from),
        )
        .vortex_expect("unable to construct a list `Scalar` from `Option<Vec<T>>`")
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Decimal conversion into `Scalar`.
////////////////////////////////////////////////////////////////////////////////////////////////////

impl From<DecimalValue> for ScalarValue {
    fn from(value: DecimalValue) -> Self {
        Self::Decimal(value)
    }
}

impl From<DecimalValue> for Scalar {
    fn from(value: DecimalValue) -> Self {
        let dtype = value.decimal_dtype();
        Self::try_new(
            DType::Decimal(dtype, Nullability::NonNullable),
            Some(ScalarValue::Decimal(value)),
        )
        .vortex_expect("unable to construct a decimal `Scalar` from `DecimalValue`")
    }
}

impl From<Option<DecimalValue>> for Scalar {
    fn from(value: Option<DecimalValue>) -> Self {
        let Some(value) = value else {
            // TODO(connor): This is definitely a footgun!
            // We have no way of knowing what the decimal value precision is (since we have no
            // data), so just choose a small one.
            return Self::null(DType::Decimal(
                DecimalDType::new(3, 0),
                Nullability::Nullable,
            ));
        };

        let dtype = value.decimal_dtype();
        Self::try_new(
            DType::Decimal(dtype, Nullability::Nullable),
            Some(ScalarValue::Decimal(value)),
        )
        .vortex_expect("unable to construct a decimal `Scalar` from `Option<DecimalValue>`")
    }
}
