// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::arrow::nulls_to_mask;
use crate::decimal::DVector;
use crate::decimal::DecimalVector;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::PrimitiveArray;
use arrow_array::types::Decimal32Type;
use arrow_array::types::Decimal64Type;
use arrow_array::types::Decimal128Type;
use arrow_array::types::Decimal256Type;
use arrow_schema::DataType;
use vortex_buffer::Buffer;
use vortex_dtype::PrecisionScale;
use vortex_dtype::i256;
use vortex_error::VortexError;
use vortex_error::vortex_err;

impl TryFrom<DecimalVector> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: DecimalVector) -> Result<Self, Self::Error> {
        match value {
            DecimalVector::D8(v) => ArrayRef::try_from(v),
            DecimalVector::D16(v) => ArrayRef::try_from(v),
            DecimalVector::D32(v) => ArrayRef::try_from(v),
            DecimalVector::D64(v) => ArrayRef::try_from(v),
            DecimalVector::D128(v) => ArrayRef::try_from(v),
            DecimalVector::D256(v) => ArrayRef::try_from(v),
        }
    }
}

macro_rules! impl_decimal_upcast_i32 {
    ($T:ty) => {
        impl TryFrom<DVector<$T>> for ArrayRef {
            type Error = VortexError;

            fn try_from(value: DVector<$T>) -> Result<Self, Self::Error> {
                let (_, elements, validity) = value.into_parts();
                // Upcast the DVector to Arrow's smallest decimal type (Decimal32)
                let elements =
                    Buffer::<i32>::from_trusted_len_iter(elements.iter().map(|i| *i as i32));
                Ok(Arc::new(PrimitiveArray::<Decimal32Type>::new(
                    elements.into_arrow_scalar_buffer(),
                    validity.into(),
                )))
            }
        }
    };
}

impl_decimal_upcast_i32!(i8);
impl_decimal_upcast_i32!(i16);

/// Direct Arrow conversion for vectors that map directly to Arrow decimal types.
macro_rules! impl_decimal_to_arrow {
    ($T:ty, $A:ty) => {
        impl TryFrom<DVector<$T>> for ArrayRef {
            type Error = VortexError;

            fn try_from(value: DVector<$T>) -> Result<Self, Self::Error> {
                let (_, elements, validity) = value.into_parts();
                Ok(Arc::new(PrimitiveArray::<$A>::new(
                    elements.into_arrow_scalar_buffer(),
                    validity.into(),
                )))
            }
        }
    };
}

impl_decimal_to_arrow!(i32, Decimal32Type);
impl_decimal_to_arrow!(i64, Decimal64Type);
impl_decimal_to_arrow!(i128, Decimal128Type);

impl TryFrom<DVector<i256>> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: DVector<i256>) -> Result<Self, Self::Error> {
        let (_, elements, validity) = value.into_parts();

        // Transmute the elements from our i256 to Arrow's.
        // SAFETY: we use Arrow's type internally for our layout.
        let elements =
            unsafe { std::mem::transmute::<Buffer<i256>, Buffer<arrow_buffer::i256>>(elements) };

        Ok(Arc::new(PrimitiveArray::<Decimal256Type>::new(
            elements.into_arrow_scalar_buffer(),
            validity.into(),
        )))
    }
}

impl TryFrom<ArrayRef> for DecimalVector {
    type Error = VortexError;

    fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
        match value.data_type() {
            DataType::Decimal32(precision, scale) => {
                let array = value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Decimal32Type>>()
                    .ok_or_else(|| {
                        vortex_err!("expected Decimal32Array, got {}", value.data_type())
                    })?;

                let elements = Buffer::<i32>::from_arrow_scalar_buffer(array.values().clone());
                let validity = nulls_to_mask(array.nulls(), array.len());
                let ps = PrecisionScale::<i32>::new(*precision, *scale);

                Ok(DecimalVector::D32(DVector::new(ps, elements, validity)))
            }
            DataType::Decimal64(precision, scale) => {
                let array = value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Decimal64Type>>()
                    .ok_or_else(|| {
                        vortex_err!("expected Decimal64Array, got {}", value.data_type())
                    })?;

                let elements = Buffer::<i64>::from_arrow_scalar_buffer(array.values().clone());
                let validity = nulls_to_mask(array.nulls(), array.len());
                let ps = PrecisionScale::<i64>::new(*precision, *scale);

                Ok(DecimalVector::D64(DVector::new(ps, elements, validity)))
            }
            DataType::Decimal128(precision, scale) => {
                let array = value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Decimal128Type>>()
                    .ok_or_else(|| {
                        vortex_err!("expected Decimal128Array, got {}", value.data_type())
                    })?;

                let elements = Buffer::<i128>::from_arrow_scalar_buffer(array.values().clone());
                let validity = nulls_to_mask(array.nulls(), array.len());
                let ps = PrecisionScale::<i128>::new(*precision, *scale);

                Ok(DecimalVector::D128(DVector::new(ps, elements, validity)))
            }
            DataType::Decimal256(precision, scale) => {
                let array = value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Decimal256Type>>()
                    .ok_or_else(|| {
                        vortex_err!("expected Decimal256Array, got {}", value.data_type())
                    })?;

                let elements =
                    Buffer::<arrow_buffer::i256>::from_arrow_scalar_buffer(array.values().clone());
                // SAFETY: we use Arrow's type internally for our layout.
                let elements = unsafe {
                    std::mem::transmute::<Buffer<arrow_buffer::i256>, Buffer<i256>>(elements)
                };
                let validity = nulls_to_mask(array.nulls(), array.len());
                let ps = PrecisionScale::<i256>::new(*precision, *scale);

                Ok(DecimalVector::D256(DVector::new(ps, elements, validity)))
            }
            dt => Err(vortex_err!("expected decimal array, got {}", dt)),
        }
    }
}
