// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::types::{Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type};
use arrow_array::{ArrayRef, PrimitiveArray};
use vortex_buffer::Buffer;
use vortex_dtype::i256;
use vortex_error::VortexResult;
use vortex_vector::decimal::{DVector, DecimalVector};

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for DecimalVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        match self {
            DecimalVector::D8(v) => v.into_arrow(),
            DecimalVector::D16(v) => v.into_arrow(),
            DecimalVector::D32(v) => v.into_arrow(),
            DecimalVector::D64(v) => v.into_arrow(),
            DecimalVector::D128(v) => v.into_arrow(),
            DecimalVector::D256(v) => v.into_arrow(),
        }
    }
}

macro_rules! impl_decimal_upcast_i32 {
    ($T:ty) => {
        impl IntoArrow<ArrayRef> for DVector<$T> {
            fn into_arrow(self) -> VortexResult<ArrayRef> {
                let (_, elements, validity) = self.into_parts();
                // Upcast the DVector to Arrow's smallest decimal type (Decimal32)
                let elements =
                    Buffer::<i32>::from_trusted_len_iter(elements.iter().map(|i| *i as i32));
                Ok(Arc::new(PrimitiveArray::<Decimal32Type>::new(
                    elements.into_arrow_scalar_buffer(),
                    validity.into_arrow()?,
                )))
            }
        }
    };
}

impl_decimal_upcast_i32!(i8);
impl_decimal_upcast_i32!(i16);

/// Direct Arrow conversion for vectors that map directly to Arrow decimal types.
macro_rules! impl_decimal {
    ($T:ty, $A:ty) => {
        impl IntoArrow<ArrayRef> for DVector<$T> {
            fn into_arrow(self) -> VortexResult<ArrayRef> {
                let (_, elements, validity) = self.into_parts();
                Ok(Arc::new(PrimitiveArray::<$A>::new(
                    elements.into_arrow_scalar_buffer(),
                    validity.into_arrow()?,
                )))
            }
        }
    };
}

impl_decimal!(i32, Decimal32Type);
impl_decimal!(i64, Decimal64Type);
impl_decimal!(i128, Decimal128Type);

impl IntoArrow<ArrayRef> for DVector<i256> {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        let (_, elements, validity) = self.into_parts();

        // Transmute the elements from our i256 to Arrow's.
        // SAFETY: we use Arrow's type internally for our layout.
        let elements =
            unsafe { std::mem::transmute::<Buffer<i256>, Buffer<arrow_buffer::i256>>(elements) };

        Ok(Arc::new(PrimitiveArray::<Decimal256Type>::new(
            elements.into_arrow_scalar_buffer(),
            validity.into_arrow()?,
        )))
    }
}
