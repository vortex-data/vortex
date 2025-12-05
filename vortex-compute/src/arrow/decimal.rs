// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::PrimitiveArray;
use arrow_array::types::Decimal32Type;
use arrow_array::types::Decimal64Type;
use arrow_array::types::Decimal128Type;
use arrow_array::types::Decimal256Type;
use vortex_buffer::Buffer;
use vortex_dtype::PrecisionScale;
use vortex_dtype::i256;
use vortex_error::VortexResult;
use vortex_vector::decimal::DVector;
use vortex_vector::decimal::DecimalVector;

use crate::arrow::IntoArrow;
use crate::arrow::IntoVector;
use crate::arrow::nulls_to_mask;

impl IntoArrow for DecimalVector {
    type Output = ArrayRef;

    fn into_arrow(self) -> VortexResult<Self::Output> {
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
        impl IntoArrow for DVector<$T> {
            type Output = ArrayRef;

            fn into_arrow(self) -> VortexResult<Self::Output> {
                let (_, elements, validity) = self.into_parts();
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
        impl IntoArrow for DVector<$T> {
            type Output = ArrayRef;

            fn into_arrow(self) -> VortexResult<Self::Output> {
                let (_, elements, validity) = self.into_parts();
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

impl IntoArrow for DVector<i256> {
    type Output = ArrayRef;

    fn into_arrow(self) -> VortexResult<Self::Output> {
        let (_, elements, validity) = self.into_parts();

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

/// Convert a Decimal32 Arrow array to a DecimalVector.
impl IntoVector for &PrimitiveArray<Decimal32Type> {
    type Output = DecimalVector;

    fn into_vector(self) -> VortexResult<Self::Output> {
        let (precision, scale) = match self.data_type() {
            arrow_schema::DataType::Decimal32(p, s) => (*p, *s),
            _ => unreachable!("PrimitiveArray<Decimal32Type> must have Decimal32 data type"),
        };

        let elements = Buffer::<i32>::from_arrow_scalar_buffer(self.values().clone());
        let validity = nulls_to_mask(self.nulls(), self.len());
        let ps = PrecisionScale::<i32>::new(precision, scale);

        Ok(DecimalVector::D32(DVector::new(ps, elements, validity)))
    }
}

/// Convert a Decimal64 Arrow array to a DecimalVector.
impl IntoVector for &PrimitiveArray<Decimal64Type> {
    type Output = DecimalVector;

    fn into_vector(self) -> VortexResult<Self::Output> {
        let (precision, scale) = match self.data_type() {
            arrow_schema::DataType::Decimal64(p, s) => (*p, *s),
            _ => unreachable!("PrimitiveArray<Decimal64Type> must have Decimal64 data type"),
        };

        let elements = Buffer::<i64>::from_arrow_scalar_buffer(self.values().clone());
        let validity = nulls_to_mask(self.nulls(), self.len());
        let ps = PrecisionScale::<i64>::new(precision, scale);

        Ok(DecimalVector::D64(DVector::new(ps, elements, validity)))
    }
}

/// Convert a Decimal128 Arrow array to a DecimalVector.
impl IntoVector for &PrimitiveArray<Decimal128Type> {
    type Output = DecimalVector;

    fn into_vector(self) -> VortexResult<Self::Output> {
        let (precision, scale) = match self.data_type() {
            arrow_schema::DataType::Decimal128(p, s) => (*p, *s),
            _ => unreachable!("PrimitiveArray<Decimal128Type> must have Decimal128 data type"),
        };

        let elements = Buffer::<i128>::from_arrow_scalar_buffer(self.values().clone());
        let validity = nulls_to_mask(self.nulls(), self.len());
        let ps = PrecisionScale::<i128>::new(precision, scale);

        Ok(DecimalVector::D128(DVector::new(ps, elements, validity)))
    }
}

/// Convert a Decimal256 Arrow array to a DecimalVector.
impl IntoVector for &PrimitiveArray<Decimal256Type> {
    type Output = DecimalVector;

    fn into_vector(self) -> VortexResult<Self::Output> {
        let (precision, scale) = match self.data_type() {
            arrow_schema::DataType::Decimal256(p, s) => (*p, *s),
            _ => unreachable!("PrimitiveArray<Decimal256Type> must have Decimal256 data type"),
        };

        let elements =
            Buffer::<arrow_buffer::i256>::from_arrow_scalar_buffer(self.values().clone());
        // SAFETY: we use Arrow's type internally for our layout.
        let elements =
            unsafe { std::mem::transmute::<Buffer<arrow_buffer::i256>, Buffer<i256>>(elements) };
        let validity = nulls_to_mask(self.nulls(), self.len());
        let ps = PrecisionScale::<i256>::new(precision, scale);

        Ok(DecimalVector::D256(DVector::new(ps, elements, validity)))
    }
}
