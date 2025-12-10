// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::types::Float16Type;
use arrow_array::types::Float32Type;
use arrow_array::types::Float64Type;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::Int8Type;
use arrow_array::types::UInt16Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use arrow_array::types::UInt8Type;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_dtype::half::f16;
use vortex_error::VortexResult;
use vortex_vector::match_each_pvector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::arrow::nulls_to_mask;
use crate::arrow::IntoArrow;
use crate::arrow::IntoVector;

impl IntoArrow for PrimitiveVector {
    type Output = ArrayRef;

    fn into_arrow(self) -> VortexResult<Self::Output> {
        match_each_pvector!(self, |v| { Ok(Arc::new(v.into_arrow()?)) })
    }
}

macro_rules! impl_primitive_to_arrow {
    ($T:ty, $A:ty) => {
        impl IntoArrow for PVector<$T> {
            type Output = PrimitiveArray<$A>;

            fn into_arrow(self) -> VortexResult<Self::Output> {
                let (elements, validity) = self.into_parts();
                Ok(PrimitiveArray::<$A>::new(
                    elements.into_arrow_scalar_buffer(),
                    validity.into(),
                ))
            }
        }
    };
}

impl_primitive_to_arrow!(u8, UInt8Type);
impl_primitive_to_arrow!(u16, UInt16Type);
impl_primitive_to_arrow!(u32, UInt32Type);
impl_primitive_to_arrow!(u64, UInt64Type);
impl_primitive_to_arrow!(i8, Int8Type);
impl_primitive_to_arrow!(i16, Int16Type);
impl_primitive_to_arrow!(i32, Int32Type);
impl_primitive_to_arrow!(i64, Int64Type);
impl_primitive_to_arrow!(f16, Float16Type);
impl_primitive_to_arrow!(f32, Float32Type);
impl_primitive_to_arrow!(f64, Float64Type);

macro_rules! impl_primitive_from_arrow {
    ($T:ty, $A:ty) => {
        impl IntoVector for &PrimitiveArray<$A> {
            type Output = PVector<$T>;

            fn into_vector(self) -> VortexResult<Self::Output> {
                let elements = Buffer::<$T>::from_arrow_scalar_buffer(self.values().clone());
                let validity = nulls_to_mask(self.nulls(), self.len());
                Ok(PVector::new(elements, validity))
            }
        }
    };
}

impl_primitive_from_arrow!(u8, UInt8Type);
impl_primitive_from_arrow!(u16, UInt16Type);
impl_primitive_from_arrow!(u32, UInt32Type);
impl_primitive_from_arrow!(u64, UInt64Type);
impl_primitive_from_arrow!(i8, Int8Type);
impl_primitive_from_arrow!(i16, Int16Type);
impl_primitive_from_arrow!(i32, Int32Type);
impl_primitive_from_arrow!(i64, Int64Type);
impl_primitive_from_arrow!(f16, Float16Type);
impl_primitive_from_arrow!(f32, Float32Type);
impl_primitive_from_arrow!(f64, Float64Type);
