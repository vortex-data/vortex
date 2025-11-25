// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::PrimitiveArray;
use arrow_array::types::Float16Type;
use arrow_array::types::Float32Type;
use arrow_array::types::Float64Type;
use arrow_array::types::Int8Type;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::UInt8Type;
use arrow_array::types::UInt16Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use vortex_dtype::half::f16;
use vortex_error::VortexResult;
use vortex_vector::match_each_pvector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for PrimitiveVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        match_each_pvector!(self, |v| { v.into_arrow() })
    }
}

macro_rules! impl_primitive {
    ($T:ty, $A:ty) => {
        impl IntoArrow<ArrayRef> for PVector<$T> {
            fn into_arrow(self) -> VortexResult<ArrayRef> {
                let (elements, validity) = self.into_parts();
                Ok(Arc::new(PrimitiveArray::<$A>::new(
                    elements.into_arrow_scalar_buffer(),
                    validity.into_arrow()?,
                )))
            }
        }
    };
}

impl_primitive!(u8, UInt8Type);
impl_primitive!(u16, UInt16Type);
impl_primitive!(u32, UInt32Type);
impl_primitive!(u64, UInt64Type);
impl_primitive!(i8, Int8Type);
impl_primitive!(i16, Int16Type);
impl_primitive!(i32, Int32Type);
impl_primitive!(i64, Int64Type);
impl_primitive!(f16, Float16Type);
impl_primitive!(f32, Float32Type);
impl_primitive!(f64, Float64Type);
