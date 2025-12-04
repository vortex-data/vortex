// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::arrow::nulls_to_mask;
use crate::match_each_pvector;
use crate::primitive::PVector;
use crate::primitive::PrimitiveVector;
use arrow_array::Array;
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
use arrow_schema::DataType;
use vortex_buffer::Buffer;
use vortex_dtype::half::f16;
use vortex_error::VortexError;
use vortex_error::vortex_err;

impl TryFrom<PrimitiveVector> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: PrimitiveVector) -> Result<Self, Self::Error> {
        match_each_pvector!(value, |v| { ArrayRef::try_from(v) })
    }
}

macro_rules! impl_primitive_to_arrow {
    ($T:ty, $A:ty) => {
        impl TryFrom<PVector<$T>> for ArrayRef {
            type Error = VortexError;

            fn try_from(value: PVector<$T>) -> Result<Self, Self::Error> {
                let (elements, validity) = value.into_parts();
                Ok(Arc::new(PrimitiveArray::<$A>::new(
                    elements.into_arrow_scalar_buffer(),
                    validity.into(),
                )))
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

impl TryFrom<&dyn Array> for PrimitiveVector {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> Result<Self, Self::Error> {
        match value.data_type() {
            DataType::UInt8 => PVector::<u8>::try_from(value).map(PrimitiveVector::from),
            DataType::UInt16 => PVector::<u16>::try_from(value).map(PrimitiveVector::from),
            DataType::UInt32 => PVector::<u32>::try_from(value).map(PrimitiveVector::from),
            DataType::UInt64 => PVector::<u64>::try_from(value).map(PrimitiveVector::from),
            DataType::Int8 => PVector::<i8>::try_from(value).map(PrimitiveVector::from),
            DataType::Int16 => PVector::<i16>::try_from(value).map(PrimitiveVector::from),
            DataType::Int32 => PVector::<i32>::try_from(value).map(PrimitiveVector::from),
            DataType::Int64 => PVector::<i64>::try_from(value).map(PrimitiveVector::from),
            DataType::Float16 => PVector::<f16>::try_from(value).map(PrimitiveVector::from),
            DataType::Float32 => PVector::<f32>::try_from(value).map(PrimitiveVector::from),
            DataType::Float64 => PVector::<f64>::try_from(value).map(PrimitiveVector::from),
            dt => Err(vortex_err!("expected primitive array, got {}", dt)),
        }
    }
}

macro_rules! impl_primitive_from_arrow {
    ($T:ty, $A:ty) => {
        impl From<&PrimitiveArray<$A>> for PVector<$T> {
            fn from(array: &PrimitiveArray<$A>) -> Self {
                let elements = Buffer::<$T>::from_arrow_scalar_buffer(array.values().clone());
                let validity = nulls_to_mask(array.nulls(), array.len());
                PVector::new(elements, validity)
            }
        }

        impl TryFrom<&dyn Array> for PVector<$T> {
            type Error = VortexError;

            fn try_from(value: &dyn Array) -> Result<Self, Self::Error> {
                let array = value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$A>>()
                    .ok_or_else(|| {
                        vortex_err!(
                            "expected PrimitiveArray<{}>, got {}",
                            stringify!($A),
                            value.data_type()
                        )
                    })?;
                Ok(PVector::from(array))
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
