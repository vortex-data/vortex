use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, Nullability};

use super::{BoolArray, PrimitiveArray, VarBinViewArray};
use crate::validity::Validity;
use crate::{ArrayData, IntoArrayData as _};

// `From<Vec<Option<!>>> for Array` requries the experimental uninhabited type: !.

impl From<Vec<Option<bool>>> for ArrayData {
    fn from(value: Vec<Option<bool>>) -> Self {
        BoolArray::from_iter(value).into_array()
    }
}

macro_rules! impl_from_primitive_for_array {
    ($P:ty) => {
        impl From<Vec<$P>> for ArrayData {
            fn from(value: Vec<$P>) -> Self {
                PrimitiveArray::from_vec(value, Validity::NonNullable).into_array()
            }
        }

        impl From<Vec<Option<$P>>> for ArrayData {
            fn from(value: Vec<Option<$P>>) -> Self {
                PrimitiveArray::from_nullable_vec(value).into_array()
            }
        }
    };
}

impl_from_primitive_for_array!(u8);
impl_from_primitive_for_array!(u16);
impl_from_primitive_for_array!(u32);
impl_from_primitive_for_array!(u64);
impl_from_primitive_for_array!(i8);
impl_from_primitive_for_array!(i16);
impl_from_primitive_for_array!(i32);
impl_from_primitive_for_array!(i64);
impl_from_primitive_for_array!(f16);
impl_from_primitive_for_array!(f32);
impl_from_primitive_for_array!(f64);

impl From<Vec<Option<BufferString>>> for ArrayData {
    fn from(value: Vec<Option<BufferString>>) -> Self {
        VarBinViewArray::from_iter(value, DType::Utf8(Nullability::Nullable)).into_array()
    }
}

impl From<Vec<Option<Buffer>>> for ArrayData {
    fn from(value: Vec<Option<Buffer>>) -> Self {
        VarBinViewArray::from_iter(value, DType::Binary(Nullability::Nullable)).into_array()
    }
}
