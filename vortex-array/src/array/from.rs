use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, Nullability};

use super::{BoolArray, PrimitiveArray, VarBinViewArray};
use crate::validity::Validity;
use crate::{ArrayData, IntoArrayData as _};

impl FromIterator<Option<bool>> for ArrayData {
    fn from_iter<T: IntoIterator<Item = Option<bool>>>(iter: T) -> Self {
        BoolArray::from_iter(iter).into_array()
    }
}

macro_rules! impl_from_primitive_for_array {
    ($P:ty) => {
        // For primitives, it's more efficient to use from_vec, instead of from_iter since
        // the values are already correctly laid out in-memory.
        impl From<Vec<$P>> for ArrayData {
            fn from(value: Vec<$P>) -> Self {
                PrimitiveArray::from_vec(value, Validity::NonNullable).into_array()
            }
        }

        impl FromIterator<Option<$P>> for ArrayData {
            fn from_iter<T: IntoIterator<Item = Option<$P>>>(iter: T) -> Self {
                PrimitiveArray::from_nullable_vec(iter.into_iter().collect()).into_array()
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

impl FromIterator<Option<String>> for ArrayData {
    fn from_iter<T: IntoIterator<Item = Option<String>>>(iter: T) -> Self {
        VarBinViewArray::from_iter(iter, DType::Utf8(Nullability::Nullable)).into_array()
    }
}

impl FromIterator<Option<BufferString>> for ArrayData {
    fn from_iter<T: IntoIterator<Item = Option<BufferString>>>(iter: T) -> Self {
        VarBinViewArray::from_iter(iter, DType::Utf8(Nullability::Nullable)).into_array()
    }
}

impl FromIterator<Option<Buffer>> for ArrayData {
    fn from_iter<T: IntoIterator<Item = Option<Buffer>>>(iter: T) -> Self {
        VarBinViewArray::from_iter(iter, DType::Binary(Nullability::Nullable)).into_array()
    }
}
