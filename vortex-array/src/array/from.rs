use vortex_buffer::{Buffer, BufferMut, BufferString, ByteBuffer};
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
        impl From<Buffer<$P>> for ArrayData {
            fn from(value: Buffer<$P>) -> Self {
                PrimitiveArray::new(value, Validity::NonNullable).into_array()
            }
        }

        impl From<BufferMut<$P>> for ArrayData {
            fn from(value: BufferMut<$P>) -> Self {
                PrimitiveArray::new(value.freeze(), Validity::NonNullable).into_array()
            }
        }

        impl FromIterator<Option<$P>> for ArrayData {
            fn from_iter<T: IntoIterator<Item = Option<$P>>>(iter: T) -> Self {
                PrimitiveArray::from_option_iter(iter).into_array()
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

impl FromIterator<Option<ByteBuffer>> for ArrayData {
    fn from_iter<T: IntoIterator<Item = Option<ByteBuffer>>>(iter: T) -> Self {
        VarBinViewArray::from_iter(iter, DType::Binary(Nullability::Nullable)).into_array()
    }
}
