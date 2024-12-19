use arrow_buffer::{ArrowNativeType, Buffer, OffsetBuffer};
use vortex_dtype::NativePType;

use crate::array::PrimitiveArray;

pub fn as_scalar_buffer<T: NativePType + ArrowNativeType>(array: PrimitiveArray) -> Buffer<T> {
    array.maybe_null_scalar_buffer::<T>().into_arrow()
}

pub fn as_offset_buffer<T: NativePType + ArrowNativeType>(
    array: PrimitiveArray,
) -> OffsetBuffer<T> {
    unsafe { OffsetBuffer::new_unchecked(as_scalar_buffer(array)) }
}
