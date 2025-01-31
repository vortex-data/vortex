use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray, IntoArrayVariant};

impl TakeFn<PrimitiveArray> for PrimitiveEncoding {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &PrimitiveArray, indices: &Array) -> VortexResult<Array> {
        let indices = indices.clone().into_primitive()?;
        let validity = array.validity().take(indices.as_ref())?;

        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = take_primitive(array.as_slice::<$T>(), indices.as_slice::<$I>());
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }

    unsafe fn take_unchecked(
        &self,
        array: &PrimitiveArray,
        indices: &Array,
    ) -> VortexResult<Array> {
        let indices = indices.clone().into_primitive()?;
        let validity = unsafe { array.validity().take_unchecked(indices.as_ref())? };

        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = take_primitive_unchecked(array.as_slice::<$T>(), indices.as_slice::<$I>());
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }
}

fn take_primitive<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: &[I],
) -> Buffer<T> {
    indices.iter().map(|idx| array[idx.as_()]).collect()
}

// We pass a Vec<I> in case we're T == u64.
// In which case, Rust should reuse the same Vec<u64> the result.
unsafe fn take_primitive_unchecked<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: &[I],
) -> Buffer<T> {
    indices
        .iter()
        .map(|idx| unsafe { *array.get_unchecked(idx.as_()) })
        .collect()
}

#[cfg(test)]
mod test {
    use crate::array::primitive::compute::take::take_primitive;

    #[test]
    fn test_take() {
        let a = vec![1i32, 2, 3, 4, 5];
        let result = take_primitive(&a, &[0, 0, 4, 2]);
        assert_eq!(result.as_slice(), &[1i32, 1, 5, 3]);
    }
}
