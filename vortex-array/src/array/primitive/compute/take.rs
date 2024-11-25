use num_traits::AsPrimitive;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::{TakeFn, TakeOptions};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

impl TakeFn<PrimitiveArray> for PrimitiveEncoding {
    #[allow(clippy::cognitive_complexity)]
    fn take(
        &self,
        array: &PrimitiveArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        let indices = indices.clone().into_primitive()?;
        let validity = array.validity().take(indices.as_ref(), options)?;

        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = if options.skip_bounds_check {
                    take_primitive_unchecked(array.maybe_null_slice::<$T>(), indices.into_maybe_null_slice::<$I>())
                } else {
                    take_primitive(array.maybe_null_slice::<$T>(), indices.into_maybe_null_slice::<$I>())
                };
                Ok(PrimitiveArray::from_vec(values,validity).into_array())
            })
        })
    }
}

// We pass a Vec<I> in case we're T == u64.
// In which case, Rust should reuse the same Vec<u64> the result.
fn take_primitive<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: Vec<I>,
) -> Vec<T> {
    indices.into_iter().map(|idx| array[idx.as_()]).collect()
}

// We pass a Vec<I> in case we're T == u64.
// In which case, Rust should reuse the same Vec<u64> the result.
fn take_primitive_unchecked<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: Vec<I>,
) -> Vec<T> {
    indices
        .into_iter()
        .map(|idx| unsafe { *array.get_unchecked(idx.as_()) })
        .collect()
}

#[cfg(test)]
mod test {
    use crate::array::primitive::compute::take::take_primitive;

    #[test]
    fn test_take() {
        let a = vec![1i32, 2, 3, 4, 5];
        let result = take_primitive(&a, vec![0, 0, 4, 2]);
        assert_eq!(result, vec![1i32, 1, 5, 3]);
    }
}
