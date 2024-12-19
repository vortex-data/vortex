use num_traits::AsPrimitive;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

impl TakeFn<PrimitiveArray> for PrimitiveEncoding {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &PrimitiveArray, indices: &ArrayData) -> VortexResult<ArrayData> {
        let indices = indices.clone().into_primitive()?;
        let validity = array.validity().take(indices.as_ref())?;

        // FIXME(DK): we could save an allocation and re-use memory if: we take the indices as
        // owned, there are no other references to the underlying indices buffer, and the indices
        // bit-width matches the array's bit-width.
        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = take_primitive(array.maybe_null_slice::<$T>(), indices.maybe_null_slice::<$I>());
                Ok(PrimitiveArray::from_vec(values, validity).into_array())
            })
        })
    }

    unsafe fn take_unchecked(
        &self,
        array: &PrimitiveArray,
        indices: &ArrayData,
    ) -> VortexResult<ArrayData> {
        let indices = indices.clone().into_primitive()?;
        let validity = unsafe { array.validity().take_unchecked(indices.as_ref())? };

        // FIXME(DK): we could save an allocation and re-use memory if: We take the indices as
        // owned, there are no other references to the underlying indices buffer, and the indices
        // bit-width matches the array's bit-width.
        match_each_native_ptype!(array.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                let values = take_primitive_unchecked(array.maybe_null_slice::<$T>(), indices.maybe_null_slice::<$I>());
                Ok(PrimitiveArray::from_vec(values, validity).into_array())
            })
        })
    }
}

fn take_primitive<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: &[I],
) -> Vec<T> {
    indices.iter().map(|idx| array[idx.as_()]).collect()
}

unsafe fn take_primitive_unchecked<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: &[I],
) -> Vec<T> {
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
        assert_eq!(result, vec![1i32, 1, 5, 3]);
    }
}
