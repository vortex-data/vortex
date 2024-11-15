use num_traits::PrimInt;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType};
use vortex_error::{vortex_panic, VortexResult};

use crate::array::primitive::PrimitiveArray;
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

impl TakeFn for PrimitiveArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        let validity = self.validity();
        let indices = indices.clone().into_primitive()?;
        match_each_native_ptype!(self.ptype(), |$T| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                Ok(PrimitiveArray::from_vec(
                    take_primitive(self.maybe_null_slice::<$T>(), indices.maybe_null_slice::<$I>()),
                    validity.take(indices.as_ref())?,
                ).into_array())
            })
        })
    }
}

fn take_primitive<T: NativePType, I: NativePType + PrimInt>(array: &[T], indices: &[I]) -> Vec<T> {
    indices
        .iter()
        .map(|&idx| {
            array[idx.to_usize().unwrap_or_else(|| {
                vortex_panic!("Failed to convert index to usize: {}", idx);
            })]
        })
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
