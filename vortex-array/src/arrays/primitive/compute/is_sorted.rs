use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::{IsSortedFn, IteratorExt};
use crate::variants::PrimitiveArrayTrait;

impl IsSortedFn<&PrimitiveArray> for PrimitiveEncoding {
    fn is_sorted(&self, array: &PrimitiveArray, strict: bool) -> VortexResult<bool> {
        match_each_native_ptype!(array.ptype(), |$P| {
            compute_is_sorted::<$P>(array, strict)
        })
    }
}

fn compute_is_sorted<T: NativePType + PartialOrd>(
    array: &PrimitiveArray,
    strict: bool,
) -> VortexResult<bool> {
    match array.validity_mask()? {
        Mask::AllFalse(_) => Ok(!strict),
        Mask::AllTrue(_) => {
            let slice = array.as_slice::<T>();
            Ok(slice.into_iter().is_sorted_with_strictness(strict))
        }
        Mask::Values(mask_values) => {
            let slice = array.as_slice::<T>();
            let set_indices = mask_values.boolean_buffer().set_indices();

            Ok(set_indices
                .map(|idx| slice[idx])
                .is_sorted_with_strictness(strict))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexUnwrap;

    use super::*;
    use crate::compute::is_sorted_opts;

    #[rstest]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), false, true)]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), true, true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), false, true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), true, false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2), None]), false, true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2), None]), true, false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1), None]), false, true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1), None]), true, false)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), true, false)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), false, true)]
    fn test_primitive_is_sorted(
        #[case] array: PrimitiveArray,
        #[case] strict: bool,
        #[case] expected: bool,
    ) {
        assert_eq!(is_sorted_opts(&array, strict).vortex_unwrap(), expected);
    }
}
