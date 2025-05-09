use itertools::Itertools;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::{Array, register_kernel};

impl IsSortedKernel for PrimitiveEncoding {
    fn is_sorted(&self, array: &PrimitiveArray) -> VortexResult<bool> {
        match_each_native_ptype!(array.ptype(), |$P| {
            compute_is_sorted::<$P>(array, false)
        })
    }

    fn is_strict_sorted(&self, array: &PrimitiveArray) -> VortexResult<bool> {
        match_each_native_ptype!(array.ptype(), |$P| {
            compute_is_sorted::<$P>(array, true)
        })
    }
}

register_kernel!(IsSortedKernelAdapter(PrimitiveEncoding).lift());

#[derive(Copy, Clone)]
struct ComparablePrimitive<T: NativePType>(T);

impl<T> From<&T> for ComparablePrimitive<T>
where
    T: NativePType,
{
    fn from(value: &T) -> Self {
        Self(*value)
    }
}

impl<T> PartialOrd for ComparablePrimitive<T>
where
    T: NativePType,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.total_compare(other.0))
    }
}

impl<T> PartialEq for ComparablePrimitive<T>
where
    T: NativePType,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.is_eq(other.0)
    }
}

fn compute_is_sorted<T: NativePType>(array: &PrimitiveArray, strict: bool) -> VortexResult<bool> {
    match array.validity_mask()? {
        Mask::AllFalse(_) => Ok(!strict),
        Mask::AllTrue(_) => {
            let slice = array.as_slice::<T>();
            let iter = slice.iter().map(ComparablePrimitive::from);

            Ok(if strict {
                iter.is_strict_sorted()
            } else {
                iter.is_sorted()
            })
        }
        Mask::Values(mask_values) => {
            let iter = mask_values
                .boolean_buffer()
                .iter()
                .zip_eq(array.as_slice::<T>())
                .map(|(is_valid, value)| is_valid.then_some(ComparablePrimitive::from(value)));

            Ok(if strict {
                iter.is_strict_sorted()
            } else {
                iter.is_sorted()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexUnwrap;

    use super::*;
    use crate::compute::{is_sorted, is_strict_sorted};

    #[rstest]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2)]), true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1)]), true)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), false)]
    fn test_primitive_is_sorted(#[case] array: PrimitiveArray, #[case] expected: bool) {
        assert_eq!(is_sorted(&array).vortex_unwrap(), expected);
    }

    #[rstest]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2), None]), false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1), None]), false)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), false)]
    fn test_primitive_is_strict_sorted(#[case] array: PrimitiveArray, #[case] expected: bool) {
        assert_eq!(is_strict_sorted(&array).vortex_unwrap(), expected);
    }
}
