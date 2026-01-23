// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::DictArray;
use super::DictVTable;
use crate::Array as _;
use crate::compute::MinMaxKernel;
use crate::compute::MinMaxKernelAdapter;
use crate::compute::MinMaxResult;
use crate::compute::mask;
use crate::compute::min_max;
use crate::register_kernel;

impl MinMaxKernel for DictVTable {
    fn min_max(&self, array: &DictArray) -> VortexResult<Option<MinMaxResult>> {
        let codes_validity = array.codes().validity_mask()?;
        if codes_validity.all_false() {
            return Ok(None);
        }

        // Fast path: if all values are referenced, directly compute min/max on values
        if array.has_all_values_referenced() {
            return min_max(array.values());
        }

        // Slow path: compute which values are unreferenced and mask them out
        let unreferenced_mask = Mask::from_buffer(array.compute_referenced_values_mask(false)?);
        min_max(&mask(array.values(), &unreferenced_mask)?)
    }
}

register_kernel!(MinMaxKernelAdapter(DictVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::DictArray;
    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::builders::dict::dict_encode;
    use crate::compute::min_max;

    fn assert_min_max(array: &dyn Array, expected: Option<(i32, i32)>) {
        match (min_max(array).unwrap(), expected) {
            (Some(result), Some((expected_min, expected_max))) => {
                assert_eq!(i32::try_from(result.min).unwrap(), expected_min);
                assert_eq!(i32::try_from(result.max).unwrap(), expected_max);
            }
            (None, None) => {}
            (got, expected) => panic!(
                "min_max mismatch: expected {:?}, got {:?}",
                expected,
                got.as_ref().map(|r| (
                    i32::try_from(r.min.clone()).ok(),
                    i32::try_from(r.max.clone()).ok()
                ))
            ),
        }
    }

    #[rstest]
    #[case::covering(
        DictArray::try_new(
            buffer![0u32, 1, 2, 3, 0, 1].into_array(),
            buffer![10i32, 20, 30, 40].into_array(),
        ).unwrap(),
        (10, 40)
    )]
    #[case::non_covering_duplicates(
        DictArray::try_new(
            buffer![1u32, 1, 1, 3, 3].into_array(),
            buffer![1i32, 2, 3, 4, 5].into_array(),
        ).unwrap(),
        (2, 4)
    )]
    // Non-covering: codes with gaps
    #[case::non_covering_gaps(
        DictArray::try_new(
            buffer![0u32, 2, 4].into_array(),
            buffer![1i32, 2, 3, 4, 5].into_array(),
        ).unwrap(),
        (1, 5)
    )]
    #[case::single(dict_encode(&buffer![42i32].into_array()).unwrap(), (42, 42))]
    #[case::nullable_codes(
        DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u32), None, Some(1), Some(2)]).into_array(),
            buffer![10i32, 20, 30].into_array(),
        ).unwrap(),
        (10, 30)
    )]
    #[case::nullable_values(
        dict_encode(
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), Some(1), None]).as_ref()
        ).unwrap(),
        (1, 2)
    )]
    fn test_min_max(#[case] dict: DictArray, #[case] expected: (i32, i32)) {
        assert_min_max(dict.as_ref(), Some(expected));
    }

    #[test]
    fn test_sliced_dict() {
        let reference = PrimitiveArray::from_iter([1, 5, 10, 50, 100]);
        let dict = dict_encode(reference.as_ref()).unwrap();
        let sliced = dict.slice(1..3).unwrap();
        assert_min_max(sliced.as_ref(), Some((5, 10)));
    }

    #[rstest]
    #[case::empty(
        DictArray::try_new(
            PrimitiveArray::from_iter(Vec::<u32>::new()).into_array(),
            buffer![10i32, 20, 30].into_array(),
        ).unwrap()
    )]
    #[case::all_null_codes(
        DictArray::try_new(
            PrimitiveArray::from_option_iter([Option::<u32>::None, None, None]).into_array(),
            buffer![10i32, 20, 30].into_array(),
        ).unwrap()
    )]
    fn test_min_max_none(#[case] dict: DictArray) {
        assert_min_max(dict.as_ref(), None);
    }
}
