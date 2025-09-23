// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod filter;
mod is_constant;
mod mask;
mod take;

// Note that there is also a `list_contains` kernel located in the file
// `vortex-array/src/compute/list_contains.rs` (it is there because other non-canonical encodings
// can implement this kernel).

use vortex_error::VortexResult;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{
    IsSortedKernel, IsSortedKernelAdapter, MinMaxKernel, MinMaxKernelAdapter, MinMaxResult,
};
use crate::register_kernel;

impl MinMaxKernel for ListVTable {
    fn min_max(&self, _array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement list min max
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(ListVTable).lift());

// TODO(ngates): Implement is sorted
impl IsSortedKernel for ListVTable {
    fn is_sorted(&self, _array: &ListArray) -> VortexResult<Option<bool>> {
        Ok(None)
    }

    fn is_strict_sorted(&self, _array: &ListArray) -> VortexResult<Option<bool>> {
        Ok(None)
    }
}

register_kernel!(IsSortedKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::ListArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::validity::Validity;

    #[test]
    fn test_mask_list() {
        let elements = buffer![0..35].into_array();
        let offsets = buffer![0, 5, 11, 18, 26, 35].into_array();
        let validity = Validity::AllValid;
        let array =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        test_mask_conformance(array.as_ref());
    }

    #[test]
    fn test_filter_list() {
        let elements = buffer![0..35].into_array();
        let offsets = buffer![0, 5, 11, 18, 26, 35].into_array();
        let validity = Validity::AllValid;
        let array =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        test_filter_conformance(array.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::{BoolArray, ListArray, PrimitiveArray};
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::validity::Validity;

    #[rstest]
    // From test_all_consistency
    #[case::list_simple(ListArray::try_new(
        buffer![0i32, 1, 2, 3, 4, 5].into_array(),
        buffer![0, 2, 3, 5, 5, 6].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    #[case::list_nullable(ListArray::try_new(
        buffer![10i32, 20, 30, 40, 50].into_array(),
        buffer![0, 2, 3, 4, 5].into_array(),
        Validity::Array(BoolArray::from_iter(vec![true, false, true, true]).into_array()),
    ).unwrap())]
    // Additional test cases
    #[case::list_empty_lists(ListArray::try_new(
        buffer![100i32, 200, 300].into_array(),
        buffer![0, 0, 2, 2, 3, 3].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    #[case::list_single_element(ListArray::try_new(
        buffer![42i32].into_array(),
        buffer![0, 1].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    #[case::list_large(ListArray::try_new(
        buffer![0..1000i32].into_array(),
        PrimitiveArray::from_iter((0..=100).map(|i| i * 10)).into_array(),
        Validity::NonNullable,
    ).unwrap())]
    fn test_list_consistency(#[case] array: ListArray) {
        test_array_consistency(array.as_ref());
    }
}
