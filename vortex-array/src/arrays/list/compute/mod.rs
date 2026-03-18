// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod filter;
mod is_constant;
mod is_sorted;
mod kernels;
mod mask;
pub(crate) mod rules;
mod slice;
mod take;

pub(crate) use kernels::PARENT_KERNELS;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::consistency::test_array_consistency;
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

        test_mask_conformance(&array.into_array());
    }

    #[test]
    fn test_filter_list() {
        let elements = buffer![0..35].into_array();
        let offsets = buffer![0, 5, 11, 18, 26, 35].into_array();
        let validity = Validity::AllValid;
        let array =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        test_filter_conformance(&array.into_array());
    }

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
        test_array_consistency(&array.into_array());
    }
}
