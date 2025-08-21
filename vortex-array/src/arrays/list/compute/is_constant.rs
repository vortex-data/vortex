// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::NumericOperator;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, numeric};
use crate::register_kernel;

const SMALL_ARRAY_THRESHOLD: usize = 64;

impl IsConstantKernel for ListVTable {
    fn is_constant(&self, array: &ListArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        // At this point, we're guaranteed:
        // - Array has at least 2 elements
        // - All elements are valid (no nulls)

        let manual_check_until = std::cmp::min(SMALL_ARRAY_THRESHOLD, array.len());

        let first_list_len = array.offset_at(1) - array.offset_at(0);
        for i in 1..manual_check_until {
            let current_list_len = array.offset_at(i + 1) - array.offset_at(i);
            if current_list_len != first_list_len {
                return Ok(Some(false));
            }
        }

        if opts.is_negligible_cost() {
            return Ok(None);
        }

        if array.len() > SMALL_ARRAY_THRESHOLD {
            // check the rest of the element lengths
            let start_offsets = array.offsets.slice(SMALL_ARRAY_THRESHOLD, array.len());
            let end_offsets = array
                .offsets
                .slice(SMALL_ARRAY_THRESHOLD + 1, array.len() + 1);
            let list_lengths = numeric(&end_offsets, &start_offsets, NumericOperator::Sub)?;

            if !list_lengths.is_constant() {
                return Ok(Some(false));
            }
        }

        // If all lists have the same length, compare the actual list contents
        let first_scalar = array.scalar_at(0);
        for i in 1..array.len() {
            let current_scalar = array.scalar_at(i);
            if current_scalar != first_scalar {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}

register_kernel!(IsConstantKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod tests {

    use rstest::rstest;
    use vortex_dtype::FieldNames;

    use crate::IntoArray;
    use crate::arrays::{ListArray, PrimitiveArray, StructArray};
    use crate::compute::is_constant;
    use crate::validity::Validity;

    #[test]
    fn test_is_constant_nested_list() {
        let xs = ListArray::try_new(
            PrimitiveArray::from_iter([0i32, 1, 0, 1]).into_array(),
            PrimitiveArray::from_iter([0u32, 2, 4]).into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let struct_of_lists = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            2,
            Validity::NonNullable,
        )
        .unwrap();
        assert!(
            is_constant(&struct_of_lists.clone().into_array())
                .unwrap()
                .unwrap()
        );
        assert!(struct_of_lists.is_constant());
    }

    #[rstest]
    #[case(
        // [1,2], [1, 2], [1, 2]
        vec![1i32, 2, 1, 2, 1, 2],
        vec![0u32, 2, 4, 6],
        true
    )]
    #[case(
        // [1, 2], [3], [4, 5]
        vec![1i32, 2, 3, 4, 5],
        vec![0u32, 2, 3, 5],
        false
    )]
    #[case(
        // [1, 2], [3, 4]
        vec![1i32, 2, 3, 4],
        vec![0u32, 2, 4],
        false
    )]
    #[case(
        // [], [], []
        vec![],
        vec![0u32, 0, 0, 0],
        true
    )]
    fn test_list_is_constant(
        #[case] elements: Vec<i32>,
        #[case] offsets: Vec<u32>,
        #[case] expected: bool,
    ) {
        let list_array = ListArray::try_new(
            PrimitiveArray::from_iter(elements).into_array(),
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let result = is_constant(&list_array.into_array()).unwrap();
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_list_is_constant_nested_lists() {
        let inner_elements = PrimitiveArray::from_iter([1i32, 2, 1, 2]).into_array();
        let inner_offsets = PrimitiveArray::from_iter([0u32, 1, 2, 3, 4]).into_array();
        let inner_lists =
            ListArray::try_new(inner_elements, inner_offsets, Validity::NonNullable).unwrap();

        let outer_offsets = PrimitiveArray::from_iter([0u32, 2, 4]).into_array();
        let outer_list = ListArray::try_new(
            inner_lists.into_array(),
            outer_offsets,
            Validity::NonNullable,
        )
        .unwrap();

        // Both outer lists contain [[1], [2]], so should be constant
        assert!(is_constant(&outer_list.into_array()).unwrap().unwrap());
    }

    #[rstest]
    #[case(
        // 100 identical [1, 2] lists
        [1i32, 2].repeat(100),
        (0..101).map(|i| (i * 2) as u32).collect(),
        true
    )]
    #[case(
        // Difference after threshold: 64 identical [1, 2] + one [3, 4]
        {
            let mut elements = [1i32, 2].repeat(64);
            elements.extend_from_slice(&[3, 4]);
            elements
        },
        (0..66).map(|i| (i * 2) as u32).collect(),
        false
    )]
    #[case(
        // Difference in first 64: first 63 identical [1, 2] + one [3, 4] + rest identical [1, 2]
        {
            let mut elements = [1i32, 2].repeat(63);
            elements.extend_from_slice(&[3, 4]);
            elements.extend([1i32, 2].repeat(37));
            elements
        },
        (0..101).map(|i| (i * 2) as u32).collect(),
        false
    )]
    fn test_large_list_is_constant(
        #[case] elements: Vec<i32>,
        #[case] offsets: Vec<u32>,
        #[case] expected: bool,
    ) {
        let list_array = ListArray::try_new(
            PrimitiveArray::from_iter(elements).into_array(),
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let result = is_constant(&list_array.into_array()).unwrap();
        assert_eq!(result.unwrap(), expected);
    }
}
