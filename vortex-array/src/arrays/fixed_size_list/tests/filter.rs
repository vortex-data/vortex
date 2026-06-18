// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::assert_nth_scalar_is_null;
use crate::compute::conformance::filter::LARGE_SIZE;
use crate::compute::conformance::filter::MEDIUM_SIZE;
use crate::compute::conformance::filter::SMALL_SIZE;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::dtype::Nullability;
use crate::validity::Validity;

// Consolidated parameterized test for degenerate (list_size=0) cases.
#[rstest]
#[case::basic_degenerate(
    5,
    Validity::NonNullable,
    vec![true, false, true, false, true],
    3
)]
#[case::degenerate_with_nulls(
    5,
    Validity::from_iter([true, false, true, true, false]),
    vec![true, false, true, true, false],
    3
)]
#[case::degenerate_to_empty(
    3,
    Validity::NonNullable,
    vec![false, false, false],
    0
)]
#[case::large_degenerate(
    1000,
    Validity::NonNullable,
    vec![true; 500].into_iter().chain(vec![false; 500]).collect(),
    500,
)]
fn test_filter_degenerate_list_size_zero(
    #[case] num_lists: usize,
    #[case] validity: Validity,
    #[case] mask_values: Vec<bool>,
    #[case] expected_len: usize,
) {
    let new_validity = if matches!(validity, Validity::NonNullable) {
        Validity::NonNullable
    } else {
        Validity::AllValid
    };

    // Degenerate case where list_size == 0.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), 0, validity, num_lists);

    let mask = Mask::from(BitBuffer::from(mask_values));
    let filtered = fsl.filter(mask).unwrap();

    assert_eq!(filtered.len(), expected_len, "Degenerate FSL filter failed");

    // For degenerate list_size=0, verify the filtered result matches expected.
    let expected_elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let expected = FixedSizeListArray::new(
        expected_elements.into_array(),
        0,
        new_validity,
        expected_len,
    );
    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_filter_with_nulls() {
    let elements =
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(4), Some(5), Some(6)]);
    let validity = Validity::from_iter([true, false, true]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, validity, 3);

    let mask = Mask::from(BitBuffer::from(vec![true, false, true]));
    let filtered = fsl.filter(mask).unwrap();

    assert_eq!(filtered.len(), 2, "Expected lists after filtering out null");

    // Construct expected: first and third lists from original (indices 0 and 2).
    // First list: [1, 2], Third list: [5, 6].
    // Both selected lists are valid, but the dtype remains nullable since the original was nullable.
    let expected_elements =
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(5), Some(6)]);
    let expected = FixedSizeListArray::new(
        expected_elements.into_array(),
        2,
        Validity::from_iter([true, true]), /* Both selected lists are valid, but type is still nullable. */
        2,
    );

    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_filter_all_null_array() {
    // Create an array where all elements are null.
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let validity = Validity::AllInvalid;
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, validity, 3);

    let mask = Mask::from(BitBuffer::from(vec![true, false, true]));
    let filtered = fsl.filter(mask).unwrap();

    // Verify the result is an array of nulls.
    assert_eq!(filtered.len(), 2, "All-null FSL should produce 2 elements");
    assert_nth_scalar_is_null!(filtered, 0);
    assert_nth_scalar_is_null!(filtered, 1);
}

#[test]
fn test_filter_nested_fixed_size_lists() {
    // Create nested fixed-size lists: FSL<FSL<i32>>.
    // Inner lists are of size 2, outer lists are of size 3.
    // So we have 2 outer lists, each containing 3 inner lists, each containing 2 i32s.
    let inner_elements = buffer![
        1i32, 2, // First inner list of first outer list.
        3, 4, // Second inner list of first outer list.
        5, 6, // Third inner list of first outer list.
        7, 8, // First inner list of second outer list.
        9, 10, // Second inner list of second outer list.
        11, 12, // Third inner list of second outer list.
    ]
    .into_array();

    let inner_fsl = FixedSizeListArray::new(
        inner_elements,
        2, // Inner list size.
        Validity::NonNullable,
        6, // 6 inner lists total.
    );

    let outer_fsl = FixedSizeListArray::new(
        inner_fsl.into_array(),
        3, // Outer list size (3 inner lists per outer list).
        Validity::NonNullable,
        2, // 2 outer lists.
    );

    // Filter to keep only the second outer list.
    let mask = Mask::from(BitBuffer::from(vec![false, true]));
    let filtered = outer_fsl.filter(mask).unwrap();

    // Construct expected: second outer list [[7,8], [9,10], [11,12]].
    let expected_inner_elements = buffer![7i32, 8, 9, 10, 11, 12].into_array();
    let expected_inner =
        FixedSizeListArray::new(expected_inner_elements, 2, Validity::NonNullable, 3);
    let expected_outer =
        FixedSizeListArray::new(expected_inner.into_array(), 3, Validity::NonNullable, 1);

    assert_arrays_eq!(filtered, expected_outer);
}

// Conformance tests using rstest for various array configurations.
#[rstest]
#[case(create_fsl_i32(SMALL_SIZE))]
#[case(create_fsl_f64(SMALL_SIZE))]
#[case(create_fsl_nullable(SMALL_SIZE))]
#[case(create_fsl_i32(MEDIUM_SIZE))]
#[case(create_fsl_f64(MEDIUM_SIZE))]
#[case(create_fsl_nullable(MEDIUM_SIZE))]
#[case(create_fsl_i32(LARGE_SIZE))]
#[case(create_fsl_mixed_nulls())]
#[case(create_fsl_single_element())]
#[case(create_fsl_empty())]
fn test_filter_fsl_conformance(#[case] array: ArrayRef) {
    test_filter_conformance(&array);
}

// Helper functions for creating test arrays.
fn create_fsl_i32(num_lists: usize) -> ArrayRef {
    let list_size = 3u32;
    let elements: Vec<i32> = (0..(num_lists * list_size as usize))
        .map(|i| i32::try_from(i).unwrap())
        .collect();
    let elements = PrimitiveArray::from_iter(elements);
    FixedSizeListArray::new(
        elements.into_array(),
        list_size,
        Validity::NonNullable,
        num_lists,
    )
    .into_array()
}

fn create_fsl_f64(num_lists: usize) -> ArrayRef {
    let list_size = 4u32;
    let elements: Vec<f64> = (0..(num_lists * list_size as usize))
        .map(|i| i as f64 * 0.5)
        .collect();
    let elements = PrimitiveArray::from_iter(elements);
    FixedSizeListArray::new(
        elements.into_array(),
        list_size,
        Validity::NonNullable,
        num_lists,
    )
    .into_array()
}

fn create_fsl_nullable(num_lists: usize) -> ArrayRef {
    let list_size = 2u32;
    let elements: Vec<Option<i32>> = (0..(num_lists * list_size as usize))
        .map(|i| {
            if i % 5 == 0 {
                None
            } else {
                Some(i32::try_from(i).unwrap())
            }
        })
        .collect();
    let elements = PrimitiveArray::from_option_iter(elements);

    let validity = Validity::from_iter((0..num_lists).map(|i| i % 3 != 0));

    FixedSizeListArray::new(elements.into_array(), list_size, validity, num_lists).into_array()
}

fn create_fsl_mixed_nulls() -> ArrayRef {
    let elements =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None, Some(6)]);
    let validity = Validity::from_iter([true, false, true]);
    FixedSizeListArray::new(elements.into_array(), 2, validity, 3).into_array()
}

fn create_fsl_single_element() -> ArrayRef {
    let elements = buffer![100u64, 200, 300, 400, 500].into_array();
    FixedSizeListArray::new(elements.into_array(), 5, Validity::NonNullable, 1).into_array()
}

fn create_fsl_empty() -> ArrayRef {
    let elements = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
    FixedSizeListArray::new(elements.into_array(), 3, Validity::NonNullable, 0).into_array()
}

#[test]
fn test_filter_all_null_various_list_sizes() {
    // Test filtering with all-null arrays of different list sizes.
    // The implementation returns ConstantArray only when validity_mask() is Mask::AllFalse.

    // Case 1: list_size == 0
    let elements0 = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl0 = FixedSizeListArray::new(elements0.into_array(), 0, Validity::AllInvalid, 3);
    let mask0 = Mask::from(BitBuffer::from(vec![true, false, true]));
    let filtered0 = fsl0.filter(mask0).unwrap();
    assert_eq!(filtered0.len(), 2);
    // Check that all elements are null (might be ConstantArray or FixedSizeListArray).
    assert_nth_scalar_is_null!(filtered0, 0);
    assert_nth_scalar_is_null!(filtered0, 1);

    // Case 2: list_size == 1.
    let elements1 = buffer![1i32, 2, 3].into_array();
    let fsl1 = FixedSizeListArray::new(elements1.into_array(), 1, Validity::AllInvalid, 3);
    let mask1 = Mask::from(BitBuffer::from(vec![false, true, true]));
    let filtered1 = fsl1.filter(mask1).unwrap();
    assert_eq!(filtered1.len(), 2);
    // Check that all elements are null.
    assert_nth_scalar_is_null!(filtered1, 0);
    assert_nth_scalar_is_null!(filtered1, 1);

    // Case 3: list_size == 10 (large).
    let elements10 = buffer![0..50i32].into_array();
    let fsl10 = FixedSizeListArray::new(elements10, 10, Validity::AllInvalid, 5);
    let mask10 = Mask::AllTrue(5);
    let filtered10 = fsl10.filter(mask10).unwrap();
    assert_eq!(filtered10.len(), 5);
    // Check that all elements are null.
    assert_nth_scalar_is_null!(filtered10, 0);
    assert_nth_scalar_is_null!(filtered10, 4);
}

// Note: test_filter_to_empty_degenerate has been consolidated into test_filter_degenerate_list_size_zero above.

#[test]
fn test_mask_expansion_threshold_boundary() {
    // Test with list_size == 8 (the FSL_SPARSE_MASK_LIST_SIZE_THRESHOLD).
    let list_size = 8u32;
    let num_lists = 100;
    let elements: Vec<i32> = (0..(num_lists * list_size as usize))
        .map(|i| i32::try_from(i).unwrap())
        .collect();
    let elements = PrimitiveArray::from_iter(elements);
    let fsl = FixedSizeListArray::new(
        elements.into_array(),
        list_size,
        Validity::NonNullable,
        num_lists,
    );

    // Test with very sparse mask (density < 0.1).
    let mut sparse_mask = vec![false; num_lists];
    sparse_mask[5] = true;
    sparse_mask[25] = true;
    sparse_mask[75] = true;
    let mask = Mask::from(BitBuffer::from(sparse_mask));

    let filtered = fsl.filter(mask.clone()).unwrap();

    // Construct expected FSL with indices 5, 25, 75 from original.
    let expected_elements: Vec<i32> = [5, 25, 75]
        .iter()
        .flat_map(|&i| {
            let start = i * list_size as usize;
            (start..(start + list_size as usize)).map(|j| i32::try_from(j).unwrap())
        })
        .collect();
    let expected = FixedSizeListArray::new(
        PrimitiveArray::from_iter(expected_elements).into_array(),
        list_size,
        Validity::NonNullable,
        3,
    );

    assert_arrays_eq!(filtered, expected);

    // Test with list_size == 7 (just below threshold).
    let list_size_7 = 7u32;
    let elements7: Vec<i32> = (0..(num_lists * list_size_7 as usize))
        .map(|i| i32::try_from(i).unwrap())
        .collect();
    let elements7 = PrimitiveArray::from_iter(elements7);
    let fsl7 = FixedSizeListArray::new(
        elements7.into_array(),
        list_size_7,
        Validity::NonNullable,
        num_lists,
    );

    let filtered7 = fsl7.filter(mask).unwrap();

    // Construct expected FSL with indices 5, 25, 75 from original (list_size=7).
    let expected_elements7: Vec<i32> = [5, 25, 75]
        .iter()
        .flat_map(|&i| {
            let start = i * list_size_7 as usize;
            (start..(start + list_size_7 as usize)).map(|j| i32::try_from(j).unwrap())
        })
        .collect();
    let expected7 = FixedSizeListArray::new(
        PrimitiveArray::from_iter(expected_elements7).into_array(),
        list_size_7,
        Validity::NonNullable,
        3,
    );

    assert_arrays_eq!(filtered7, expected7);
}

// Test FSL-specific behavior with very large list sizes.
#[test]
fn test_filter_large_list_size() {
    // Test with list_size=100, which is significantly larger than typical use cases.
    let list_size = 100u32;
    let num_lists = 5;
    let elements: Vec<i64> = (0..(num_lists * list_size as usize))
        .map(|i| i as i64)
        .collect();
    let elements = PrimitiveArray::from_iter(elements);
    let fsl = FixedSizeListArray::new(
        elements.into_array(),
        list_size,
        Validity::NonNullable,
        num_lists,
    );

    // Apply a filter keeping lists 1, 3, 4.
    let mask = Mask::from_iter([false, true, false, true, true]);
    let filtered = fsl.filter(mask).unwrap();

    // Construct expected FSL with indices 1, 3, 4 from original.
    let expected_elements: Vec<i64> = [1, 3, 4]
        .iter()
        .flat_map(|&i| {
            let start = i * list_size as usize;
            (start..(start + list_size as usize)).map(|j| j as i64)
        })
        .collect();
    let expected = FixedSizeListArray::new(
        PrimitiveArray::from_iter(expected_elements).into_array(),
        list_size,
        Validity::NonNullable,
        3,
    );

    assert_arrays_eq!(filtered, expected);

    // Test edge case: filter out all but one large list.
    let mask_single = Mask::from_iter([false, false, true, false, false]);
    let filtered_single = fsl.filter(mask_single).unwrap();

    // Construct expected FSL with index 2 from original.
    let expected_single_elements: Vec<i64> = {
        let start = 2 * list_size as usize;
        (start..(start + list_size as usize))
            .map(|j| j as i64)
            .collect()
    };
    let expected_single = FixedSizeListArray::new(
        PrimitiveArray::from_iter(expected_single_elements).into_array(),
        list_size,
        Validity::NonNullable,
        1,
    );

    assert_arrays_eq!(filtered_single, expected_single);
}
