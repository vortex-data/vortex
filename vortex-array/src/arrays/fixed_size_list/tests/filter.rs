// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBuffer;
use rstest::rstest;
use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_mask::Mask;

use crate::arrays::{ConstantVTable, FixedSizeListArray, FixedSizeListVTable, PrimitiveArray};
use crate::compute::conformance::filter::{
    LARGE_SIZE, MEDIUM_SIZE, SMALL_SIZE, test_filter_conformance,
};
use crate::compute::filter;
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray};

#[test]
fn test_filter_all_true() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let mask = Mask::from(BooleanBuffer::from(vec![true, true, true]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 3);
    assert_eq!(filtered_fsl.list_size(), 2);
    assert_eq!(filtered_fsl.elements().len(), 6);

    // Verify the data is unchanged.
    assert_eq!(filtered_fsl.scalar_at(0), fsl.scalar_at(0));
    assert_eq!(filtered_fsl.scalar_at(1), fsl.scalar_at(1));
    assert_eq!(filtered_fsl.scalar_at(2), fsl.scalar_at(2));
}

#[test]
fn test_filter_all_false() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let mask = Mask::from(BooleanBuffer::from(vec![false, false, false]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 0);
    assert_eq!(filtered_fsl.list_size(), 2);
    assert_eq!(filtered_fsl.elements().len(), 0);
}

#[test]
fn test_filter_alternating() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 4);

    let mask = Mask::from(BooleanBuffer::from(vec![true, false, true, false]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 2);
    assert_eq!(filtered_fsl.list_size(), 2);
    assert_eq!(filtered_fsl.elements().len(), 4);

    // First list should be [1, 2].
    let first = filtered_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), 1i32.into());
    assert_eq!(first.scalar_at(1), 2i32.into());

    // Second list should be [5, 6].
    let second = filtered_fsl.fixed_size_list_elements_at(1);
    assert_eq!(second.scalar_at(0), 5i32.into());
    assert_eq!(second.scalar_at(1), 6i32.into());
}

#[test]
fn test_filter_selective() {
    let elements = buffer![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].into_array();
    let fsl = FixedSizeListArray::new(elements, 3, Validity::NonNullable, 3);

    let mask = Mask::from(BooleanBuffer::from(vec![false, true, true]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 2);
    assert_eq!(filtered_fsl.list_size(), 3);
    assert_eq!(filtered_fsl.elements().len(), 6);

    // First list should be [4.0, 5.0, 6.0].
    let first = filtered_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), 4.0f64.into());
    assert_eq!(first.scalar_at(1), 5.0f64.into());
    assert_eq!(first.scalar_at(2), 6.0f64.into());

    // Second list should be [7.0, 8.0, 9.0].
    let second = filtered_fsl.fixed_size_list_elements_at(1);
    assert_eq!(second.scalar_at(0), 7.0f64.into());
    assert_eq!(second.scalar_at(1), 8.0f64.into());
    assert_eq!(second.scalar_at(2), 9.0f64.into());
}

#[test]
fn test_filter_empty_array() {
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 0);

    let mask = Mask::AllTrue(0);
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 0);
    assert_eq!(filtered_fsl.list_size(), 2);
    assert_eq!(filtered_fsl.elements().len(), 0);
}

#[test]
fn test_filter_single_element() {
    let elements = buffer![42i32, 43, 44].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 3, Validity::NonNullable, 1);

    // Keep the single element.
    let mask = Mask::from(BooleanBuffer::from(vec![true]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 1);
    assert_eq!(filtered_fsl.list_size(), 3);
    assert_eq!(filtered_fsl.elements().len(), 3);

    let first = filtered_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), 42i32.into());
    assert_eq!(first.scalar_at(1), 43i32.into());
    assert_eq!(first.scalar_at(2), 44i32.into());

    // Filter out the single element.
    let mask = Mask::from(BooleanBuffer::from(vec![false]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 0);
    assert_eq!(filtered_fsl.list_size(), 3);
    assert_eq!(filtered_fsl.elements().len(), 0);
}

// Consolidated parameterized test for degenerate (list_size=0) cases.
#[rstest]
#[case::basic_degenerate(
    5,
    Validity::NonNullable,
    vec![true, false, true, false, true],
    3,
    false
)]
#[case::degenerate_with_nulls(
    5,
    Validity::from_iter([true, false, true, true, false]),
    vec![true, false, true, true, false],
    3,
    false
)]
#[case::degenerate_to_empty(
    3,
    Validity::NonNullable,
    vec![false, false, false],
    0,
    false
)]
#[case::degenerate_all_null(
    4,
    Validity::AllInvalid,
    vec![true, true, false, true],
    3,
    true
)]
#[case::large_degenerate(
    1000,
    Validity::NonNullable,
    vec![true; 500].into_iter().chain(vec![false; 500]).collect(),
    500,
    false
)]
fn test_filter_degenerate_list_size_zero(
    #[case] num_lists: usize,
    #[case] validity: Validity,
    #[case] mask_values: Vec<bool>,
    #[case] expected_len: usize,
    #[case] expect_constant_array: bool,
) {
    // Degenerate case where list_size == 0.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), 0, validity, num_lists);

    let mask = Mask::from(BooleanBuffer::from(mask_values));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();

    assert_eq!(filtered.len(), expected_len);

    if expect_constant_array {
        // Should return a ConstantArray of nulls when all are invalid.
        let filtered_const = filtered.as_::<ConstantVTable>();
        for i in 0..expected_len {
            assert!(filtered_const.scalar_at(i).is_null());
        }
    } else {
        let filtered_fsl = filtered.as_::<FixedSizeListVTable>();
        assert_eq!(filtered_fsl.list_size(), 0);
        assert_eq!(filtered_fsl.elements().len(), 0);
    }
}

#[test]
fn test_filter_with_nulls() {
    let elements =
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(4), Some(5), Some(6)]);
    let validity = Validity::from_iter([true, false, true]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, validity, 3);

    let mask = Mask::from(BooleanBuffer::from(vec![true, false, true]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 2);
    assert_eq!(filtered_fsl.list_size(), 2);

    // First list should be [1, 2] and valid.
    let first = filtered_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), 1i32.into());
    assert_eq!(first.scalar_at(1), 2i32.into());

    // Second list should be [5, 6] and valid.
    let second = filtered_fsl.fixed_size_list_elements_at(1);
    assert_eq!(second.scalar_at(0), 5i32.into());
    assert_eq!(second.scalar_at(1), 6i32.into());
}

#[test]
fn test_filter_all_null_array() {
    // Create an array where all elements are null.
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let validity = Validity::AllInvalid;
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, validity, 3);

    let mask = Mask::from(BooleanBuffer::from(vec![true, false, true]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();

    // This should return a ConstantArray of nulls.
    let filtered_const = filtered.as_::<ConstantVTable>();
    assert_eq!(filtered_const.len(), 2);
    assert!(filtered_const.scalar_at(0).is_null());
    assert!(filtered_const.scalar_at(1).is_null());
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
    let mask = Mask::from(BooleanBuffer::from(vec![false, true]));
    let filtered = filter(outer_fsl.as_ref(), &mask).unwrap();
    let filtered_outer = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_outer.len(), 1);
    assert_eq!(filtered_outer.list_size(), 3);

    // The inner array should also be filtered appropriately.
    let filtered_inner = filtered_outer.elements().as_::<FixedSizeListVTable>();
    assert_eq!(filtered_inner.len(), 3);
    assert_eq!(filtered_inner.list_size(), 2);

    // Check the actual values.
    let inner_list_0 = filtered_inner.fixed_size_list_elements_at(0);
    assert_eq!(inner_list_0.scalar_at(0), 7i32.into());
    assert_eq!(inner_list_0.scalar_at(1), 8i32.into());

    let inner_list_1 = filtered_inner.fixed_size_list_elements_at(1);
    assert_eq!(inner_list_1.scalar_at(0), 9i32.into());
    assert_eq!(inner_list_1.scalar_at(1), 10i32.into());

    let inner_list_2 = filtered_inner.fixed_size_list_elements_at(2);
    assert_eq!(inner_list_2.scalar_at(0), 11i32.into());
    assert_eq!(inner_list_2.scalar_at(1), 12i32.into());
}

// Parameterized test for different mask types.
#[rstest]
#[case::all_true(Mask::AllTrue(3), 3)]
#[case::all_false(Mask::AllFalse(3), 0)]
#[case::values_partial(Mask::from(BooleanBuffer::from(vec![true, true, false])), 2)]
#[case::values_alternating(Mask::from(BooleanBuffer::from(vec![true, false, true])), 2)]
fn test_filter_mask_types(#[case] mask: Mask, #[case] expected_len: usize) {
    use vortex_buffer::buffer;

    let elements = buffer![1u32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    assert_eq!(filtered.len(), expected_len);
}

#[test]
fn test_filter_preserves_dtype() {
    let elements = buffer![1.5f32, 2.5, 3.5, 4.5].into_array();
    let fsl = FixedSizeListArray::new(elements, 2, Validity::AllValid, 2);

    let mask = Mask::from(BooleanBuffer::from(vec![true, false]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    // Check that the dtype is preserved.
    assert!(matches!(
        filtered_fsl.dtype(),
        DType::FixedSizeList(elem_dtype, 2, Nullability::Nullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::F32, Nullability::NonNullable))
    ));
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
    test_filter_conformance(array.as_ref());
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
fn test_complex_filter_pattern() {
    // Create a larger array to test complex patterns.
    let elements = buffer![0..30i32].into_array();
    let fsl = FixedSizeListArray::new(elements, 3, Validity::NonNullable, 10);

    // Complex mask pattern: [T, T, F, T, F, F, T, T, T, F].
    let mask = Mask::from(BooleanBuffer::from(vec![
        true, true, false, true, false, false, true, true, true, false,
    ]));

    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 6);
    assert_eq!(filtered_fsl.list_size(), 3);
    assert_eq!(filtered_fsl.elements().len(), 18);

    // Verify the filtered lists are correct.
    // Original indices kept: 0, 1, 3, 6, 7, 8.
    let expected_starts = [0i32, 3, 9, 18, 21, 24];
    for (i, &start) in expected_starts.iter().enumerate() {
        let list = filtered_fsl.fixed_size_list_elements_at(i);
        assert_eq!(list.scalar_at(0), (start).into());
        assert_eq!(list.scalar_at(1), (start + 1).into());
        assert_eq!(list.scalar_at(2), (start + 2).into());
    }
}

#[test]
fn test_filter_large_list_size() {
    // Test with a large list size.
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

    let mask = Mask::from(BooleanBuffer::from(vec![false, true, false, true, true]));
    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 3);
    assert_eq!(filtered_fsl.list_size(), list_size);
    assert_eq!(filtered_fsl.elements().len(), 3 * list_size as usize);

    // Check that the correct lists were kept (indices 1, 3, 4).
    let list_0 = filtered_fsl.fixed_size_list_elements_at(0);
    assert_eq!(list_0.scalar_at(0), 100i64.into()); // Start of original list 1.

    let list_1 = filtered_fsl.fixed_size_list_elements_at(1);
    assert_eq!(list_1.scalar_at(0), 300i64.into()); // Start of original list 3.

    let list_2 = filtered_fsl.fixed_size_list_elements_at(2);
    assert_eq!(list_2.scalar_at(0), 400i64.into()); // Start of original list 4.
}

// Note: test_filter_degenerate_with_nulls and test_filter_degenerate_all_null
// have been consolidated into the parameterized test_filter_degenerate_list_size_zero above.

#[test]
fn test_filter_all_null_various_list_sizes() {
    // Test filtering with all-null arrays of different list sizes.
    // The implementation returns ConstantArray only when validity_mask() is Mask::AllFalse.

    // Case 1: list_size == 0
    let elements0 = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl0 = FixedSizeListArray::new(elements0.into_array(), 0, Validity::AllInvalid, 3);
    let mask0 = Mask::from(BooleanBuffer::from(vec![true, false, true]));
    let filtered0 = filter(fsl0.as_ref(), &mask0).unwrap();
    assert_eq!(filtered0.len(), 2);
    // Check that all elements are null (might be ConstantArray or FixedSizeListArray)
    assert!(filtered0.scalar_at(0).is_null());
    assert!(filtered0.scalar_at(1).is_null());

    // Case 2: list_size == 1
    let elements1 = buffer![1i32, 2, 3].into_array();
    let fsl1 = FixedSizeListArray::new(elements1.into_array(), 1, Validity::AllInvalid, 3);
    let mask1 = Mask::from(BooleanBuffer::from(vec![false, true, true]));
    let filtered1 = filter(fsl1.as_ref(), &mask1).unwrap();
    assert_eq!(filtered1.len(), 2);
    // Check that all elements are null
    assert!(filtered1.scalar_at(0).is_null());
    assert!(filtered1.scalar_at(1).is_null());

    // Case 3: list_size == 10 (large)
    let elements10 = buffer![0..50i32].into_array();
    let fsl10 = FixedSizeListArray::new(elements10, 10, Validity::AllInvalid, 5);
    let mask10 = Mask::AllTrue(5);
    let filtered10 = filter(fsl10.as_ref(), &mask10).unwrap();
    assert_eq!(filtered10.len(), 5);
    // Check that all elements are null
    assert!(filtered10.scalar_at(0).is_null());
    assert!(filtered10.scalar_at(4).is_null());
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
    let mask = Mask::from(BooleanBuffer::from(sparse_mask));

    let filtered = filter(fsl.as_ref(), &mask).unwrap();
    let filtered_fsl = filtered.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl.len(), 3);
    assert_eq!(filtered_fsl.list_size(), list_size);
    assert_eq!(filtered_fsl.elements().len(), 3 * list_size as usize);

    // Verify correct elements were kept.
    let first = filtered_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), (5i32 * list_size as i32).into());

    let second = filtered_fsl.fixed_size_list_elements_at(1);
    assert_eq!(second.scalar_at(0), (25i32 * list_size as i32).into());

    let third = filtered_fsl.fixed_size_list_elements_at(2);
    assert_eq!(third.scalar_at(0), (75i32 * list_size as i32).into());

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

    let filtered7 = filter(fsl7.as_ref(), &mask).unwrap();
    let filtered_fsl7 = filtered7.as_::<FixedSizeListVTable>();

    assert_eq!(filtered_fsl7.len(), 3);
    assert_eq!(filtered_fsl7.list_size(), list_size_7);
}

// Note: test_nested_degenerate_filter has been consolidated into test_filter_nested_fixed_size_lists.

// Note: test_large_degenerate_array has been consolidated into test_filter_degenerate_list_size_zero above.

// Note: test_degenerate_all_mask_types has been consolidated into the parameterized tests above.
