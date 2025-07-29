// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for consistency between related compute operations

use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, PrimitiveArray};
use crate::compute::{filter, mask, take};
use crate::{Array, IntoArray};

/// Tests that filter and take operations produce consistent results
/// filter(array, mask) should equal take(array, indices_where_mask_is_true)
pub fn test_filter_take_consistency(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Create a test mask
    let mask_pattern: Vec<bool> = (0..len).map(|i| i % 3 != 1).collect();
    let mask = Mask::try_from(&BoolArray::from_iter(mask_pattern.clone())).vortex_unwrap();

    // Filter the array
    let filtered = filter(array, &mask).vortex_unwrap();

    // Create indices where mask is true
    let indices: Vec<u64> = mask_pattern
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| v.then_some(i as u64))
        .collect();
    let indices_array = PrimitiveArray::from_iter(indices).into_array();

    // Take using those indices
    let taken = take(array, &indices_array).vortex_unwrap();

    // Results should be identical
    assert_eq!(filtered.len(), taken.len());
    for i in 0..filtered.len() {
        assert_eq!(
            filtered.scalar_at(i).vortex_unwrap(),
            taken.scalar_at(i).vortex_unwrap()
        );
    }
}

/// Tests that double masking is consistent with combined mask
/// mask(mask(array, mask1), mask2) should equal mask(array, mask1 | mask2)
pub fn test_double_mask_consistency(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Create two different mask patterns
    let mask1_pattern: Vec<bool> = (0..len).map(|i| i % 3 == 0).collect();
    let mask2_pattern: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();

    let mask1 = Mask::try_from(&BoolArray::from_iter(mask1_pattern.clone())).vortex_unwrap();
    let mask2 = Mask::try_from(&BoolArray::from_iter(mask2_pattern.clone())).vortex_unwrap();

    // Apply masks sequentially
    let first_masked = mask(array, &mask1).vortex_unwrap();
    let double_masked = mask(&first_masked, &mask2).vortex_unwrap();

    // Create combined mask (OR operation)
    let combined_pattern: Vec<bool> = mask1_pattern
        .iter()
        .zip(mask2_pattern.iter())
        .map(|(&a, &b)| a || b)
        .collect();
    let combined_mask = Mask::try_from(&BoolArray::from_iter(combined_pattern)).vortex_unwrap();

    // Apply combined mask directly
    let directly_masked = mask(array, &combined_mask).vortex_unwrap();

    // Results should be identical
    assert_eq!(double_masked.len(), directly_masked.len());
    for i in 0..double_masked.len() {
        assert_eq!(
            double_masked.scalar_at(i).vortex_unwrap(),
            directly_masked.scalar_at(i).vortex_unwrap()
        );
    }
}

/// Tests consistency when filtering with all-true mask vs no operation
pub fn test_filter_identity(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    let all_true_mask = Mask::new_true(len);
    let filtered = filter(array, &all_true_mask).vortex_unwrap();

    // Filtered array should be identical to original
    assert_eq!(filtered.len(), array.len());
    for i in 0..len {
        assert_eq!(
            filtered.scalar_at(i).vortex_unwrap(),
            array.scalar_at(i).vortex_unwrap()
        );
    }
}

/// Tests consistency when masking with all-false mask vs no masking
pub fn test_mask_identity(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    let all_false_mask = Mask::new_false(len);
    let masked = mask(array, &all_false_mask).vortex_unwrap();

    // Masked array should have same values (just nullable)
    assert_eq!(masked.len(), array.len());
    for i in 0..len {
        assert_eq!(
            masked.scalar_at(i).vortex_unwrap(),
            array.scalar_at(i).vortex_unwrap().into_nullable()
        );
    }
}

/// Tests that slice and filter with contiguous mask produce same results
pub fn test_slice_filter_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 4 {
        return;
    }

    // Create a contiguous mask (true from index 1 to 3)
    let mut mask_pattern = vec![false; len];
    mask_pattern[1..4.min(len)].fill(true);

    let mask = Mask::try_from(&BoolArray::from_iter(mask_pattern)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();

    // Slice should produce the same result
    let sliced = array.slice(1, 4.min(len)).vortex_unwrap();

    assert_eq!(filtered.len(), sliced.len());
    for i in 0..filtered.len() {
        assert_eq!(
            filtered.scalar_at(i).vortex_unwrap(),
            sliced.scalar_at(i).vortex_unwrap()
        );
    }
}

/// Tests that take with sequential indices equals slice
pub fn test_take_slice_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 3 {
        return;
    }

    // Take indices [1, 2, 3]
    let end = 4.min(len);
    let indices = PrimitiveArray::from_iter((1..end).map(|i| i as u64)).into_array();
    let taken = take(array, &indices).vortex_unwrap();

    // Slice from 1 to end
    let sliced = array.slice(1, end).vortex_unwrap();

    assert_eq!(taken.len(), sliced.len());
    for i in 0..taken.len() {
        assert_eq!(
            taken.scalar_at(i).vortex_unwrap(),
            sliced.scalar_at(i).vortex_unwrap()
        );
    }
}

/// Tests that filter preserves relative ordering
pub fn test_filter_preserves_order(array: &dyn Array) {
    let len = array.len();
    if len < 4 {
        return;
    }

    // Create a mask that selects elements at indices 0, 2, 3
    let mask_pattern: Vec<bool> = (0..len).map(|i| i == 0 || i == 2 || i == 3).collect();
    let mask = Mask::try_from(&BoolArray::from_iter(mask_pattern)).vortex_unwrap();

    let filtered = filter(array, &mask).vortex_unwrap();

    // Verify the filtered array contains the right elements in order
    assert_eq!(filtered.len(), 3.min(len));
    if len >= 4 {
        assert_eq!(
            filtered.scalar_at(0).vortex_unwrap(),
            array.scalar_at(0).vortex_unwrap()
        );
        assert_eq!(
            filtered.scalar_at(1).vortex_unwrap(),
            array.scalar_at(2).vortex_unwrap()
        );
        assert_eq!(
            filtered.scalar_at(2).vortex_unwrap(),
            array.scalar_at(3).vortex_unwrap()
        );
    }
}

/// Tests that take with repeated indices works correctly
pub fn test_take_repeated_indices(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Take the first element three times
    let indices = PrimitiveArray::from_iter([0u64, 0, 0]).into_array();
    let taken = take(array, &indices).vortex_unwrap();

    assert_eq!(taken.len(), 3);
    for i in 0..3 {
        assert_eq!(
            taken.scalar_at(i).vortex_unwrap(),
            array.scalar_at(0).vortex_unwrap()
        );
    }
}

/// Tests mask and filter interaction with nulls
pub fn test_mask_filter_null_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 3 {
        return;
    }

    // First mask some elements
    let mask_pattern: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let mask_array = Mask::try_from(&BoolArray::from_iter(mask_pattern)).vortex_unwrap();
    let masked = mask(array, &mask_array).vortex_unwrap();

    // Then filter to remove the nulls
    let filter_pattern: Vec<bool> = (0..len).map(|i| i % 2 != 0).collect();
    let filter_mask = Mask::try_from(&BoolArray::from_iter(filter_pattern)).vortex_unwrap();
    let filtered = filter(&masked, &filter_mask).vortex_unwrap();

    // This should be equivalent to directly filtering the original array
    let direct_filtered = filter(array, &filter_mask).vortex_unwrap();

    assert_eq!(filtered.len(), direct_filtered.len());
    for i in 0..filtered.len() {
        assert_eq!(
            filtered.scalar_at(i).vortex_unwrap(),
            direct_filtered.scalar_at(i).vortex_unwrap()
        );
    }
}

/// Tests that empty operations are consistent
pub fn test_empty_operations_consistency(array: &dyn Array) {
    let len = array.len();

    // Empty filter
    let empty_filter = filter(array, &Mask::new_false(len)).vortex_unwrap();
    assert_eq!(empty_filter.len(), 0);
    assert_eq!(empty_filter.dtype(), array.dtype());

    // Empty take
    let empty_indices =
        PrimitiveArray::empty::<u64>(vortex_dtype::Nullability::NonNullable).into_array();
    let empty_take = take(array, &empty_indices).vortex_unwrap();
    assert_eq!(empty_take.len(), 0);
    assert_eq!(empty_take.dtype(), array.dtype());

    // Empty slice (if array is non-empty)
    if len > 0 {
        let empty_slice = array.slice(0, 0).vortex_unwrap();
        assert_eq!(empty_slice.len(), 0);
        assert_eq!(empty_slice.dtype(), array.dtype());
    }
}

/// Tests that take preserves array properties
pub fn test_take_preserves_properties(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Take all elements in original order
    let indices = PrimitiveArray::from_iter((0..len).map(|i| i as u64)).into_array();
    let taken = take(array, &indices).vortex_unwrap();

    // Should be identical to original
    assert_eq!(taken.len(), array.len());
    assert_eq!(taken.dtype(), array.dtype());
    for i in 0..len {
        assert_eq!(
            taken.scalar_at(i).vortex_unwrap(),
            array.scalar_at(i).vortex_unwrap()
        );
    }
}

/// Tests consistency with nullable indices
pub fn test_nullable_indices_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 3 {
        return;
    }

    // Create nullable indices where some indices are null
    let indices = PrimitiveArray::from_option_iter([Some(0u64), None, Some(2u64)]).into_array();

    let taken = take(array, &indices).vortex_unwrap();

    // Result should have nulls where indices were null
    assert_eq!(taken.len(), 3);
    assert!(taken.dtype().is_nullable());
    assert_eq!(
        taken.scalar_at(0).vortex_unwrap(),
        array.scalar_at(0).vortex_unwrap().into_nullable()
    );
    assert!(taken.scalar_at(1).vortex_unwrap().is_null());
    assert_eq!(
        taken.scalar_at(2).vortex_unwrap(),
        array.scalar_at(2).vortex_unwrap().into_nullable()
    );
}

/// Tests large array consistency
pub fn test_large_array_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 1000 {
        return;
    }

    // Test with every 10th element
    let indices: Vec<u64> = (0..len).step_by(10).map(|i| i as u64).collect();
    let indices_array = PrimitiveArray::from_iter(indices).into_array();
    let taken = take(array, &indices_array).vortex_unwrap();

    // Create equivalent filter mask
    let mask_pattern: Vec<bool> = (0..len).map(|i| i % 10 == 0).collect();
    let mask = Mask::try_from(&BoolArray::from_iter(mask_pattern)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();

    // Results should match
    assert_eq!(taken.len(), filtered.len());
    for i in 0..taken.len() {
        assert_eq!(
            taken.scalar_at(i).vortex_unwrap(),
            filtered.scalar_at(i).vortex_unwrap()
        );
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::ArrayRef;
    use crate::arrays::BoolArray;

    #[rstest]
    #[case::primitive_i32(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array())]
    #[case::nullable_i32(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]).into_array())]
    #[case::primitive_i64(PrimitiveArray::from_iter(0..100i64).into_array())]
    #[case::bool_array(BoolArray::from_iter([true, false, true, true, false]).into_array())]
    #[case::nullable_bool(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]).into_array())]
    #[case::large_array(PrimitiveArray::from_iter(0..2000u32).into_array())]
    #[case::single_element(PrimitiveArray::from_iter([42i32]).into_array())]
    #[case::two_elements(PrimitiveArray::from_iter([1u64, 2]).into_array())]
    fn test_all_consistency(#[case] array: ArrayRef) {
        // Core consistency tests
        test_filter_take_consistency(array.as_ref());
        test_double_mask_consistency(array.as_ref());
        test_filter_identity(array.as_ref());
        test_mask_identity(array.as_ref());

        // Additional consistency tests
        test_slice_filter_consistency(array.as_ref());
        test_take_slice_consistency(array.as_ref());
        test_filter_preserves_order(array.as_ref());
        test_take_repeated_indices(array.as_ref());
        test_mask_filter_null_consistency(array.as_ref());
        test_empty_operations_consistency(array.as_ref());
        test_take_preserves_properties(array.as_ref());
        test_nullable_indices_consistency(array.as_ref());
        test_large_array_consistency(array.as_ref());
    }
}
