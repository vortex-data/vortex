// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # Array Consistency Tests
//!
//! This module contains tests that verify consistency between related compute operations
//! on Vortex arrays. These tests ensure that different ways of achieving the same result
//! produce identical outputs.
//!
//! ## Test Categories
//!
//! - **Filter/Take Consistency**: Verifies that filtering with a mask produces the same
//!   result as taking with the indices where the mask is true.
//! - **Mask Composition**: Ensures that applying multiple masks sequentially produces
//!   the same result as applying a combined mask.
//! - **Identity Operations**: Tests that operations with identity inputs (all-true masks,
//!   sequential indices) preserve the original array.
//! - **Null Handling**: Verifies consistent behavior when operations introduce or
//!   interact with null values.
//! - **Edge Cases**: Tests empty arrays, single elements, and boundary conditions.

use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, PrimitiveArray};
use crate::compute::{filter, mask, take};
use crate::{Array, IntoArray};

/// Tests that filter and take operations produce consistent results.
///
/// # Invariant
/// `filter(array, mask)` should equal `take(array, indices_where_mask_is_true)`
///
/// # Test Details
/// - Creates a mask that keeps elements where index % 3 != 1
/// - Applies filter with this mask
/// - Creates indices array containing positions where mask is true
/// - Applies take with these indices
/// - Verifies both results are identical
pub fn test_filter_take_consistency(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Create a test mask (keep elements where index % 3 != 1)
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
    assert_eq!(
        filtered.len(),
        taken.len(),
        "Filter and take should produce arrays of the same length. \
         Filtered length: {}, Taken length: {}",
        filtered.len(),
        taken.len()
    );

    for i in 0..filtered.len() {
        let filtered_val = filtered.scalar_at(i).vortex_unwrap();
        let taken_val = taken.scalar_at(i).vortex_unwrap();
        assert_eq!(
            filtered_val, taken_val,
            "Filter and take produced different values at index {i}. \
             Filtered value: {filtered_val:?}, Taken value: {taken_val:?}"
        );
    }
}

/// Tests that double masking is consistent with combined mask.
///
/// # Invariant
/// `mask(mask(array, mask1), mask2)` should equal `mask(array, mask1 | mask2)`
///
/// # Test Details
/// - Creates two masks: mask1 (every 3rd element) and mask2 (every 2nd element)
/// - Applies masks sequentially: first mask1, then mask2 on the result
/// - Creates a combined mask using OR operation (element is masked if either mask is true)
/// - Applies the combined mask directly to the original array
/// - Verifies both approaches produce identical results
///
/// # Why This Matters
/// This test ensures that mask operations compose correctly, which is critical for
/// complex query operations that may apply multiple filters.
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

    // Create combined mask (OR operation - element is masked if EITHER mask is true)
    let combined_pattern: Vec<bool> = mask1_pattern
        .iter()
        .zip(mask2_pattern.iter())
        .map(|(&a, &b)| a || b)
        .collect();
    let combined_mask = Mask::try_from(&BoolArray::from_iter(combined_pattern)).vortex_unwrap();

    // Apply combined mask directly
    let directly_masked = mask(array, &combined_mask).vortex_unwrap();

    // Results should be identical
    assert_eq!(
        double_masked.len(),
        directly_masked.len(),
        "Sequential masking and combined masking should produce arrays of the same length. \
         Sequential length: {}, Combined length: {}",
        double_masked.len(),
        directly_masked.len()
    );

    for i in 0..double_masked.len() {
        let double_val = double_masked.scalar_at(i).vortex_unwrap();
        let direct_val = directly_masked.scalar_at(i).vortex_unwrap();
        assert_eq!(
            double_val, direct_val,
            "Sequential masking and combined masking produced different values at index {i}. \
             Sequential masking value: {double_val:?}, Combined masking value: {direct_val:?}\n\
             This likely indicates an issue with how masks are composed in the array implementation."
        );
    }
}

/// Tests that filtering with an all-true mask preserves the array.
///
/// # Invariant
/// `filter(array, all_true_mask)` should equal `array`
///
/// # Test Details
/// - Creates a mask with all elements set to true
/// - Applies filter with this mask
/// - Verifies the result is identical to the original array
///
/// # Why This Matters
/// This is an identity operation that should be optimized in implementations
/// to avoid unnecessary copying.
pub fn test_filter_identity(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    let all_true_mask = Mask::new_true(len);
    let filtered = filter(array, &all_true_mask).vortex_unwrap();

    // Filtered array should be identical to original
    assert_eq!(
        filtered.len(),
        array.len(),
        "Filtering with all-true mask should preserve array length. \
         Original length: {}, Filtered length: {}",
        array.len(),
        filtered.len()
    );

    for i in 0..len {
        let original_val = array.scalar_at(i).vortex_unwrap();
        let filtered_val = filtered.scalar_at(i).vortex_unwrap();
        assert_eq!(
            filtered_val, original_val,
            "Filtering with all-true mask should preserve all values. \
             Value at index {i} changed from {original_val:?} to {filtered_val:?}"
        );
    }
}

/// Tests that masking with an all-false mask preserves values while making them nullable.
///
/// # Invariant
/// `mask(array, all_false_mask)` should have same values as `array` but with nullable type
///
/// # Test Details
/// - Creates a mask with all elements set to false (no elements are nullified)
/// - Applies mask operation
/// - Verifies all values are preserved but the array type becomes nullable
///
/// # Why This Matters
/// Masking always produces a nullable array, even when no values are actually masked.
/// This test ensures the type system handles this correctly.
pub fn test_mask_identity(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    let all_false_mask = Mask::new_false(len);
    let masked = mask(array, &all_false_mask).vortex_unwrap();

    // Masked array should have same values (just nullable)
    assert_eq!(
        masked.len(),
        array.len(),
        "Masking with all-false mask should preserve array length. \
         Original length: {}, Masked length: {}",
        array.len(),
        masked.len()
    );

    assert!(
        masked.dtype().is_nullable(),
        "Mask operation should always produce a nullable array, but dtype is {:?}",
        masked.dtype()
    );

    for i in 0..len {
        let original_val = array.scalar_at(i).vortex_unwrap();
        let masked_val = masked.scalar_at(i).vortex_unwrap();
        let expected_val = original_val.clone().into_nullable();
        assert_eq!(
            masked_val, expected_val,
            "Masking with all-false mask should preserve values (as nullable). \
             Value at index {i}: original = {original_val:?}, masked = {masked_val:?}, expected = {expected_val:?}"
        );
    }
}

/// Tests that slice and filter with contiguous mask produce same results.
///
/// # Invariant
/// `filter(array, contiguous_true_mask)` should equal `slice(array, start, end)`
///
/// # Test Details
/// - Creates a mask that is true only for indices 1, 2, and 3
/// - Filters the array with this mask
/// - Slices the array from index 1 to 4
/// - Verifies both operations produce identical results
///
/// # Why This Matters
/// When a filter mask represents a contiguous range, it should be equivalent to
/// a slice operation. Some implementations may optimize this case.
pub fn test_slice_filter_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 4 {
        return; // Need at least 4 elements for meaningful test
    }

    // Create a contiguous mask (true from index 1 to 3)
    let mut mask_pattern = vec![false; len];
    mask_pattern[1..4.min(len)].fill(true);

    let mask = Mask::try_from(&BoolArray::from_iter(mask_pattern)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();

    // Slice should produce the same result
    let sliced = array.slice(1, 4.min(len)).vortex_unwrap();

    assert_eq!(
        filtered.len(),
        sliced.len(),
        "Filter with contiguous mask and slice should produce same length. \
         Filtered length: {}, Sliced length: {}",
        filtered.len(),
        sliced.len()
    );

    for i in 0..filtered.len() {
        let filtered_val = filtered.scalar_at(i).vortex_unwrap();
        let sliced_val = sliced.scalar_at(i).vortex_unwrap();
        assert_eq!(
            filtered_val, sliced_val,
            "Filter with contiguous mask and slice produced different values at index {i}. \
             Filtered value: {filtered_val:?}, Sliced value: {sliced_val:?}"
        );
    }
}

/// Tests that take with sequential indices equals slice.
///
/// # Invariant
/// `take(array, [1, 2, 3, ...])` should equal `slice(array, 1, n)`
///
/// # Test Details
/// - Creates indices array with sequential values [1, 2, 3]
/// - Takes elements at these indices
/// - Slices array from index 1 to 4
/// - Verifies both operations produce identical results
///
/// # Why This Matters
/// Sequential takes are a common pattern that can be optimized to slice operations.
pub fn test_take_slice_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 3 {
        return; // Need at least 3 elements
    }

    // Take indices [1, 2, 3]
    let end = 4.min(len);
    let indices = PrimitiveArray::from_iter((1..end).map(|i| i as u64)).into_array();
    let taken = take(array, &indices).vortex_unwrap();

    // Slice from 1 to end
    let sliced = array.slice(1, end).vortex_unwrap();

    assert_eq!(
        taken.len(),
        sliced.len(),
        "Take with sequential indices and slice should produce same length. \
         Taken length: {}, Sliced length: {}",
        taken.len(),
        sliced.len()
    );

    for i in 0..taken.len() {
        let taken_val = taken.scalar_at(i).vortex_unwrap();
        let sliced_val = sliced.scalar_at(i).vortex_unwrap();
        assert_eq!(
            taken_val, sliced_val,
            "Take with sequential indices and slice produced different values at index {i}. \
             Taken value: {taken_val:?}, Sliced value: {sliced_val:?}"
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

/// Tests consistency with nullable indices.
///
/// # Invariant
/// `take(array, [Some(0), None, Some(2)])` should produce `[array[0], null, array[2]]`
///
/// # Test Details
/// - Creates an indices array with null at position 1: `[Some(0), None, Some(2)]`
/// - Takes elements using these indices
/// - Verifies that:
///   - Position 0 contains the value from array index 0
///   - Position 1 contains null
///   - Position 2 contains the value from array index 2
///   - The result array has nullable type
///
/// # Why This Matters
/// Nullable indices are a powerful feature that allows introducing nulls during
/// a take operation, which is useful for outer joins and similar operations.
pub fn test_nullable_indices_consistency(array: &dyn Array) {
    let len = array.len();
    if len < 3 {
        return; // Need at least 3 elements to test indices 0 and 2
    }

    // Create nullable indices where some indices are null
    let indices = PrimitiveArray::from_option_iter([Some(0u64), None, Some(2u64)]).into_array();

    let taken = take(array, &indices).vortex_unwrap();

    // Result should have nulls where indices were null
    assert_eq!(
        taken.len(),
        3,
        "Take with nullable indices should produce array of length 3, got {}",
        taken.len()
    );

    assert!(
        taken.dtype().is_nullable(),
        "Take with nullable indices should produce nullable array, but dtype is {:?}",
        taken.dtype()
    );

    // Check first element (from index 0)
    let expected_0 = array.scalar_at(0).vortex_unwrap().into_nullable();
    let actual_0 = taken.scalar_at(0).vortex_unwrap();
    assert_eq!(
        actual_0, expected_0,
        "Take with nullable indices: element at position 0 should be from array index 0. \
         Expected: {expected_0:?}, Actual: {actual_0:?}"
    );

    // Check second element (should be null)
    let actual_1 = taken.scalar_at(1).vortex_unwrap();
    assert!(
        actual_1.is_null(),
        "Take with nullable indices: element at position 1 should be null, but got {actual_1:?}"
    );

    // Check third element (from index 2)
    let expected_2 = array.scalar_at(2).vortex_unwrap().into_nullable();
    let actual_2 = taken.scalar_at(2).vortex_unwrap();
    assert_eq!(
        actual_2, expected_2,
        "Take with nullable indices: element at position 2 should be from array index 2. \
         Expected: {expected_2:?}, Actual: {actual_2:?}"
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

/// Run all consistency tests on an array.
///
/// This function executes a comprehensive suite of consistency tests that verify
/// the correctness of compute operations on Vortex arrays.
///
/// # Test Suite Overview
///
/// ## Core Operation Consistency
/// - **Filter/Take**: Verifies `filter(array, mask)` equals `take(array, true_indices)`
/// - **Mask Composition**: Ensures sequential masks equal combined masks
/// - **Slice/Filter**: Checks contiguous filters equal slice operations
/// - **Take/Slice**: Validates sequential takes equal slice operations
///
/// ## Identity Operations
/// - **Filter Identity**: All-true mask preserves the array
/// - **Mask Identity**: All-false mask preserves values (as nullable)
/// - **Take Identity**: Taking all indices preserves the array
///
/// ## Edge Cases
/// - **Empty Operations**: Empty filters, takes, and slices behave correctly
/// - **Single Element**: Operations work with single-element arrays
/// - **Repeated Indices**: Take with duplicate indices works correctly
///
/// ## Null Handling
/// - **Nullable Indices**: Null indices produce null values
/// - **Mask/Filter Interaction**: Masking then filtering behaves predictably
///
/// ## Large Arrays
/// - **Performance**: Operations scale correctly to large arrays (1000+ elements)
/// ```
pub fn test_array_consistency(array: &dyn Array) {
    // Core operation consistency
    test_filter_take_consistency(array);
    test_double_mask_consistency(array);
    test_slice_filter_consistency(array);
    test_take_slice_consistency(array);

    // Identity operations
    test_filter_identity(array);
    test_mask_identity(array);
    test_take_preserves_properties(array);

    // Ordering and correctness
    test_filter_preserves_order(array);
    test_take_repeated_indices(array);

    // Null handling
    test_mask_filter_null_consistency(array);
    test_nullable_indices_consistency(array);

    // Edge cases
    test_empty_operations_consistency(array);
    test_large_array_consistency(array);
}
