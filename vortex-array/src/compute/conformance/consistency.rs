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

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar_fn::fns::operators::Operator;

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
fn test_filter_take_consistency(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Create a test mask (keep elements where index % 3 != 1)
    let mask_pattern: BitBuffer = (0..len).map(|i| i % 3 != 1).collect();
    let mask = Mask::from_buffer(mask_pattern.clone());

    // Filter the array
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");

    // Create indices where mask is true
    let indices: Vec<u64> = mask_pattern
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.then_some(i as u64))
        .collect();
    let indices_array = PrimitiveArray::from_iter(indices).into_array();

    // Take using those indices
    let taken = array
        .take(indices_array)
        .vortex_expect("take should succeed in conformance test");

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
        let filtered_val = filtered
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
        let taken_val = taken
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
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
fn test_double_mask_consistency(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Create two different mask patterns
    let mask1: Mask = (0..len).map(|i| i % 3 == 0).collect();
    let mask2: Mask = (0..len).map(|i| i % 2 == 0).collect();

    // Apply masks sequentially
    let first_masked = array
        .clone()
        .mask((!&mask1).into_array())
        .vortex_expect("mask should succeed in conformance test");
    let double_masked = first_masked
        .mask((!&mask2).into_array())
        .vortex_expect("mask should succeed in conformance test");

    // Create combined mask (OR operation - element is masked if EITHER mask is true)
    let combined_pattern: BitBuffer = mask1
        .to_bit_buffer()
        .iter()
        .zip(mask2.to_bit_buffer().iter())
        .map(|(a, b)| a || b)
        .collect();
    let combined_mask = Mask::from_buffer(combined_pattern);

    // Apply combined mask directly
    let directly_masked = array
        .clone()
        .mask((!&combined_mask).into_array())
        .vortex_expect("mask should succeed in conformance test");

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
        let double_val = double_masked
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
        let direct_val = directly_masked
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
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
fn test_filter_identity(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    let all_true_mask = Mask::new_true(len);
    let filtered = array
        .filter(all_true_mask)
        .vortex_expect("filter should succeed in conformance test");

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
        let original_val = array
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
        let filtered_val = filtered
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
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
fn test_mask_identity(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    let all_false_mask = Mask::new_false(len);
    let masked = array
        .clone()
        .mask((!&all_false_mask).into_array())
        .vortex_expect("mask should succeed in conformance test");

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
        "Mask operation should always produce a nullable array, but dtype is {}",
        masked.dtype()
    );

    for i in 0..len {
        let original_val = array
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
        let masked_val = masked
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
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
fn test_slice_filter_consistency(array: &ArrayRef) {
    let len = array.len();
    if len < 4 {
        return; // Need at least 4 elements for meaningful test
    }

    // Create a contiguous mask (true from index 1 to 3)
    let mut mask_pattern = vec![false; len];
    mask_pattern[1..4.min(len)].fill(true);

    let mask = Mask::from_iter(mask_pattern);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");

    // Slice should produce the same result
    let sliced = array
        .slice(1..4.min(len))
        .vortex_expect("slice should succeed in conformance test");

    assert_eq!(
        filtered.len(),
        sliced.len(),
        "Filter with contiguous mask and slice should produce same length. \
         Filtered length: {}, Sliced length: {}",
        filtered.len(),
        sliced.len()
    );

    for i in 0..filtered.len() {
        let filtered_val = filtered
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
        let sliced_val = sliced
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
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
fn test_take_slice_consistency(array: &ArrayRef) {
    let len = array.len();
    if len < 3 {
        return; // Need at least 3 elements
    }

    // Take indices [1, 2, 3]
    let end = 4.min(len);
    let indices = PrimitiveArray::from_iter((1..end).map(|i| i as u64)).into_array();
    let taken = array
        .take(indices)
        .vortex_expect("take should succeed in conformance test");

    // Slice from 1 to end
    let sliced = array
        .slice(1..end)
        .vortex_expect("slice should succeed in conformance test");

    assert_eq!(
        taken.len(),
        sliced.len(),
        "Take with sequential indices and slice should produce same length. \
         Taken length: {}, Sliced length: {}",
        taken.len(),
        sliced.len()
    );

    for i in 0..taken.len() {
        let taken_val = taken
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
        let sliced_val = sliced
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test");
        assert_eq!(
            taken_val, sliced_val,
            "Take with sequential indices and slice produced different values at index {i}. \
             Taken value: {taken_val:?}, Sliced value: {sliced_val:?}"
        );
    }
}

/// Tests that filter preserves relative ordering
fn test_filter_preserves_order(array: &ArrayRef) {
    let len = array.len();
    if len < 4 {
        return;
    }

    // Create a mask that selects elements at indices 0, 2, 3
    let mask_pattern: Vec<bool> = (0..len).map(|i| i == 0 || i == 2 || i == 3).collect();
    let mask = Mask::from_iter(mask_pattern);

    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");

    // Verify the filtered array contains the right elements in order
    assert_eq!(filtered.len(), 3.min(len));
    if len >= 4 {
        assert_eq!(
            filtered
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
        assert_eq!(
            filtered
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
        assert_eq!(
            filtered
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

/// Tests that take with repeated indices works correctly
fn test_take_repeated_indices(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Take the first element three times
    let indices = PrimitiveArray::from_iter([0u64, 0, 0]).into_array();
    let taken = array
        .take(indices)
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(taken.len(), 3);
    for i in 0..3 {
        assert_eq!(
            taken
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

/// Tests mask and filter interaction with nulls
fn test_mask_filter_null_consistency(array: &ArrayRef) {
    let len = array.len();
    if len < 3 {
        return;
    }

    // First mask some elements
    let mask_pattern: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let mask_array = Mask::from_iter(mask_pattern);
    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");

    // Then filter to remove the nulls
    let filter_pattern: Vec<bool> = (0..len).map(|i| i % 2 != 0).collect();
    let filter_mask = Mask::from_iter(filter_pattern);
    let filtered = masked
        .filter(filter_mask.clone())
        .vortex_expect("filter should succeed in conformance test");

    // This should be equivalent to directly filtering the original array
    let direct_filtered = array
        .filter(filter_mask)
        .vortex_expect("filter should succeed in conformance test");

    assert_eq!(filtered.len(), direct_filtered.len());
    for i in 0..filtered.len() {
        assert_eq!(
            filtered
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            direct_filtered
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

/// Tests that empty operations are consistent
fn test_empty_operations_consistency(array: &ArrayRef) {
    let len = array.len();

    // Empty filter
    let empty_filter = array
        .filter(Mask::new_false(len))
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(empty_filter.len(), 0);
    assert_eq!(empty_filter.dtype(), array.dtype());

    // Empty take
    let empty_indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array();
    let empty_take = array
        .take(empty_indices)
        .vortex_expect("take should succeed in conformance test");
    assert_eq!(empty_take.len(), 0);
    assert_eq!(empty_take.dtype(), array.dtype());

    // Empty slice (if array is non-empty)
    if len > 0 {
        let empty_slice = array
            .slice(0..0)
            .vortex_expect("slice should succeed in conformance test");
        assert_eq!(empty_slice.len(), 0);
        assert_eq!(empty_slice.dtype(), array.dtype());
    }
}

/// Tests that take preserves array properties
fn test_take_preserves_properties(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Take all elements in original order
    let indices = PrimitiveArray::from_iter((0..len).map(|i| i as u64)).into_array();
    let taken = array
        .take(indices)
        .vortex_expect("take should succeed in conformance test");

    // Should be identical to original
    assert_eq!(taken.len(), array.len());
    assert_eq!(taken.dtype(), array.dtype());
    for i in 0..len {
        assert_eq!(
            taken
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
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
fn test_nullable_indices_consistency(array: &ArrayRef) {
    let len = array.len();
    if len < 3 {
        return; // Need at least 3 elements to test indices 0 and 2
    }

    // Create nullable indices where some indices are null
    let indices = PrimitiveArray::from_option_iter([Some(0u64), None, Some(2u64)]).into_array();

    let taken = array
        .take(indices)
        .vortex_expect("take should succeed in conformance test");

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
    let expected_0 = array
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .vortex_expect("scalar_at should succeed in conformance test")
        .into_nullable();
    let actual_0 = taken
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .vortex_expect("scalar_at should succeed in conformance test");
    assert_eq!(
        actual_0, expected_0,
        "Take with nullable indices: element at position 0 should be from array index 0. \
         Expected: {expected_0:?}, Actual: {actual_0:?}"
    );

    // Check second element (should be null)
    let actual_1 = taken
        .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
        .vortex_expect("scalar_at should succeed in conformance test");
    assert!(
        actual_1.is_null(),
        "Take with nullable indices: element at position 1 should be null, but got {actual_1:?}"
    );

    // Check third element (from index 2)
    let expected_2 = array
        .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
        .vortex_expect("scalar_at should succeed in conformance test")
        .into_nullable();
    let actual_2 = taken
        .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
        .vortex_expect("scalar_at should succeed in conformance test");
    assert_eq!(
        actual_2, expected_2,
        "Take with nullable indices: element at position 2 should be from array index 2. \
         Expected: {expected_2:?}, Actual: {actual_2:?}"
    );
}

/// Tests large array consistency
fn test_large_array_consistency(array: &ArrayRef) {
    let len = array.len();
    if len < 1000 {
        return;
    }

    // Test with every 10th element
    let indices: Vec<u64> = (0..len).step_by(10).map(|i| i as u64).collect();
    let indices_array = PrimitiveArray::from_iter(indices).into_array();
    let taken = array
        .take(indices_array)
        .vortex_expect("take should succeed in conformance test");

    // Create equivalent filter mask
    let mask_pattern: Vec<bool> = (0..len).map(|i| i % 10 == 0).collect();
    let mask = Mask::from_iter(mask_pattern);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");

    // Results should match
    assert_eq!(taken.len(), filtered.len());
    for i in 0..taken.len() {
        assert_eq!(
            taken
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            filtered
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

/// Tests that comparison operations follow inverse relationships.
///
/// # Invariants
/// - `compare(array, value, Eq)` is the inverse of `compare(array, value, NotEq)`
/// - `compare(array, value, Gt)` is the inverse of `compare(array, value, Lte)`
/// - `compare(array, value, Lt)` is the inverse of `compare(array, value, Gte)`
///
/// # Test Details
/// - Creates comparison results for each operator
/// - Verifies that inverse operations produce opposite boolean values
/// - Tests with multiple scalar values to ensure consistency
///
/// # Why This Matters
/// Comparison operations must maintain logical consistency across encodings.
/// This test catches bugs where an encoding might implement one comparison
/// correctly but fail on its logical inverse.
fn test_comparison_inverse_consistency(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Skip non-comparable types.
    match array.dtype() {
        DType::Null | DType::Extension(_) | DType::Struct(..) | DType::List(..) => return,
        _ => {}
    }

    // Get a test value from the middle of the array
    let test_scalar = if len == 0 {
        return;
    } else {
        array
            .execute_scalar(len / 2, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test")
    };

    // Test Eq vs NotEq
    let const_array = crate::arrays::ConstantArray::new(test_scalar, len);
    if let (Ok(eq_result), Ok(neq_result)) = (
        array
            .clone()
            .binary(const_array.clone().into_array(), Operator::Eq),
        array
            .clone()
            .binary(const_array.clone().into_array(), Operator::NotEq),
    ) {
        let inverted_eq = eq_result
            .not()
            .vortex_expect("not should succeed in conformance test");

        assert_eq!(
            inverted_eq.len(),
            neq_result.len(),
            "Inverted Eq should have same length as NotEq"
        );

        for i in 0..inverted_eq.len() {
            let inv_val = inverted_eq
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let neq_val = neq_result
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                inv_val, neq_val,
                "At index {i}: NOT(Eq) should equal NotEq. \
                 NOT(Eq) = {inv_val:?}, NotEq = {neq_val:?}"
            );
        }
    }

    // Test Gt vs Lte
    if let (Ok(gt_result), Ok(lte_result)) = (
        array
            .clone()
            .binary(const_array.clone().into_array(), Operator::Gt),
        array
            .clone()
            .binary(const_array.clone().into_array(), Operator::Lte),
    ) {
        let inverted_gt = gt_result
            .not()
            .vortex_expect("not should succeed in conformance test");

        for i in 0..inverted_gt.len() {
            let inv_val = inverted_gt
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let lte_val = lte_result
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                inv_val, lte_val,
                "At index {i}: NOT(Gt) should equal Lte. \
                 NOT(Gt) = {inv_val:?}, Lte = {lte_val:?}"
            );
        }
    }

    // Test Lt vs Gte
    if let (Ok(lt_result), Ok(gte_result)) = (
        array
            .clone()
            .binary(const_array.clone().into_array(), Operator::Lt),
        array
            .clone()
            .binary(const_array.into_array(), Operator::Gte),
    ) {
        let inverted_lt = lt_result
            .not()
            .vortex_expect("not should succeed in conformance test");

        for i in 0..inverted_lt.len() {
            let inv_val = inverted_lt
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let gte_val = gte_result
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                inv_val, gte_val,
                "At index {i}: NOT(Lt) should equal Gte. \
                 NOT(Lt) = {inv_val:?}, Gte = {gte_val:?}"
            );
        }
    }
}

/// Tests that comparison operations maintain proper symmetry relationships.
///
/// # Invariants
/// - `compare(array, value, Gt)` should equal `compare_scalar_array(value, array, Lt)`
/// - `compare(array, value, Lt)` should equal `compare_scalar_array(value, array, Gt)`
/// - `compare(array, value, Eq)` should equal `compare_scalar_array(value, array, Eq)`
///
/// # Test Details
/// - Compares array-scalar operations with their symmetric scalar-array versions
/// - Verifies that ordering relationships are properly reversed
/// - Tests equality which should be symmetric
///
/// # Why This Matters
/// Ensures that comparison operations maintain mathematical ordering properties
/// regardless of operand order.
fn test_comparison_symmetry_consistency(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Skip non-comparable types.
    match array.dtype() {
        DType::Null | DType::Extension(_) | DType::Struct(..) | DType::List(..) => return,
        _ => {}
    }

    // Get test values
    let test_scalar = if len == 2 {
        return;
    } else {
        array
            .execute_scalar(len / 2, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test")
    };

    // Create a constant array with the test scalar for reverse comparison
    let const_array = crate::arrays::ConstantArray::new(test_scalar, len);

    // Test Gt vs Lt symmetry
    if let (Ok(arr_gt_scalar), Ok(scalar_lt_arr)) = (
        array
            .clone()
            .binary(const_array.clone().into_array(), Operator::Gt),
        const_array
            .clone()
            .into_array()
            .binary(array.clone(), Operator::Lt),
    ) {
        assert_eq!(
            arr_gt_scalar.len(),
            scalar_lt_arr.len(),
            "Symmetric comparisons should have same length"
        );

        for i in 0..arr_gt_scalar.len() {
            let arr_gt = arr_gt_scalar
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let scalar_lt = scalar_lt_arr
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                arr_gt, scalar_lt,
                "At index {i}: (array > scalar) should equal (scalar < array). \
                 array > scalar = {arr_gt:?}, scalar < array = {scalar_lt:?}"
            );
        }
    }

    // Test Eq symmetry
    if let (Ok(arr_eq_scalar), Ok(scalar_eq_arr)) = (
        array
            .clone()
            .binary(const_array.clone().into_array(), Operator::Eq),
        const_array.into_array().binary(array.clone(), Operator::Eq),
    ) {
        for i in 0..arr_eq_scalar.len() {
            let arr_eq = arr_eq_scalar
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let scalar_eq = scalar_eq_arr
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                arr_eq, scalar_eq,
                "At index {i}: (array == scalar) should equal (scalar == array). \
                 array == scalar = {arr_eq:?}, scalar == array = {scalar_eq:?}"
            );
        }
    }
}

/// Tests that boolean operations follow De Morgan's laws.
///
/// # Invariants
/// - `NOT(A AND B)` equals `(NOT A) OR (NOT B)`
/// - `NOT(A OR B)` equals `(NOT A) AND (NOT B)`
///
/// # Test Details
/// - If the array is boolean, uses it directly for testing boolean operations
/// - Creates two boolean masks from patterns based on the array
/// - Computes AND/OR operations and their inversions
/// - Verifies De Morgan's laws hold for all elements
///
/// # Why This Matters
/// Boolean operations must maintain logical consistency across encodings.
/// This test catches bugs where encodings might optimize boolean operations
/// incorrectly, breaking fundamental logical properties.
fn test_boolean_demorgan_consistency(array: &ArrayRef) {
    if !matches!(array.dtype(), DType::Bool(_)) {
        return;
    }

    let bool_mask = {
        let mask_pattern: Vec<bool> = (0..array.len()).map(|i| i % 3 == 0).collect();
        BoolArray::from_iter(mask_pattern)
    };
    let bool_mask = bool_mask.into_array();

    // Test first De Morgan's law: NOT(A AND B) = (NOT A) OR (NOT B)
    if let (Ok(a_and_b), Ok(not_a), Ok(not_b)) = (
        array.clone().binary(bool_mask.clone(), Operator::And),
        array.not(),
        bool_mask.not(),
    ) {
        let not_a_and_b = a_and_b
            .not()
            .vortex_expect("not should succeed in conformance test");
        let not_a_or_not_b = not_a
            .binary(not_b, Operator::Or)
            .vortex_expect("or should succeed in conformance test");

        assert_eq!(
            not_a_and_b.len(),
            not_a_or_not_b.len(),
            "De Morgan's law results should have same length"
        );

        for i in 0..not_a_and_b.len() {
            let left = not_a_and_b
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let right = not_a_or_not_b
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                left, right,
                "De Morgan's first law failed at index {i}: \
                 NOT(A AND B) = {left:?}, (NOT A) OR (NOT B) = {right:?}"
            );
        }
    }

    // Test second De Morgan's law: NOT(A OR B) = (NOT A) AND (NOT B)
    if let (Ok(a_or_b), Ok(not_a), Ok(not_b)) = (
        array.clone().binary(bool_mask.clone(), Operator::Or),
        array.not(),
        bool_mask.not(),
    ) {
        let not_a_or_b = a_or_b
            .not()
            .vortex_expect("not should succeed in conformance test");
        let not_a_and_not_b = not_a
            .binary(not_b, Operator::And)
            .vortex_expect("and should succeed in conformance test");

        for i in 0..not_a_or_b.len() {
            let left = not_a_or_b
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let right = not_a_and_not_b
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                left, right,
                "De Morgan's second law failed at index {i}: \
                 NOT(A OR B) = {left:?}, (NOT A) AND (NOT B) = {right:?}"
            );
        }
    }
}

/// Tests that slice and aggregate operations produce consistent results.
///
/// # Invariants
/// - Aggregating a sliced array should equal aggregating the corresponding
///   elements from the canonical form
/// - This applies to sum, count, min/max, and other aggregate functions
///
/// # Test Details
/// - Slices the array and computes aggregates
/// - Compares against aggregating the canonical form's slice
/// - Tests multiple aggregate functions where applicable
///
/// # Why This Matters
/// Aggregate operations on sliced arrays must produce correct results
/// regardless of the underlying encoding's offset handling.
fn test_slice_aggregate_consistency(array: &ArrayRef) {
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::aggregate_fn::fns::nan_count::nan_count;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::dtype::DType;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    let len = array.len();
    if len < 5 {
        return; // Need enough elements for meaningful slice
    }

    // Define slice bounds
    let start = 1;
    let end = (len - 1).min(start + 10); // Take up to 10 elements

    // Get sliced array and canonical slice
    let sliced = array
        .slice(start..end)
        .vortex_expect("slice should succeed in conformance test");
    #[expect(deprecated)]
    let canonical = array.to_canonical().vortex_expect("to_canonical failed");
    let canonical_sliced = canonical
        .into_array()
        .slice(start..end)
        .vortex_expect("slice should succeed in conformance test");

    // Test null count through invalid_count
    let sliced_invalid_count = sliced
        .invalid_count(&mut ctx)
        .vortex_expect("invalid_count should succeed in conformance test");
    let canonical_invalid_count = canonical_sliced
        .invalid_count(&mut ctx)
        .vortex_expect("invalid_count should succeed in conformance test");
    assert_eq!(
        sliced_invalid_count, canonical_invalid_count,
        "null_count on sliced array should match canonical. \
             Sliced: {sliced_invalid_count}, Canonical: {canonical_invalid_count}",
    );

    // Test sum for numeric types
    if !matches!(array.dtype(), DType::Primitive(..)) {
        return;
    }

    if let (Ok(slice_sum), Ok(canonical_sum)) =
        (sum(&sliced, &mut ctx), sum(&canonical_sliced, &mut ctx))
    {
        // Compare sum scalars
        assert_eq!(
            slice_sum, canonical_sum,
            "sum on sliced array should match canonical. \
                 Sliced: {slice_sum:?}, Canonical: {canonical_sum:?}"
        );
    }

    // Test min_max
    if let (Ok(slice_minmax), Ok(canonical_minmax)) = (
        min_max(&sliced, &mut ctx),
        min_max(&canonical_sliced, &mut ctx),
    ) {
        match (slice_minmax, canonical_minmax) {
            (Some(s_result), Some(c_result)) => {
                assert_eq!(
                    s_result.min, c_result.min,
                    "min on sliced array should match canonical. \
                         Sliced: {:?}, Canonical: {:?}",
                    s_result.min, c_result.min
                );
                assert_eq!(
                    s_result.max, c_result.max,
                    "max on sliced array should match canonical. \
                         Sliced: {:?}, Canonical: {:?}",
                    s_result.max, c_result.max
                );
            }
            (None, None) => {} // Both empty, OK
            _ => vortex_panic!("min_max results don't match"),
        }
    }

    // Test nan_count for floating point types
    if array.dtype().is_float()
        && let (Ok(slice_nan_count), Ok(canonical_nan_count)) = (
            nan_count(&sliced, &mut ctx),
            nan_count(&canonical_sliced, &mut ctx),
        )
    {
        assert_eq!(
            slice_nan_count, canonical_nan_count,
            "nan_count on sliced array should match canonical. \
                 Sliced: {slice_nan_count}, Canonical: {canonical_nan_count}"
        );
    }
}

/// Tests that cast operations preserve array properties when sliced.
///
/// # Invariant
/// `cast(slice(array, start, end), dtype)` should equal `slice(cast(array, dtype), start, end)`
///
/// # Test Details
/// - Slices the array from index 2 to 7 (or len-2 if smaller)
/// - Casts the sliced array to a different type
/// - Compares against the canonical form of the array (without slicing or casting the canonical form)
/// - Verifies both approaches produce identical results
///
/// # Why This Matters
/// This test specifically catches bugs where encodings (like RunEndArray) fail to preserve
/// offset information during cast operations. Such bugs can lead to incorrect data being
/// returned after casting a sliced array.
fn test_cast_slice_consistency(array: &ArrayRef) {
    let len = array.len();
    if len < 5 {
        return; // Need at least 5 elements for meaningful slice
    }

    // Define slice bounds
    let start = 2;
    let end = 7.min(len - 2).max(start + 1); // Ensure we have at least 1 element

    // Get canonical form of the original array
    #[expect(deprecated)]
    let canonical = array.to_canonical().vortex_expect("to_canonical failed");

    // Choose appropriate target dtype based on the array's type
    let target_dtypes = match array.dtype() {
        DType::Null => vec![],
        DType::Bool(nullability) => vec![
            DType::Primitive(PType::U8, *nullability),
            DType::Primitive(PType::I32, *nullability),
        ],
        DType::Primitive(ptype, nullability) => {
            let mut targets = vec![];
            // Test nullability changes
            let opposite_nullability = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            targets.push(DType::Primitive(*ptype, opposite_nullability));

            // Test widening casts
            match ptype {
                PType::U8 => {
                    targets.push(DType::Primitive(PType::U16, *nullability));
                    targets.push(DType::Primitive(PType::I16, *nullability));
                }
                PType::U16 => {
                    targets.push(DType::Primitive(PType::U32, *nullability));
                    targets.push(DType::Primitive(PType::I32, *nullability));
                }
                PType::U32 => {
                    targets.push(DType::Primitive(PType::U64, *nullability));
                    targets.push(DType::Primitive(PType::I64, *nullability));
                }
                PType::U64 => {
                    targets.push(DType::Primitive(PType::F64, *nullability));
                }
                PType::I8 => {
                    targets.push(DType::Primitive(PType::I16, *nullability));
                    targets.push(DType::Primitive(PType::F32, *nullability));
                }
                PType::I16 => {
                    targets.push(DType::Primitive(PType::I32, *nullability));
                    targets.push(DType::Primitive(PType::F32, *nullability));
                }
                PType::I32 => {
                    targets.push(DType::Primitive(PType::I64, *nullability));
                    targets.push(DType::Primitive(PType::F64, *nullability));
                }
                PType::I64 => {
                    targets.push(DType::Primitive(PType::F64, *nullability));
                }
                PType::F16 => {
                    targets.push(DType::Primitive(PType::F32, *nullability));
                }
                PType::F32 => {
                    targets.push(DType::Primitive(PType::F64, *nullability));
                    targets.push(DType::Primitive(PType::I32, *nullability));
                }
                PType::F64 => {
                    targets.push(DType::Primitive(PType::I64, *nullability));
                }
            }
            targets
        }
        DType::Utf8(nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![DType::Utf8(opposite), DType::Binary(*nullability)]
        }
        DType::Binary(nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![
                DType::Binary(opposite),
                DType::Utf8(*nullability), // May fail if not valid UTF-8
            ]
        }
        DType::Decimal(decimal_type, nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![DType::Decimal(*decimal_type, opposite)]
        }
        DType::Struct(fields, nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![DType::Struct(fields.clone(), opposite)]
        }
        DType::List(element_type, nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![DType::List(Arc::clone(element_type), opposite)]
        }
        DType::FixedSizeList(element_type, list_size, nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![DType::FixedSizeList(
                Arc::clone(element_type),
                *list_size,
                opposite,
            )]
        }
        DType::Extension(_) => vec![], // Extension types typically only cast to themselves
        DType::Variant(_) => unimplemented!(),
    };

    // Test each target dtype
    for target_dtype in target_dtypes {
        // Slice the array
        let sliced = array
            .slice(start..end)
            .vortex_expect("slice should succeed in conformance test");

        // Try to cast the sliced array (force execution via to_canonical)
        let slice_then_cast = match sliced.cast(target_dtype.clone()).and_then(|a| {
            #[expect(deprecated)]
            a.to_canonical().map(|c| c.into_array())
        }) {
            Ok(result) => result,
            Err(_) => continue, // Skip if cast fails
        };

        // Verify against canonical form
        assert_eq!(
            slice_then_cast.len(),
            end - start,
            "Sliced and casted array should have length {}, but has {}",
            end - start,
            slice_then_cast.len()
        );

        // Compare each value against the canonical form
        for i in 0..slice_then_cast.len() {
            let slice_cast_val = slice_then_cast
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");

            // Get the corresponding value from the canonical array (adjusted for slice offset)
            let canonical_val = canonical
                .clone()
                .into_array()
                .execute_scalar(start + i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");

            // Cast the canonical scalar to the target dtype
            let expected_val = match canonical_val.cast(&target_dtype) {
                Ok(val) => val,
                Err(_) => {
                    // If scalar cast fails, we can't compare - skip this target dtype
                    // This can happen for some type conversions that aren't supported at scalar level
                    break;
                }
            };

            assert_eq!(
                slice_cast_val,
                expected_val,
                "Cast of sliced array produced incorrect value at index {i}. \
                 Got: {slice_cast_val:?}, Expected: {expected_val:?} \
                 (canonical value at index {}: {canonical_val:?})\n\
                 This likely indicates the array encoding doesn't preserve offset information during cast.",
                start + i
            );
        }

        // Also test the other way: cast then slice
        let casted = match array.clone().cast(target_dtype.clone()).and_then(|a| {
            #[expect(deprecated)]
            a.to_canonical().map(|c| c.into_array())
        }) {
            Ok(result) => result,
            Err(_) => continue, // Skip if cast fails
        };
        let cast_then_slice = casted
            .slice(start..end)
            .vortex_expect("slice should succeed in conformance test");

        // Verify the two approaches produce identical results
        assert_eq!(
            slice_then_cast.len(),
            cast_then_slice.len(),
            "Slice-then-cast and cast-then-slice should produce arrays of the same length"
        );

        for i in 0..slice_then_cast.len() {
            let slice_cast_val = slice_then_cast
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            let cast_slice_val = cast_then_slice
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test");
            assert_eq!(
                slice_cast_val, cast_slice_val,
                "Slice-then-cast and cast-then-slice produced different values at index {i}. \
                 Slice-then-cast: {slice_cast_val:?}, Cast-then-slice: {cast_slice_val:?}"
            );
        }
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
/// - **Cast/Slice**: Ensures cast operations preserve sliced array properties
///
/// ## Boolean Operations
/// - **De Morgan's Laws**: Verifies boolean operations follow logical laws
///
/// ## Comparison Operations
/// - **Inverse Relationships**: Verifies logical inverses (Eq/NotEq, Gt/Lte, Lt/Gte)
/// - **Symmetry**: Ensures proper ordering relationships when operands are swapped
///
/// ## Aggregate Operations
/// - **Slice/Aggregate**: Verifies aggregates on sliced arrays match canonical
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
/// ```text
pub fn test_array_consistency(array: &ArrayRef) {
    // Core operation consistency
    test_filter_take_consistency(array);
    test_double_mask_consistency(array);
    test_slice_filter_consistency(array);
    test_take_slice_consistency(array);
    test_cast_slice_consistency(array);

    // Boolean operations
    test_boolean_demorgan_consistency(array);

    // Comparison operations
    test_comparison_inverse_consistency(array);
    test_comparison_symmetry_consistency(array);

    // Aggregate operations
    test_slice_aggregate_consistency(array);

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
