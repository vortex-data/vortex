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
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::nan_count::nan_count;
use crate::aggregate_fn::fns::sum::sum;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
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
fn test_filter_take_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
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
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
        .vortex_expect("take should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
            .execute_scalar(i, ctx)
            .vortex_expect("scalar_at should succeed in conformance test");
        let taken_val = taken
            .execute_scalar(i, ctx)
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
fn test_double_mask_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
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
        .vortex_expect("mask should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
        .vortex_expect("mask should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
            .execute_scalar(i, ctx)
            .vortex_expect("scalar_at should succeed in conformance test");
        let direct_val = directly_masked
            .execute_scalar(i, ctx)
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
fn test_filter_identity(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len == 0 {
        return;
    }

    let all_true_mask = Mask::new_true(len);
    let filtered = array
        .filter(all_true_mask)
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
            .execute_scalar(i, ctx)
            .vortex_expect("scalar_at should succeed in conformance test");
        let filtered_val = filtered
            .execute_scalar(i, ctx)
            .vortex_expect("scalar_at should succeed in conformance test");
        assert_eq!(
            filtered_val, original_val,
            "Filtering with all-true mask should preserve all values. \
             Value at index {i} changed from {original_val:?} to {filtered_val:?}"
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
fn test_slice_filter_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
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
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

    // Slice should produce the same result
    let sliced = array
        .slice(1..4.min(len))
        .vortex_expect("slice should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
            .execute_scalar(i, ctx)
            .vortex_expect("scalar_at should succeed in conformance test");
        let sliced_val = sliced
            .execute_scalar(i, ctx)
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
fn test_take_slice_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len < 3 {
        return; // Need at least 3 elements
    }

    // Take indices [1, 2, 3]
    let end = 4.min(len);
    let indices = PrimitiveArray::from_iter((1..end).map(|i| i as u64)).into_array();
    let taken = array
        .take(indices)
        .vortex_expect("take should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

    // Slice from 1 to end
    let sliced = array
        .slice(1..end)
        .vortex_expect("slice should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
            .execute_scalar(i, ctx)
            .vortex_expect("scalar_at should succeed in conformance test");
        let sliced_val = sliced
            .execute_scalar(i, ctx)
            .vortex_expect("scalar_at should succeed in conformance test");
        assert_eq!(
            taken_val, sliced_val,
            "Take with sequential indices and slice produced different values at index {i}. \
             Taken value: {taken_val:?}, Sliced value: {sliced_val:?}"
        );
    }
}

/// Tests that filter preserves relative ordering
fn test_filter_preserves_order(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len < 4 {
        return;
    }

    // Create a mask that selects elements at indices 0, 2, 3
    let mask_pattern: Vec<bool> = (0..len).map(|i| i == 0 || i == 2 || i == 3).collect();
    let mask = Mask::from_iter(mask_pattern);

    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

    // Verify the filtered array contains the right elements in order
    assert_eq!(filtered.len(), 3.min(len));
    if len >= 4 {
        assert_eq!(
            filtered
                .execute_scalar(0, ctx)
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(0, ctx)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
        assert_eq!(
            filtered
                .execute_scalar(1, ctx)
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(2, ctx)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
        assert_eq!(
            filtered
                .execute_scalar(2, ctx)
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(3, ctx)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

/// Tests mask and filter interaction with nulls
fn test_mask_filter_null_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
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
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

    // This should be equivalent to directly filtering the original array
    let direct_filtered = array
        .filter(filter_mask)
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

    assert_eq!(filtered.len(), direct_filtered.len());
    for i in 0..filtered.len() {
        assert_eq!(
            filtered
                .execute_scalar(i, ctx)
                .vortex_expect("scalar_at should succeed in conformance test"),
            direct_filtered
                .execute_scalar(i, ctx)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

/// Tests that empty operations are consistent
fn test_empty_operations_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();

    // Empty filter
    let empty_filter = array
        .filter(Mask::new_false(len))
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();
    assert_eq!(empty_filter.len(), 0);
    assert_eq!(empty_filter.dtype(), array.dtype());

    // Empty take
    let empty_indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array();
    let empty_take = array
        .take(empty_indices)
        .vortex_expect("take should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();
    assert_eq!(empty_take.len(), 0);
    assert_eq!(empty_take.dtype(), array.dtype());

    // Empty slice (if array is non-empty)
    if len > 0 {
        let empty_slice = array
            .slice(0..0)
            .vortex_expect("slice should succeed in conformance test")
            .execute::<RecursiveCanonical>(ctx)
            .vortex_expect("batch execution should succeed in conformance test")
            .0
            .into_array();
        assert_eq!(empty_slice.len(), 0);
        assert_eq!(empty_slice.dtype(), array.dtype());
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
fn test_nullable_indices_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len < 3 {
        return; // Need at least 3 elements to test indices 0 and 2
    }

    // Create nullable indices where some indices are null
    let indices = PrimitiveArray::from_option_iter([Some(0u64), None, Some(2u64)]).into_array();

    let taken = array
        .take(indices)
        .vortex_expect("take should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

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
        .execute_scalar(0, ctx)
        .vortex_expect("scalar_at should succeed in conformance test")
        .into_nullable();
    let actual_0 = taken
        .execute_scalar(0, ctx)
        .vortex_expect("scalar_at should succeed in conformance test");
    assert_eq!(
        actual_0, expected_0,
        "Take with nullable indices: element at position 0 should be from array index 0. \
         Expected: {expected_0:?}, Actual: {actual_0:?}"
    );

    // Check second element (should be null)
    let actual_1 = taken
        .execute_scalar(1, ctx)
        .vortex_expect("scalar_at should succeed in conformance test");
    assert!(
        actual_1.is_null(),
        "Take with nullable indices: element at position 1 should be null, but got {actual_1:?}"
    );

    // Check third element (from index 2)
    let expected_2 = array
        .execute_scalar(2, ctx)
        .vortex_expect("scalar_at should succeed in conformance test")
        .into_nullable();
    let actual_2 = taken
        .execute_scalar(2, ctx)
        .vortex_expect("scalar_at should succeed in conformance test");
    assert_eq!(
        actual_2, expected_2,
        "Take with nullable indices: element at position 2 should be from array index 2. \
         Expected: {expected_2:?}, Actual: {actual_2:?}"
    );
}

/// Tests large array consistency
fn test_large_array_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len < 1000 {
        return;
    }

    // Test with every 10th element
    let indices: Vec<u64> = (0..len).step_by(10).map(|i| i as u64).collect();
    let indices_array = PrimitiveArray::from_iter(indices).into_array();
    let taken = array
        .take(indices_array)
        .vortex_expect("take should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

    // Create equivalent filter mask
    let mask_pattern: Vec<bool> = (0..len).map(|i| i % 10 == 0).collect();
    let mask = Mask::from_iter(mask_pattern);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("batch execution should succeed in conformance test")
        .0
        .into_array();

    // Results should match
    assert_eq!(taken.len(), filtered.len());
    for i in 0..taken.len() {
        assert_eq!(
            taken
                .execute_scalar(i, ctx)
                .vortex_expect("scalar_at should succeed in conformance test"),
            filtered
                .execute_scalar(i, ctx)
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
fn test_comparison_inverse_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Skip non-comparable types.
    match array.dtype() {
        DType::Null
        | DType::List(..)
        | DType::Struct(..)
        | DType::Union(..)
        | DType::Extension(_) => return,
        _ => {}
    }

    let test_scalar = array
        .execute_scalar(len / 2, ctx)
        .vortex_expect("scalar_at should succeed in conformance test");
    let const_array = ConstantArray::new(test_scalar, len).into_array();

    for (operator, counterpart) in [
        (Operator::Eq, Operator::NotEq),
        (Operator::Gt, Operator::Lte),
        (Operator::Lt, Operator::Gte),
    ] {
        let left = array
            .binary(const_array.clone(), operator)
            .vortex_expect("comparison should succeed in conformance test");
        let left = left
            .not()
            .vortex_expect("not should succeed in conformance test");
        let right = array
            .binary(const_array.clone(), counterpart)
            .vortex_expect("comparison should succeed in conformance test");
        assert_arrays_eq!(left, right, ctx);
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
fn test_comparison_symmetry_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Skip non-comparable types.
    match array.dtype() {
        DType::Null
        | DType::List(..)
        | DType::Struct(..)
        | DType::Union(..)
        | DType::Extension(_) => return,
        _ => {}
    }

    let test_scalar = array
        .execute_scalar(len / 2, ctx)
        .vortex_expect("scalar_at should succeed in conformance test");
    let const_array = ConstantArray::new(test_scalar, len).into_array();

    for (operator, counterpart) in [(Operator::Gt, Operator::Lt), (Operator::Eq, Operator::Eq)] {
        let left = array
            .binary(const_array.clone(), operator)
            .vortex_expect("comparison should succeed in conformance test");
        let right = const_array
            .binary(array.clone(), counterpart)
            .vortex_expect("comparison should succeed in conformance test");
        assert_arrays_eq!(left, right, ctx);
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
fn test_boolean_demorgan_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    if !matches!(array.dtype(), DType::Bool(_)) {
        return;
    }

    let bool_mask = BoolArray::from_iter((0..array.len()).map(|i| i % 3 == 0)).into_array();
    for (operator, counterpart) in [(Operator::And, Operator::Or), (Operator::Or, Operator::And)] {
        let left = array
            .binary(bool_mask.clone(), operator)
            .vortex_expect("boolean operation should succeed in conformance test")
            .not()
            .vortex_expect("not should succeed in conformance test");
        let not_a = array
            .not()
            .vortex_expect("not should succeed in conformance test");
        let not_b = bool_mask
            .not()
            .vortex_expect("not should succeed in conformance test");
        let right = not_a
            .binary(not_b, counterpart)
            .vortex_expect("boolean operation should succeed in conformance test");
        assert_arrays_eq!(left, right, ctx);
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
fn test_slice_aggregate_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
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
    let canonical = array
        .clone()
        .execute::<Canonical>(ctx)
        .vortex_expect("to_canonical failed");
    let canonical_sliced = canonical
        .into_array()
        .slice(start..end)
        .vortex_expect("slice should succeed in conformance test");

    // Test null count through invalid_count
    let sliced_invalid_count = sliced
        .invalid_count(ctx)
        .vortex_expect("invalid_count should succeed in conformance test");
    let canonical_invalid_count = canonical_sliced
        .invalid_count(ctx)
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

    let slice_sum = sum(&sliced, ctx).vortex_expect("slice sum should succeed");
    let canonical_sum = sum(&canonical_sliced, ctx).vortex_expect("canonical sum should succeed");
    assert_eq!(
        slice_sum, canonical_sum,
        "sum on sliced array should match canonical"
    );

    let slice_minmax = min_max(&sliced, ctx, NumericalAggregateOpts::default())
        .vortex_expect("slice min_max should succeed");
    let canonical_minmax = min_max(&canonical_sliced, ctx, NumericalAggregateOpts::default())
        .vortex_expect("canonical min_max should succeed");
    assert_eq!(
        slice_minmax, canonical_minmax,
        "min_max on sliced array should match canonical"
    );

    if array.dtype().is_float() {
        let slice_nan_count =
            nan_count(&sliced, ctx).vortex_expect("slice nan_count should succeed");
        let canonical_nan_count =
            nan_count(&canonical_sliced, ctx).vortex_expect("canonical nan_count should succeed");
        assert_eq!(
            slice_nan_count, canonical_nan_count,
            "nan_count on sliced array should match canonical"
        );
    }
}

fn widened_primitive_dtype(dtype: &DType) -> Option<DType> {
    let DType::Primitive(ptype, nullability) = dtype else {
        return None;
    };
    let widened = match ptype {
        PType::U8 => PType::U16,
        PType::U16 => PType::U32,
        PType::U32 => PType::U64,
        PType::U64 => return None,
        PType::I8 => PType::I16,
        PType::I16 => PType::I32,
        PType::I32 => PType::I64,
        PType::I64 => return None,
        PType::F16 => PType::F32,
        PType::F32 => PType::F64,
        PType::F64 => return None,
    };
    Some(DType::Primitive(widened, *nullability))
}

/// Cast targets exercised for each logical dtype.
fn slice_cast_targets(dtype: &DType) -> Vec<DType> {
    match dtype {
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
        DType::Decimal(decimal_type, nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![DType::Decimal(*decimal_type, opposite)]
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
        DType::Map(map_dtype, nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            let key_dtype = map_dtype.key_dtype();
            let value_dtype = map_dtype.value_dtype();
            let mut targets = vec![DType::Map(map_dtype.clone(), opposite)];

            if let Some(widened_key) = widened_primitive_dtype(&key_dtype)
                && let Ok(dtype) = DType::map(
                    widened_key,
                    value_dtype.clone(),
                    map_dtype.keys_sorted(),
                    *nullability,
                )
            {
                targets.push(dtype);
            }

            if let Some(widened_value) = widened_primitive_dtype(&value_dtype)
                && let Ok(dtype) = DType::map(
                    key_dtype.clone(),
                    widened_value,
                    map_dtype.keys_sorted(),
                    *nullability,
                )
            {
                targets.push(dtype);
            }

            if map_dtype.keys_sorted()
                && let Ok(dtype) = DType::map(key_dtype, value_dtype, false, *nullability)
            {
                targets.push(dtype);
            }

            targets
        }
        DType::Struct(fields, nullability) => {
            let opposite = match nullability {
                Nullability::NonNullable => Nullability::Nullable,
                Nullability::Nullable => Nullability::NonNullable,
            };
            vec![DType::Struct(fields.clone(), opposite)]
        }
        DType::Union(..) => vortex_panic!("TODO(connor)[Union]: unimplemented"),
        DType::Variant(_) => {
            vortex_panic!("Variant conformance casting is not implemented")
        }
        DType::Extension(_) => vec![], // Extension types typically only cast to themselves
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
fn test_cast_slice_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    if len < 5 {
        return; // Need at least 5 elements for meaningful slice
    }

    // Define slice bounds
    let start = 2;
    let end = 7.min(len - 2).max(start + 1); // Ensure we have at least 1 element

    // Get canonical form of the original array
    let canonical = array
        .clone()
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("to_canonical failed")
        .0
        .into_array();

    let target_dtypes = slice_cast_targets(array.dtype());

    for target_dtype in target_dtypes {
        let mut slice_then_cast: Option<ArrayRef> = None;
        // Whole-array casts can fail on rows outside the selected slice. Establish
        // support from each canonical input independently before running its encoding.
        for slice_first in [true, false] {
            let reference_input = if slice_first {
                canonical
                    .slice(start..end)
                    .vortex_expect("canonical slice should succeed")
            } else {
                canonical.clone()
            };
            let expected = reference_input.cast(target_dtype.clone()).and_then(|a| {
                a.execute::<RecursiveCanonical>(ctx)
                    .map(|c| c.0.into_array())
            });
            let input = if slice_first {
                array
                    .slice(start..end)
                    .vortex_expect("slice should succeed")
            } else {
                array.clone()
            };
            let actual = input.cast(target_dtype.clone()).and_then(|a| {
                a.execute::<RecursiveCanonical>(ctx)
                    .map(|c| c.0.into_array())
            });
            let expected = match expected {
                Ok(expected) => expected,
                Err(_) => {
                    assert!(
                        actual.is_err(),
                        "encoded cast to {target_dtype} succeeded where canonical cast failed (slice_first={slice_first})"
                    );
                    continue;
                }
            };
            let actual =
                actual.vortex_expect("encoded cast should succeed when canonical cast succeeds");
            let (actual, expected) = if slice_first {
                (actual, expected)
            } else {
                (
                    actual
                        .slice(start..end)
                        .vortex_expect("cast result slice should succeed")
                        .execute::<RecursiveCanonical>(ctx)
                        .vortex_expect("slice batch execution should succeed")
                        .0
                        .into_array(),
                    expected
                        .slice(start..end)
                        .vortex_expect("canonical cast result slice should succeed")
                        .execute::<RecursiveCanonical>(ctx)
                        .vortex_expect("canonical slice batch execution should succeed")
                        .0
                        .into_array(),
                )
            };
            assert_eq!(actual.dtype(), &target_dtype);
            assert_eq!(actual.len(), end - start);
            assert_eq!(expected.dtype(), &target_dtype);
            assert_eq!(expected.len(), end - start);
            // AllNonDistinct does not support unions, including nested union children.
            // Both inputs have already executed their batch and child operations.
            for i in 0..actual.len() {
                assert_eq!(
                    actual
                        .execute_scalar(i, ctx)
                        .vortex_expect("cast scalar_at should succeed"),
                    expected
                        .execute_scalar(i, ctx)
                        .vortex_expect("reference scalar_at should succeed"),
                    "cast value differs from canonical reference at index {i}"
                );
                if let Some(sliced_cast) = &slice_then_cast {
                    assert_eq!(
                        actual
                            .execute_scalar(i, ctx)
                            .vortex_expect("cast scalar_at should succeed"),
                        sliced_cast
                            .execute_scalar(i, ctx)
                            .vortex_expect("slice cast scalar_at should succeed"),
                        "slice/cast order differs at index {i}"
                    );
                }
            }
            if slice_first {
                slice_then_cast = Some(actual.clone());
            }

            // Scalar casts do not implement every array conversion. When supported,
            // retain a complete independent per-value oracle rather than a partial prefix.
            let scalar_expected: Result<Vec<_>, _> = (start..end)
                .map(|i| {
                    canonical
                        .execute_scalar(i, ctx)
                        .vortex_expect("canonical scalar_at should succeed")
                        .cast(&target_dtype)
                })
                .collect();
            if let Ok(scalars) = scalar_expected {
                for (i, expected) in scalars.into_iter().enumerate() {
                    let actual = actual
                        .execute_scalar(i, ctx)
                        .vortex_expect("scalar_at should succeed");
                    assert_eq!(actual, expected, "cast value differs at index {i}");
                }
            }
        }
    }
}

/// Tests that a repeated probe agrees with one-off scalar reads.
///
/// # Invariant
/// `array.repeated_probe().execute_scalar(i)` equals `array.execute_scalar(i)` for every `i`,
/// whatever order the rows are visited in.
///
/// # Test Details
/// - Reads every row one-off to build the expected values
/// - Reads the same rows through a single retained probe, backwards, then forwards, then
///   sparsely, so a probe that caches per-row state has to stay correct across jumps and
///   revisits
/// - Checks nullness through the probe as well as the value
///
/// # Why This Matters
/// Encodings serve one-off and repeated reads from the same `probe_scalar` body, and a parent
/// encoding passes its children's probes down. A retained cache that outlives the row it was
/// built for, or a slot read through the wrong index, shows up here as a wrong value.
fn test_repeated_probe_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    // Long arrays are covered by their prefix; the three passes below already cost 3n reads.
    let len = array.len().min(1024);
    if len == 0 {
        return;
    }

    let expected: Vec<_> = (0..len)
        .map(|i| {
            array
                .execute_scalar(i, ctx)
                .vortex_expect("scalar_at should succeed in conformance test")
        })
        .collect();

    let mut probe = array.repeated_probe();
    let order = (0..len).rev().chain(0..len).chain((0..len).step_by(7));
    for i in order {
        let actual = probe
            .execute_scalar(i, ctx)
            .vortex_expect("repeated probe read should succeed in conformance test");
        assert_eq!(
            actual, expected[i],
            "Repeated probe and one-off read disagree at index {i}. \
             Probe value: {actual:?}, one-off value: {:?}",
            expected[i]
        );
        let valid = probe
            .execute_is_valid(i, ctx)
            .vortex_expect("repeated probe validity read should succeed in conformance test");
        assert_eq!(
            valid,
            !expected[i].is_null(),
            "Repeated probe validity disagrees with the value read at index {i}"
        );
    }
}

/// Check one-off and retained scalar probes, including nullness, revisits, and slices.
///
/// Use this for encodings that do not support the full compute conformance suite.
pub fn test_scalar_probe_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    test_repeated_probe_consistency(array, ctx);
    test_probe_consistency(array, ctx);
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
pub fn test_array_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    // Core operation consistency
    test_filter_take_consistency(array, ctx);
    test_double_mask_consistency(array, ctx);
    test_slice_filter_consistency(array, ctx);
    test_take_slice_consistency(array, ctx);
    test_cast_slice_consistency(array, ctx);

    // Boolean operations
    test_boolean_demorgan_consistency(array, ctx);

    // Comparison operations
    test_comparison_inverse_consistency(array, ctx);
    test_comparison_symmetry_consistency(array, ctx);

    // Aggregate operations
    test_slice_aggregate_consistency(array, ctx);

    // Identity operations
    test_filter_identity(array, ctx);
    if !array.is_empty() {
        super::mask::test_empty_mask(array, ctx);
        super::take::test_take_all(array, ctx);
    }

    // Row access
    test_repeated_probe_consistency(array, ctx);

    // Ordering and correctness
    test_filter_preserves_order(array, ctx);
    super::take::test_take_repeated_indices(array, ctx);

    // Null handling
    test_mask_filter_null_consistency(array, ctx);
    test_nullable_indices_consistency(array, ctx);

    // Edge cases
    test_empty_operations_consistency(array, ctx);
    test_large_array_consistency(array, ctx);

    // Row access
    test_probe_consistency(array, ctx);
}

/// Tests that one-off and repeated probes agree with each other and with validity.
///
/// # Invariant
/// For every index `i`, on the array and a slice of it:
/// - `probe().execute_scalar(i)` equals `repeated_probe().execute_scalar(i)`
/// - `execute_is_valid(i)` equals `!execute_scalar(i).is_null()` on both probes
fn test_probe_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    check_probe_consistency(array, ctx);
    if array.len() > 2 {
        let sliced = array
            .slice(1..array.len() - 1)
            .vortex_expect("slice should succeed in conformance test");
        check_probe_consistency(&sliced, ctx);
    }
}

fn check_probe_consistency(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    let stride = if len <= 1024 { 1 } else { 7 };
    let indices: Vec<usize> = (0..len).step_by(stride).collect();

    let mut once = array.probe();
    let mut repeated = array.repeated_probe();
    for &i in indices.iter().chain(indices.iter().rev()) {
        let expected = once
            .execute_scalar(i, ctx)
            .vortex_expect("one-off probe should succeed in conformance test");
        let valid = once
            .execute_is_valid(i, ctx)
            .vortex_expect("one-off validity should succeed in conformance test");
        assert_eq!(
            valid,
            !expected.is_null(),
            "One-off validity and scalar disagree at index {i}: valid {valid}, scalar {expected:?}"
        );

        let retained_valid = repeated
            .execute_is_valid(i, ctx)
            .vortex_expect("retained validity should succeed in conformance test");
        assert_eq!(
            retained_valid, valid,
            "Retained and one-off validity disagree at index {i}"
        );
        let actual = repeated
            .execute_scalar(i, ctx)
            .vortex_expect("retained probe should succeed in conformance test");
        assert_eq!(
            actual, expected,
            "Retained and one-off probes disagree at index {i}"
        );
    }
}
