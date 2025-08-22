// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use crate::arrays::BoolArray;
use crate::compute::filter;
use crate::{Array, IntoArray};

// Standard test array sizes
pub const SMALL_SIZE: usize = 5;
pub const MEDIUM_SIZE: usize = 100;
pub const LARGE_SIZE: usize = 1024;

/// Test filter compute function with various array sizes and patterns.
/// The input array can be of any length.
pub fn test_filter_conformance(array: &dyn Array) {
    let len = array.len();

    // Test with arrays of any size
    if len > 0 {
        test_all_filter(array);
        test_none_filter(array);
        test_selective_filter(array);
        test_single_element_filter(array);
        test_nullable_filter(array);
        test_alternating_pattern_filter(array);
        test_runs_pattern_filter(array);
        test_sparse_true_filter(array);
        test_sparse_false_filter(array);
    }

    // Test random pattern for arrays with at least 4 elements
    if len >= 4 {
        test_random_pattern_filter(array);
    }

    // Test edge cases
    test_empty_array_filter(array.dtype());
    test_mismatched_lengths(array);

    // Test with nullable masks
    if len > 0 {
        test_nullable_mask_input(array);
    }
}

// Helper functions for creating standard patterns
pub fn create_alternating_pattern(len: usize) -> Vec<bool> {
    (0..len).map(|i| i % 2 == 0).collect()
}

pub fn create_sparse_pattern(len: usize, true_ratio: f64) -> Vec<bool> {
    (0..len)
        .map(|i| (i as f64 / len as f64) < true_ratio)
        .collect()
}

pub fn create_runs_pattern(len: usize, run_length: usize) -> Vec<bool> {
    (0..len).map(|i| (i / run_length) % 2 == 0).collect()
}

/// Tests that filtering with an all-true mask returns all elements unchanged
fn test_all_filter(array: &dyn Array) {
    let len = array.len();
    let mask = Mask::new_true(len);
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), len);

    // Verify all elements are preserved
    for i in 0..len {
        assert_eq!(filtered.scalar_at(i), array.scalar_at(i));
    }
}

/// Tests that filtering with an all-false mask returns an empty array with the same dtype
fn test_none_filter(array: &dyn Array) {
    let len = array.len();
    let mask = Mask::new_false(len);
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 0);
    assert_eq!(filtered.dtype(), array.dtype());
}

fn test_selective_filter(array: &dyn Array) {
    let len = array.len();
    if len < 2 {
        return; // Skip for very small arrays
    }

    // Test alternating pattern
    let mask_values: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let expected_count = mask_values.iter().filter(|&&v| v).count();
    let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);

    // Verify correct elements are kept
    for (filtered_idx, i) in (0..len).step_by(2).enumerate() {
        assert_eq!(filtered.scalar_at(filtered_idx), array.scalar_at(i));
    }

    // Test first and last only
    if len >= 2 {
        let mut mask_values = vec![false; len];
        mask_values[0] = true;
        mask_values[len - 1] = true;
        let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
        let filtered = filter(array, &mask).vortex_unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.scalar_at(0), array.scalar_at(0));
        assert_eq!(filtered.scalar_at(1), array.scalar_at(len - 1));
    }
}

fn test_single_element_filter(array: &dyn Array) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Test selecting only the first element
    let mut mask_values = vec![false; len];
    mask_values[0] = true;
    let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered.scalar_at(0), array.scalar_at(0));

    // Test selecting only the last element
    if len > 1 {
        let mut mask_values = vec![false; len];
        mask_values[len - 1] = true;
        let mask = Mask::try_from(&BoolArray::from_iter(mask_values)).vortex_unwrap();
        let filtered = filter(array, &mask).vortex_unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.scalar_at(0), array.scalar_at(len - 1));
    }
}

fn test_nullable_filter(array: &dyn Array) {
    let len = array.len();
    if len < 2 {
        return; // Skip for very small arrays
    }

    // Create a nullable mask where nulls are treated as false
    let bool_values: Vec<bool> = (0..len).map(|i| i % 3 == 0).collect();
    let validity_values: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();

    let expected_count = bool_values
        .iter()
        .zip(validity_values.iter())
        .filter(|(b, v)| **b && **v)
        .count();

    let bool_array = BoolArray::from_iter(bool_values.clone());
    let validity = crate::validity::Validity::from_iter(validity_values.clone());
    let nullable_mask = BoolArray::new(bool_array.boolean_buffer().clone(), validity);

    let mask = Mask::try_from(&nullable_mask).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);

    // Verify correct elements are kept
    let mut filtered_idx = 0;
    for i in 0..len {
        if bool_values[i] && validity_values[i] {
            assert_eq!(filtered.scalar_at(filtered_idx), array.scalar_at(i));
            filtered_idx += 1;
        }
    }
}

fn test_empty_array_filter(dtype: &DType) {
    use crate::Canonical;

    let empty_array = Canonical::empty(dtype).into_array();
    let empty_mask = Mask::new_false(0);
    let filtered = filter(&empty_array, &empty_mask).vortex_unwrap();
    assert_eq!(filtered.len(), 0);

    let empty_mask = Mask::new_true(0);
    let filtered = filter(&empty_array, &empty_mask).vortex_unwrap();
    assert_eq!(filtered.len(), 0);
}

fn test_mismatched_lengths(array: &dyn Array) {
    let len = array.len();

    // Test mask shorter than array
    if len > 0 {
        let short_mask = Mask::new_true(len - 1);
        let result = filter(array, &short_mask);
        assert!(
            result.is_err(),
            "Filter should fail with mismatched lengths"
        );
    }

    // Test mask longer than array
    let long_mask = Mask::new_true(len + 1);
    let result = filter(array, &long_mask);
    assert!(
        result.is_err(),
        "Filter should fail with mismatched lengths"
    );
}

/// Tests filtering with alternating true/false pattern
fn test_alternating_pattern_filter(array: &dyn Array) {
    let len = array.len();
    let pattern = create_alternating_pattern(len);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::try_from(&BoolArray::from_iter(pattern.clone())).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);

    // Verify correct elements are kept
    let mut filtered_idx = 0;
    for (i, &keep) in pattern.iter().enumerate() {
        if keep {
            assert_eq!(filtered.scalar_at(filtered_idx), array.scalar_at(i));
            filtered_idx += 1;
        }
    }
}

/// Tests filtering with runs of true/false values
fn test_runs_pattern_filter(array: &dyn Array) {
    let len = array.len();
    if len < 4 {
        return; // Skip for very small arrays
    }

    let run_length = len.min(3);
    let pattern = create_runs_pattern(len, run_length);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::try_from(&BoolArray::from_iter(pattern)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);
}

/// Tests filtering with sparse true values (mostly false)
fn test_sparse_true_filter(array: &dyn Array) {
    let len = array.len();
    if len < 10 {
        return; // Skip for small arrays
    }

    // Only keep about 10% of values
    let pattern = create_sparse_pattern(len, 0.1);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::try_from(&BoolArray::from_iter(pattern)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);
}

/// Tests filtering with sparse false values (mostly true)
fn test_sparse_false_filter(array: &dyn Array) {
    let len = array.len();
    if len < 10 {
        return; // Skip for small arrays
    }

    // Keep about 90% of values
    let pattern = create_sparse_pattern(len, 0.9);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::try_from(&BoolArray::from_iter(pattern)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);
}

/// Tests filtering with random pattern
fn test_random_pattern_filter(array: &dyn Array) {
    let len = array.len();

    // Create a pseudo-random pattern based on array length
    let pattern: Vec<bool> = (0..len)
        .map(|i| ((i * 37 + 17) % 5) < 3) // Deterministic pseudo-random
        .collect();
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::try_from(&BoolArray::from_iter(pattern)).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);
}

/// Tests filtering with nullable mask (nulls treated as false)
fn test_nullable_mask_input(array: &dyn Array) {
    let len = array.len();
    if len < 3 {
        return; // Skip for very small arrays
    }

    // Create a nullable mask where every third value is null
    let bool_values: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let validity_values: Vec<bool> = (0..len).map(|i| i % 3 != 0).collect();

    // Only values that are true AND valid should pass the filter
    let expected_count = bool_values
        .iter()
        .zip(validity_values.iter())
        .filter(|(b, v)| **b && **v)
        .count();

    let bool_array = BoolArray::from_iter(bool_values.clone());
    let validity = crate::validity::Validity::from_iter(validity_values.clone());
    let nullable_mask = BoolArray::new(bool_array.boolean_buffer().clone(), validity);

    let mask = Mask::try_from(&nullable_mask).vortex_unwrap();
    let filtered = filter(array, &mask).vortex_unwrap();
    assert_eq!(filtered.len(), expected_count);

    // Verify correct elements are kept
    let mut filtered_idx = 0;
    for i in 0..len {
        if bool_values[i] && validity_values[i] {
            assert_eq!(filtered.scalar_at(filtered_idx), array.scalar_at(i),);
            filtered_idx += 1;
        }
    }
}
