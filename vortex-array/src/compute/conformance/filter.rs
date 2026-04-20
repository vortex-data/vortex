// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::assert_arrays_eq;
use crate::dtype::DType;

// Standard test array sizes
pub const SMALL_SIZE: usize = 5;
pub const MEDIUM_SIZE: usize = 100;
pub const LARGE_SIZE: usize = 1024;

/// Test filter compute function with various array sizes and patterns.
/// The input array can be of any length.
pub fn test_filter_conformance(array: &ArrayRef) {
    let len = array.len();

    // Test with arrays of any size
    if len > 0 {
        test_all_filter(array);
        test_none_filter(array);
        test_selective_filter(array);
        test_single_element_filter(array);
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
    (0..len)
        .map(|i| (i / run_length).is_multiple_of(2))
        .collect()
}

/// Tests that filtering with an all-true mask returns all elements unchanged
fn test_all_filter(array: &ArrayRef) {
    let len = array.len();
    let mask = Mask::new_true(len);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_arrays_eq!(filtered, array);
}

/// Tests that filtering with an all-false mask returns an empty array with the same dtype
fn test_none_filter(array: &ArrayRef) {
    let len = array.len();
    let mask = Mask::new_false(len);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), 0);
    assert_eq!(filtered.dtype(), array.dtype());
}

fn test_selective_filter(array: &ArrayRef) {
    let len = array.len();
    if len < 2 {
        return; // Skip for very small arrays
    }

    // Test alternating pattern
    let mask_values: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let expected_count = mask_values.iter().filter(|&&v| v).count();
    let mask = Mask::from_iter(mask_values);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), expected_count);

    // Verify correct elements are kept
    for (filtered_idx, i) in (0..len).step_by(2).enumerate() {
        assert_eq!(
            filtered
                .execute_scalar(filtered_idx, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }

    // Test first and last only
    if len >= 2 {
        let mut mask_values = vec![false; len];
        mask_values[0] = true;
        mask_values[len - 1] = true;
        let mask = Mask::from_iter(mask_values);
        let filtered = array
            .filter(mask)
            .vortex_expect("filter should succeed in conformance test");
        assert_eq!(filtered.len(), 2);
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
                .execute_scalar(len - 1, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_single_element_filter(array: &ArrayRef) {
    let len = array.len();
    if len == 0 {
        return;
    }

    // Test selecting only the first element
    let mut mask_values = vec![false; len];
    mask_values[0] = true;
    let mask = Mask::from_iter(mask_values);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), 1);
    assert_eq!(
        filtered
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test"),
        array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("scalar_at should succeed in conformance test")
    );

    // Test selecting only the last element
    if len > 1 {
        let mut mask_values = vec![false; len];
        mask_values[len - 1] = true;
        let mask = Mask::from_iter(mask_values);
        let filtered = array
            .filter(mask)
            .vortex_expect("filter should succeed in conformance test");
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(len - 1, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_empty_array_filter(dtype: &DType) {
    use crate::Canonical;

    let empty_array = Canonical::empty(dtype).into_array();
    let empty_mask = Mask::new_false(0);
    let filtered = empty_array
        .filter(empty_mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), 0);

    let empty_mask = Mask::new_true(0);
    let filtered = empty_array
        .filter(empty_mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), 0);
}

fn test_mismatched_lengths(array: &ArrayRef) {
    let len = array.len();

    // Test mask shorter than array
    if len > 0 {
        let short_mask = Mask::new_true(len - 1);
        let result = array.filter(short_mask);
        assert!(
            result.is_err(),
            "Filter should fail with mismatched lengths"
        );
    }

    // Test mask longer than array
    let long_mask = Mask::new_true(len + 1);
    let result = array.filter(long_mask);
    assert!(
        result.is_err(),
        "Filter should fail with mismatched lengths"
    );
}

/// Tests filtering with alternating true/false pattern
fn test_alternating_pattern_filter(array: &ArrayRef) {
    let len = array.len();
    let pattern = create_alternating_pattern(len);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::from_iter(pattern.clone());
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), expected_count);

    // Verify correct elements are kept
    let mut filtered_idx = 0;
    for (i, &keep) in pattern.iter().enumerate() {
        if keep {
            assert_eq!(
                filtered
                    .execute_scalar(filtered_idx, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test"),
                array
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test")
            );
            filtered_idx += 1;
        }
    }
}

/// Tests filtering with runs of true/false values
fn test_runs_pattern_filter(array: &ArrayRef) {
    let len = array.len();
    if len < 4 {
        return; // Skip for very small arrays
    }

    let run_length = len.min(3);
    let pattern = create_runs_pattern(len, run_length);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::from_iter(pattern);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), expected_count);
}

/// Tests filtering with sparse true values (mostly false)
fn test_sparse_true_filter(array: &ArrayRef) {
    let len = array.len();
    if len < 10 {
        return; // Skip for small arrays
    }

    // Only keep about 10% of values
    let pattern = create_sparse_pattern(len, 0.1);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::from_iter(pattern);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), expected_count);
}

/// Tests filtering with sparse false values (mostly true)
fn test_sparse_false_filter(array: &ArrayRef) {
    let len = array.len();
    if len < 10 {
        return; // Skip for small arrays
    }

    // Keep about 90% of values
    let pattern = create_sparse_pattern(len, 0.9);
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::from_iter(pattern);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), expected_count);
}

/// Tests filtering with random pattern
fn test_random_pattern_filter(array: &ArrayRef) {
    let len = array.len();

    // Create a pseudo-random pattern based on array length
    let pattern: Vec<bool> = (0..len)
        .map(|i| ((i * 37 + 17) % 5) < 3) // Deterministic pseudo-random
        .collect();
    let expected_count = pattern.iter().filter(|&&v| v).count();

    let mask = Mask::from_iter(pattern);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_eq!(filtered.len(), expected_count);
}
