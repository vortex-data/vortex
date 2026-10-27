// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::assert_arrays_eq;
use crate::dtype::DType;

// Standard test array sizes
pub const SMALL_SIZE: usize = 5;
pub const MEDIUM_SIZE: usize = 100;
pub const LARGE_SIZE: usize = 1024;

/// Test filter compute function with various array sizes and patterns.
/// The input array can be of any length.
pub fn test_filter_conformance(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();

    // Test with arrays of any size
    if len > 0 {
        test_all_filter(array, ctx);
        test_none_filter(array);
        test_selective_filter(array, ctx);
        test_single_element_filter(array, ctx);
    }

    // Test edge cases
    test_empty_array_filter(array.dtype());
    test_mismatched_lengths(array);
}

/// Tests that filtering with an all-true mask returns all elements unchanged
fn test_all_filter(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    let mask = Mask::new_true(len);
    let filtered = array
        .filter(mask)
        .vortex_expect("filter should succeed in conformance test");
    assert_arrays_eq!(filtered, array, ctx);
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

fn test_selective_filter(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    let len = array.len();
    let mut patterns: Vec<(&str, Vec<bool>)> = vec![
        ("alternating", (0..len).map(|i| i % 2 == 0).collect()),
        (
            "first_and_last",
            (0..len).map(|i| i == 0 || i == len - 1).collect(),
        ),
    ];
    if len >= 4 {
        patterns.push((
            "runs",
            (0..len).map(|i| (i / 3).is_multiple_of(2)).collect(),
        ));
        patterns.push((
            "pseudo_random",
            (0..len).map(|i| ((i * 37 + 17) % 5) < 3).collect(),
        ));
    }
    if len >= 10 {
        for (name, ratio) in [("sparse_true", 0.1), ("sparse_false", 0.9)] {
            patterns.push((
                name,
                (0..len).map(|i| (i as f64 / len as f64) < ratio).collect(),
            ));
        }
    }

    for (name, pattern) in patterns {
        let expected_indices: Vec<_> = pattern
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| keep.then_some(i))
            .collect();
        let filtered = array
            .filter(Mask::from_iter(pattern))
            .vortex_expect("filter should succeed in conformance test")
            .execute::<RecursiveCanonical>(ctx)
            .vortex_expect("filter batch execution should succeed")
            .0
            .into_array();
        assert_eq!(filtered.len(), expected_indices.len(), "{name}");
        assert_eq!(filtered.dtype(), array.dtype(), "{name}");
        for (output_idx, input_idx) in expected_indices.into_iter().enumerate() {
            assert_eq!(
                filtered
                    .execute_scalar(output_idx, ctx)
                    .vortex_expect("scalar_at should succeed in conformance test"),
                array
                    .execute_scalar(input_idx, ctx)
                    .vortex_expect("scalar_at should succeed in conformance test"),
                "{name}: output index {output_idx}, input index {input_idx}"
            );
        }
    }
}

fn test_single_element_filter(array: &ArrayRef, ctx: &mut ExecutionCtx) {
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
        .vortex_expect("filter should succeed in conformance test")
        .execute::<RecursiveCanonical>(ctx)
        .vortex_expect("filter batch execution should succeed")
        .0
        .into_array();
    assert_eq!(filtered.len(), 1);
    assert_eq!(
        filtered
            .execute_scalar(0, ctx)
            .vortex_expect("scalar_at should succeed in conformance test"),
        array
            .execute_scalar(0, ctx)
            .vortex_expect("scalar_at should succeed in conformance test")
    );

    // Test selecting only the last element
    if len > 1 {
        let mut mask_values = vec![false; len];
        mask_values[len - 1] = true;
        let mask = Mask::from_iter(mask_values);
        let filtered = array
            .filter(mask)
            .vortex_expect("filter should succeed in conformance test")
            .execute::<RecursiveCanonical>(ctx)
            .vortex_expect("filter batch execution should succeed")
            .0
            .into_array();
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered
                .execute_scalar(0, ctx)
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(len - 1, ctx)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_empty_array_filter(dtype: &DType) {
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
