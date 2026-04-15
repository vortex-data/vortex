// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builtins::ArrayBuiltins;

/// Test mask compute function with various array sizes and patterns.
/// The mask operation sets elements to null where the mask is true.
pub fn test_mask_conformance(array: &ArrayRef) {
    let len = array.len();

    if len > 0 {
        test_heterogenous_mask(array);
        test_empty_mask(array);
        test_full_mask(array);
        test_alternating_mask(array);
        test_sparse_mask(array);
        test_single_element_mask(array);
    }

    if len >= 5 {
        test_double_mask(array);
    }

    if len > 0 {
        test_nullable_mask_input(array);
    }
}

/// Tests masking with a heterogeneous pattern
fn test_heterogenous_mask(array: &ArrayRef) {
    let len = array.len();

    // Create a pattern where roughly half the values are masked
    let mask_pattern: Vec<bool> = (0..len).map(|i| i % 3 != 1).collect();
    let mask_array = Mask::from_iter(mask_pattern.clone());

    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");
    assert_eq!(masked.len(), array.len());

    // Verify masked elements are null and unmasked elements are preserved
    for (i, &masked_out) in mask_pattern.iter().enumerate() {
        if masked_out {
            assert!(
                !masked
                    .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("is_valid should succeed in conformance test")
            );
        } else {
            assert_eq!(
                masked
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test"),
                array
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test")
                    .into_nullable()
            );
        }
    }
}

/// Tests that an empty mask (all false) preserves all elements
fn test_empty_mask(array: &ArrayRef) {
    let len = array.len();
    let all_unmasked = vec![false; len];
    let mask_array = Mask::from_iter(all_unmasked);

    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");
    assert_eq!(masked.len(), array.len());

    // All elements should be preserved
    for i in 0..len {
        assert_eq!(
            masked
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
                .into_nullable()
        );
    }
}

/// Tests that a full mask (all true) makes all elements null
fn test_full_mask(array: &ArrayRef) {
    let len = array.len();
    let all_masked = vec![true; len];
    let mask_array = Mask::from_iter(all_masked);

    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");
    assert_eq!(masked.len(), array.len());

    // All elements should be null
    for i in 0..len {
        assert!(
            !masked
                .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("is_valid should succeed in conformance test")
        );
    }
}

/// Tests alternating mask pattern
fn test_alternating_mask(array: &ArrayRef) {
    let len = array.len();
    let pattern: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let mask_array = Mask::from_iter(pattern);

    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");
    assert_eq!(masked.len(), array.len());

    for i in 0..len {
        if i % 2 == 0 {
            assert!(
                !masked
                    .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("is_valid should succeed in conformance test")
            );
        } else {
            assert_eq!(
                masked
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test"),
                array
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test")
                    .into_nullable()
            );
        }
    }
}

/// Tests sparse mask (only a few elements masked)
fn test_sparse_mask(array: &ArrayRef) {
    let len = array.len();
    if len < 10 {
        return; // Skip for small arrays
    }

    // Mask only about 10% of elements
    let pattern: Vec<bool> = (0..len).map(|i| i % 10 == 0).collect();
    let mask_array = Mask::from_iter(pattern.clone());

    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");
    assert_eq!(masked.len(), array.len());

    // Count how many elements are valid after masking
    let valid_count = (0..len)
        .filter(|&i| {
            masked
                .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("is_valid should succeed in conformance test")
        })
        .count();

    // Count how many elements should be invalid:
    // - Elements that were masked (pattern[i] == true)
    // - Elements that were already invalid in the original array
    let expected_invalid_count = (0..len)
        .filter(|&i| {
            pattern[i]
                || !array
                    .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("is_valid should succeed in conformance test")
        })
        .count();

    assert_eq!(valid_count, len - expected_invalid_count);
}

/// Tests masking a single element
fn test_single_element_mask(array: &ArrayRef) {
    let len = array.len();

    // Mask only the first element
    let mut pattern = vec![false; len];
    pattern[0] = true;
    let mask_array = Mask::from_iter(pattern);

    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");
    assert!(
        !masked
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("is_valid should succeed in conformance test")
    );

    for i in 1..len {
        assert_eq!(
            masked
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test"),
            array
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("scalar_at should succeed in conformance test")
                .into_nullable()
        );
    }
}

/// Tests double masking operations
fn test_double_mask(array: &ArrayRef) {
    let len = array.len();

    // Create two different mask patterns
    let mask1_pattern: Vec<bool> = (0..len).map(|i| i % 3 == 0).collect();
    let mask2_pattern: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();

    let mask1 = Mask::from_iter(mask1_pattern.clone());
    let mask2 = Mask::from_iter(mask2_pattern.clone());

    let first_masked = array
        .clone()
        .mask((!&mask1).into_array())
        .vortex_expect("mask should succeed in conformance test");
    let double_masked = first_masked
        .mask((!&mask2).into_array())
        .vortex_expect("mask should succeed in conformance test");

    // Elements should be null if either mask is true
    for i in 0..len {
        if mask1_pattern[i] || mask2_pattern[i] {
            assert!(
                !double_masked
                    .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("is_valid should succeed in conformance test")
            );
        } else {
            assert_eq!(
                double_masked
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test"),
                array
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test")
                    .into_nullable()
            );
        }
    }
}

/// Tests masking with nullable mask (nulls treated as false)
fn test_nullable_mask_input(array: &ArrayRef) {
    let len = array.len();
    if len < 3 {
        return; // Skip for very small arrays
    }

    // Create a nullable mask
    let bool_values: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
    let validity_values: Vec<bool> = (0..len).map(|i| i % 3 != 0).collect();

    let bool_array = BoolArray::from_iter(bool_values.clone());
    let validity = crate::validity::Validity::from_iter(validity_values.clone());
    let nullable_mask = BoolArray::new(bool_array.to_bit_buffer(), validity);

    let mask_array =
        nullable_mask.to_mask_fill_null_false(&mut LEGACY_SESSION.create_execution_ctx());
    let masked = array
        .clone()
        .mask((!&mask_array).into_array())
        .vortex_expect("mask should succeed in conformance test");

    // Elements are masked only if the mask is true AND valid
    for i in 0..len {
        if bool_values[i] && validity_values[i] {
            assert!(
                !masked
                    .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("is_valid should succeed in conformance test")
            );
        } else {
            assert_eq!(
                masked
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test"),
                array
                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                    .vortex_expect("scalar_at should succeed in conformance test")
                    .into_nullable()
            );
        }
    }
}
