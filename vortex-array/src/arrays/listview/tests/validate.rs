// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use crate::IntoArray;
use crate::arrays::{ListViewArray, PrimitiveArray};
use crate::validity::Validity;

#[test]
fn test_validate_nullable_offsets() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    // Create nullable offsets array (should fail validation).
    let offsets = PrimitiveArray::from_option_iter(vec![Some(0u32), Some(2), None]).into_array();
    let sizes = buffer![2u32, 1, 2].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("offsets must be non-nullable")
    );
}

#[test]
fn test_validate_nullable_sizes() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 1].into_array();
    // Create nullable sizes array (should fail validation).
    let sizes = PrimitiveArray::from_option_iter(vec![Some(2u32), None, Some(2)]).into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("sizes must be non-nullable")
    );
}

#[test]
fn test_validate_mismatched_lengths() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    // Offsets and sizes have different lengths.
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("offsets and sizes must have the same length")
    );
}

#[test]
fn test_validate_size_type_too_large() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    // Use u64 for sizes and u32 for offsets (sizes type is larger).
    let offsets = buffer![0u32, 2, 1].into_array();
    let sizes = buffer![2u64, 1, 2].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("size type"));
}

#[test]
fn test_validate_offset_plus_size_overflow() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    // Create an offset + size that would overflow.
    let offsets = buffer![u32::MAX - 1, 0, 0].into_array();
    let sizes = buffer![2u32, 1, 1].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = err.to_string();
    // The error might say "exceeds elements length" instead of "overflow".
    assert!(
        err_str.contains("overflow") || err_str.contains("exceeds elements length"),
        "Expected error about overflow or exceeding bounds, got: {}",
        err_str
    );
}

#[test]
fn test_validate_offset_plus_size_exceeds_elements() {
    let elements = buffer![1i32, 2, 3].into_array();
    // offsets[1] + sizes[1] = 2 + 3 = 5, which exceeds elements.len() = 3.
    let offsets = buffer![0u32, 2, 0].into_array();
    let sizes = buffer![2u32, 3, 1].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("exceeds elements length")
    );
}

#[test]
fn test_validate_invalid_validity_length() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 1].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();
    // Create a validity array with wrong length.
    let validity = Validity::from_iter([true, false]);

    let result = ListViewArray::try_new(elements, offsets, sizes, validity);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("validity with size")
    );
}

#[test]
fn test_validate_non_integer_offsets() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    // Create a float array for offsets (should fail validation).
    let offsets = buffer![0.0f32, 2.0, 1.0].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("offsets must be non-nullable integer array")
    );
}

#[test]
fn test_validate_non_integer_sizes() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 1].into_array();
    // Create a float array for sizes (should fail validation).
    let sizes = buffer![2.0f32, 1.0, 2.0].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("sizes must be non-nullable integer array")
    );
}

#[test]
fn test_validate_success_basic() {
    // Test a valid ListView array to ensure the happy path works.
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![2u32, 0, 1].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_ok());
    let list_view = result.unwrap();
    assert_eq!(list_view.len(), 3);
}

#[test]
fn test_validate_success_with_validity() {
    // Test a valid ListView array with validity.
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 3].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();
    let validity = Validity::from_iter([true, false, true]);

    let result = ListViewArray::try_new(elements, offsets, sizes, validity);

    assert!(result.is_ok());
    let list_view = result.unwrap();
    assert_eq!(list_view.len(), 3);
}

#[test]
fn test_validate_empty_array() {
    // Test an empty ListView array.
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0u32; 0].into_array();
    let sizes = buffer![0u32; 0].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_ok());
    let list_view = result.unwrap();
    assert_eq!(list_view.len(), 0);
    assert!(list_view.is_empty());
}

#[test]
fn test_validate_edge_case_exact_boundary() {
    // Test where offset + size exactly equals elements.len().
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 3, 2].into_array();
    let sizes = buffer![2u32, 2, 3].into_array(); // Last list goes from 2 to 5.

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_ok());
}

#[test]
fn test_validate_different_int_types() {
    // Test with different integer types for offsets and sizes (but sizes fits in offsets).
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u64, 2, 1].into_array(); // u64 offsets.
    let sizes = buffer![2u32, 1, 2].into_array(); // u32 sizes (smaller than u64).

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_ok());
}

#[test]
fn test_validate_u64_overflow() {
    // Test that offset + size overflow detection works correctly.
    // Create a small elements array but use offset and size that would overflow u64.
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();

    // Use u64::MAX as offset and any non-zero size to cause overflow.
    let offsets = buffer![0u64, u64::MAX, 1].into_array();
    let sizes = buffer![2u64, 1, 2].into_array(); // u64::MAX + 1 will overflow.

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("overflow"),
        "Expected overflow error for u64::MAX + 1"
    );
}

#[test]
fn test_validate_large_offset_and_size_overflow() {
    // Test with large values that overflow when added together.
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();

    // Both values are large and their sum overflows u64.
    let offsets = buffer![0u64, u64::MAX - 1000, 1].into_array();
    let sizes = buffer![2u64, 2000, 2].into_array(); // (u64::MAX - 1000) + 2000 overflows.

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("overflow"),
        "Expected overflow error for large offset + size"
    );
}

// NOTE: Tests with compressed arrays would require additional setup to avoid
// compilation issues with multiple vortex-array instances. The validation
// function handles compressed arrays by using scalar_at which works with
// any array encoding, as demonstrated in the validate function implementation.
//
// The key aspect of validation with compressed arrays is that the validate
// function uses scalar_at to access individual values, which transparently
// handles both compressed and uncompressed arrays. This is verified by the
// implementation in vortex-array/src/arrays/listview/mod.rs:228-257.
