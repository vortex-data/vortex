// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::list_view_from_list;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

#[test]
fn test_basic_listview_comprehensive() {
    // Comprehensive test for basic ListView functionality including scalar_at.
    // Logical lists: [[1,2,3], [4,5], [6,7,8,9]]
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![0i32, 3, 5].into_array();
    let sizes = buffer![3i32, 2, 4].into_array();

    let listview = unsafe {
        ListViewArray::new_unchecked(elements.into_array(), offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    };

    assert_eq!(listview.len(), 3);
    assert!(!listview.is_empty());

    // Check the dtype.
    assert!(matches!(
        listview.dtype(),
        DType::List(elem_dtype, Nullability::NonNullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::I32, Nullability::NonNullable))
    ));

    // Check individual list elements.
    assert_arrays_eq!(
        listview.list_elements_at(0).unwrap(),
        PrimitiveArray::from_iter([1i32, 2, 3])
    );

    // Test scalar_at which returns entire lists as Scalar values.
    let first_scalar = listview
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert_eq!(
        first_scalar,
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1i32.into(), 2i32.into(), 3i32.into()],
            Nullability::NonNullable,
        )
    );

    assert_arrays_eq!(
        listview.list_elements_at(1).unwrap(),
        PrimitiveArray::from_iter([4i32, 5])
    );

    assert_arrays_eq!(
        listview.list_elements_at(2).unwrap(),
        PrimitiveArray::from_iter([6i32, 7, 8, 9])
    );
}

#[test]
fn test_out_of_order_offsets() {
    // ListView-specific: Tests that offsets can be non-sequential and out-of-order.
    // Logical lists: [[7,8,9], [1,2,3], [4,5,6]]
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![6i32, 0, 3].into_array(); // Out-of-order offsets.
    let sizes = buffer![3i32, 3, 3].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 3);

    // First list starts at offset 6: [7, 8, 9].
    assert_arrays_eq!(
        listview.list_elements_at(0).unwrap(),
        PrimitiveArray::from_iter([7i32, 8, 9])
    );

    // Second list starts at offset 0: [1, 2, 3].
    assert_arrays_eq!(
        listview.list_elements_at(1).unwrap(),
        PrimitiveArray::from_iter([1i32, 2, 3])
    );
}

#[test]
fn test_empty_listview() {
    // Test empty ListView array (0 lists).
    // Logical lists: [] (empty ListView)
    let elements = buffer![1i32].into_array(); // Dummy element.
    let offsets = buffer![0i32; 0].into_array();
    let sizes = buffer![0i32; 0].into_array();

    let listview = unsafe {
        ListViewArray::new_unchecked(elements.into_array(), offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    };

    assert_eq!(listview.len(), 0);
    assert!(listview.is_empty());
}

#[test]
fn test_from_list_array() -> VortexResult<()> {
    // Test conversion from ListArray to ListViewArray.
    // Logical lists: [[1,2], null, [5,6,7]]
    let offsets = buffer![0i64, 2, 4, 7].into_array();
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7].into_array();
    let validity = Validity::from_iter([true, false, true]);

    let list_array = ListArray::try_new(elements, offsets, validity).unwrap();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let list_view = list_view_from_list(list_array, &mut ctx)?;

    assert_eq!(list_view.len(), 3);

    // Check first list.
    assert_arrays_eq!(
        list_view.list_elements_at(0).unwrap(),
        PrimitiveArray::from_iter([1i32, 2])
    );

    // Check validity is preserved.
    assert!(
        list_view
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert!(
        list_view
            .is_invalid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert!(
        list_view
            .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );

    // Check third list.
    assert_arrays_eq!(
        list_view.list_elements_at(2)?,
        PrimitiveArray::from_iter([5i32, 6, 7])
    );
    Ok(())
}

// Parameterized tests for ConstantArray scenarios.
#[rstest]
#[case::constant_sizes(true, false)] // Constant sizes, varying offsets
#[case::constant_offsets(false, true)] // Varying sizes, constant offsets
#[case::both_constant(true, true)] // Both constant
fn test_listview_with_constant_arrays(#[case] const_sizes: bool, #[case] const_offsets: bool) {
    // Logical lists vary by case:
    // - constant_sizes: [[1,2,3], [4,5,6], [7,8,9]] (size 3 each, varying offsets)
    // - constant_offsets: [[1,2,3], [1,2], [1]] (all start at 0, varying sizes)
    // - both_constant: [[1,2,3], [1,2,3], [1,2,3]] (all identical)
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();

    let offsets = if const_offsets {
        ConstantArray::new(0i32, 3).into_array()
    } else {
        buffer![0i32, 3, 6].into_array()
    };

    let sizes = if const_sizes {
        ConstantArray::new(3i32, 3).into_array()
    } else {
        buffer![3i32, 2, 1].into_array()
    };

    // Determine if the array is zero-copy to list based on test case.
    // The array is NOT zero-copy when there are overlaps (const_offsets case).
    let is_zctl = !const_offsets;

    let listview = unsafe {
        ListViewArray::new_unchecked(elements.into_array(), offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(is_zctl)
    };
    assert_eq!(listview.len(), 3);

    if const_sizes && const_offsets {
        // All lists are identical [1, 2, 3] (overlapping).
        let expected = PrimitiveArray::from_iter([1i32, 2, 3]);
        for i in 0..3 {
            assert_arrays_eq!(listview.list_elements_at(i).unwrap(), expected);
        }
    } else if const_sizes {
        // All lists have size 3, different offsets (no overlap).
        assert_eq!(listview.list_elements_at(0).unwrap().len(), 3);
        assert_eq!(listview.list_elements_at(1).unwrap().len(), 3);
        assert_eq!(listview.list_elements_at(2).unwrap().len(), 3);
    } else if const_offsets {
        // All lists start at offset 0, different sizes (overlapping).
        assert_eq!(
            listview
                .list_elements_at(0)
                .unwrap()
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            1i32.into()
        );
        assert_eq!(
            listview
                .list_elements_at(1)
                .unwrap()
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            1i32.into()
        );
        assert_eq!(
            listview
                .list_elements_at(2)
                .unwrap()
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            1i32.into()
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Validation tests
////////////////////////////////////////////////////////////////////////////////////////////////////

// Parameterized validation error tests.
#[rstest]
#[case::offset_size_overflow(
    buffer![1i32, 2, 3],
    buffer![2i32, 0],
    buffer![3i32, 1],
    "exceeds elements length"
)]
#[case::length_mismatch(
    buffer![1i32, 2, 3],
    buffer![0i32, 1],
    buffer![1i32, 1, 1],
    "same length"
)]
fn test_validation_errors(
    #[case] elements: vortex_buffer::Buffer<i32>,
    #[case] offsets: vortex_buffer::Buffer<i32>,
    #[case] sizes: vortex_buffer::Buffer<i32>,
    #[case] expected_error: &str,
) {
    let result = ListViewArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains(expected_error));
}

#[test]
fn test_validate_nullable_offsets() {
    // Logical lists (invalid due to nullable offsets): [[1,2], [3], ???]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
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
    // Logical lists (invalid due to nullable sizes): [[1,2], ???, [2,3]]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 1].into_array();
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
fn test_validate_offset_plus_size_overflow() {
    // Logical lists (invalid due to overflow): would overflow, [[1], [1]]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    // Create an offset + size that would overflow.
    let offsets = buffer![u32::MAX - 1, 0, 0].into_array();
    let sizes = buffer![2u32, 1, 1].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("overflow") || err.to_string().contains("exceeds"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_validate_invalid_validity_length() {
    // Logical lists (invalid due to validity length mismatch): [[1,2], [3,4], [5]]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 1].into_array();
    // Validity has wrong length.
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false]).into_array());

    let result = ListViewArray::try_new(elements, offsets, sizes, validity);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("validity") && err.to_string().contains("size"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_validate_non_integer_offsets() {
    // Logical lists (invalid due to float offsets): [[1,2], [3,4], [5]]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    // Try to use float offsets.
    let offsets = buffer![0.0f32, 2.0, 4.0].into_array();
    let sizes = buffer![2u32, 2, 1].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("integer"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_validate_different_int_types() {
    // Test that different integer types work as long as sizes type ≤ offsets type.
    // Logical lists: [[1,2], [3], [2,3]]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u64, 2, 1].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();

    let _listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);
}

#[test]
fn test_validate_u64_overflow() {
    // Test overflow with u64 offsets and sizes.
    // Logical lists (invalid due to u64 overflow): would overflow, [[0], [0]]
    let elements = PrimitiveArray::from_iter(0i32..100).into_array();
    let offsets = buffer![u64::MAX - 10, 0, 0].into_array();
    let sizes = buffer![20u64, 1, 1].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_verify_is_zero_copy_to_list() {
    // Create a ListView that IS zero-copyable to List.
    // Logical lists: [[1,2], [3,4], [5]]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0i32, 2, 4].into_array(); // Sorted, no gaps
    let sizes = buffer![2i32, 2, 1].into_array(); // No overlaps

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

    // Should return true since offsets are sorted and no overlaps exist.
    assert!(listview.verify_is_zero_copy_to_list());

    // Create a ListView that is NOT zero-copyable to List due to overlapping views.
    // Logical lists: [[1,2], [2,3,4], [3,4]]
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0i32, 1, 2].into_array(); // Sorted but overlapping
    let sizes = buffer![2i32, 3, 2].into_array(); // These cause overlaps

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable);

    // Should return false due to overlapping list views.
    assert!(!listview.verify_is_zero_copy_to_list());
}

#[test]
#[should_panic(expected = "Zero-copy-to-list requires views to be non-overlapping and ordered")]
fn test_validate_monotonic_ends_with_nulls() {
    // Regression test for issue #5412
    // Tests that validate_zctl catches incorrect NULL offsets

    // Create an array with buggy NULL offsets (as would be produced by the old naive_rebuild)
    // Elements: [1, 2, 3, 4]
    // View 0: [1, 2] at offset 0
    // View 1: [3, 4] at offset 2
    // View 2 (NULL): incorrectly at offset 2 (should be 4)
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0u32, 2, 2].into_array(); // Bug: NULL reuses offset 2
    let sizes = buffer![2u32, 2, 0].into_array();
    let validity = Validity::from_iter(vec![true, true, false]);

    let listview = ListViewArray::new(elements, offsets, sizes, validity);

    // The array itself is valid (can be constructed)
    assert_eq!(listview.len(), 3);

    // But it should NOT be valid as zero-copy-to-list due to the monotonic violation
    // offset[1] + size[1] = 2 + 2 = 4, but offset[2] = 2, violating 4 <= 2
    // This should panic with our new monotonic check
    unsafe {
        let _zctl = listview.with_zero_copy_to_list(true);
    }
}

#[test]
fn test_validate_monotonic_ends_correct_nulls() {
    // Test that correctly placed NULLs pass validation
    // Elements: [1, 2, 3, 4]
    // View 0: [1, 2] at offset 0
    // View 1: [3, 4] at offset 2
    // View 2 (NULL): correctly at offset 4 (after all data)
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0u32, 2, 4].into_array(); // Correct: NULL at position 4
    let sizes = buffer![2u32, 2, 0].into_array();
    let validity = Validity::from_iter(vec![true, true, false]);

    let listview = ListViewArray::new(elements, offsets, sizes, validity);

    // Should be valid as zero-copy-to-list - this should NOT panic
    let zctl_listview = unsafe { listview.clone().with_zero_copy_to_list(true) };
    assert!(zctl_listview.is_zero_copy_to_list());

    // verify_is_zero_copy_to_list should also return true
    assert!(listview.verify_is_zero_copy_to_list());
}
