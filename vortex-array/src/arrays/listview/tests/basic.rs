// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_scalar::Scalar;

use crate::arrays::{
    BoolArray, ConstantArray, ListArray, ListViewArray, PrimitiveArray, list_view_from_list,
};
use crate::validity::Validity;
use crate::{Array, IntoArray};

#[test]
fn test_basic_listview_comprehensive() {
    // Comprehensive test for basic ListView functionality including scalar_at.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![0i32, 3, 5].into_array();
    let sizes = buffer![3i32, 2, 4].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 3);
    assert!(!listview.is_empty());

    // Check the dtype.
    assert!(matches!(
        listview.dtype(),
        DType::List(elem_dtype, Nullability::NonNullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::I32, Nullability::NonNullable))
    ));

    // Check individual list elements.
    let first_list = listview.list_elements_at(0);
    assert_eq!(first_list.len(), 3);
    assert_eq!(first_list.scalar_at(0), 1i32.into());
    assert_eq!(first_list.scalar_at(1), 2i32.into());
    assert_eq!(first_list.scalar_at(2), 3i32.into());

    // Test scalar_at which returns entire lists as Scalar values.
    let first_scalar = listview.scalar_at(0);
    assert_eq!(
        first_scalar,
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1i32.into(), 2i32.into(), 3i32.into()],
            Nullability::NonNullable,
        )
    );

    let second_list = listview.list_elements_at(1);
    assert_eq!(second_list.len(), 2);
    assert_eq!(second_list.scalar_at(0), 4i32.into());
    assert_eq!(second_list.scalar_at(1), 5i32.into());

    let third_list = listview.list_elements_at(2);
    assert_eq!(third_list.len(), 4);
    assert_eq!(third_list.scalar_at(0), 6i32.into());
    assert_eq!(third_list.scalar_at(1), 7i32.into());
    assert_eq!(third_list.scalar_at(2), 8i32.into());
    assert_eq!(third_list.scalar_at(3), 9i32.into());
}

#[test]
fn test_out_of_order_offsets() {
    // ListView-specific: Tests that offsets can be non-sequential and out-of-order.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![6i32, 0, 3].into_array(); // Out-of-order: [7,8,9], [1,2,3], [4,5,6].
    let sizes = buffer![3i32, 3, 3].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 3);

    // First list starts at offset 6: [7, 8, 9].
    let first = listview.list_elements_at(0);
    assert_eq!(first.scalar_at(0), 7i32.into());
    assert_eq!(first.scalar_at(1), 8i32.into());
    assert_eq!(first.scalar_at(2), 9i32.into());

    // Second list starts at offset 0: [1, 2, 3].
    let second = listview.list_elements_at(1);
    assert_eq!(second.scalar_at(0), 1i32.into());
    assert_eq!(second.scalar_at(1), 2i32.into());
    assert_eq!(second.scalar_at(2), 3i32.into());
}

#[test]
fn test_empty_listview() {
    // Test empty ListView array (0 lists).
    let elements = buffer![1i32].into_array(); // Dummy element.
    let offsets = buffer![0i32; 0].into_array();
    let sizes = buffer![0i32; 0].into_array();

    let listview =
        ListViewArray::try_new(elements.into_array(), offsets, sizes, Validity::NonNullable)
            .unwrap();

    assert_eq!(listview.len(), 0);
    assert!(listview.is_empty());
}

#[test]
fn test_from_list_array() {
    // Test conversion from ListArray to ListViewArray.
    let offsets = buffer![0i64, 2, 4, 7].into_array();
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7].into_array();
    let validity = Validity::from_iter([true, false, true]);

    let list_array = ListArray::try_new(elements, offsets, validity).unwrap();
    let list_view = list_view_from_list(list_array);

    assert_eq!(list_view.len(), 3);

    // Check first list.
    let first = list_view.list_elements_at(0);
    assert_eq!(first.scalar_at(0), 1i32.into());
    assert_eq!(first.scalar_at(1), 2i32.into());

    // Check validity is preserved.
    assert!(list_view.is_valid(0));
    assert!(list_view.is_invalid(1));
    assert!(list_view.is_valid(2));

    // Check third list.
    let third = list_view.list_elements_at(2);
    assert_eq!(third.scalar_at(0), 5i32.into());
    assert_eq!(third.scalar_at(1), 6i32.into());
    assert_eq!(third.scalar_at(2), 7i32.into());
}

// Parameterized tests for ConstantArray scenarios.
#[rstest]
#[case::constant_sizes(true, false)] // Constant sizes, varying offsets
#[case::constant_offsets(false, true)] // Varying sizes, constant offsets
#[case::both_constant(true, true)] // Both constant
fn test_listview_with_constant_arrays(#[case] const_sizes: bool, #[case] const_offsets: bool) {
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

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);
    assert_eq!(listview.len(), 3);

    if const_sizes && const_offsets {
        // All lists are identical [1, 2, 3].
        for i in 0..3 {
            let list = listview.list_elements_at(i);
            assert_eq!(list.scalar_at(0), 1i32.into());
            assert_eq!(list.scalar_at(1), 2i32.into());
            assert_eq!(list.scalar_at(2), 3i32.into());
        }
    } else if const_sizes {
        // All lists have size 3, different offsets.
        assert_eq!(listview.list_elements_at(0).len(), 3);
        assert_eq!(listview.list_elements_at(1).len(), 3);
        assert_eq!(listview.list_elements_at(2).len(), 3);
    } else if const_offsets {
        // All lists start at offset 0, different sizes.
        assert_eq!(listview.list_elements_at(0).scalar_at(0), 1i32.into());
        assert_eq!(listview.list_elements_at(1).scalar_at(0), 1i32.into());
        assert_eq!(listview.list_elements_at(2).scalar_at(0), 1i32.into());
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
    assert!(
        err.to_string().contains("overflow") || err.to_string().contains("exceeds"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_validate_invalid_validity_length() {
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
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u64, 2, 1].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();

    let result = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable);
    assert!(result.is_ok());
}

#[test]
fn test_validate_u64_overflow() {
    // Test overflow with u64 offsets and sizes.
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
