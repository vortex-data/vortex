// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_scalar::Scalar;

use crate::IntoArray;
use crate::arrays::{BoolArray, ListViewArray, PrimitiveArray};
use crate::validity::Validity;

#[test]
fn test_nullable_listview_comprehensive() {
    // Comprehensive test for nullable ListView including scalar_at with nulls.
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();
    let validity = Validity::from_iter([true, false, true]);

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity).unwrap();

    assert_eq!(listview.len(), 3);

    // Check validity.
    assert!(listview.is_valid(0));
    assert!(listview.is_invalid(1));
    assert!(listview.is_valid(2));

    // Check dtype reflects nullability.
    assert!(matches!(
        listview.dtype(),
        DType::List(_, Nullability::Nullable)
    ));

    // Test scalar_at with nulls.
    let first = listview.scalar_at(0);
    assert!(!first.is_null());
    assert_eq!(
        first,
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1i32.into(), 2i32.into()],
            Nullability::Nullable,
        )
    );

    let second = listview.scalar_at(1);
    assert!(second.is_null());

    let third = listview.scalar_at(2);
    assert!(!third.is_null());
    assert_eq!(
        third,
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![5i32.into(), 6i32.into()],
            Nullability::Nullable,
        )
    );

    // list_elements_at still returns data even for null lists.
    let null_list_data = listview.list_elements_at(1);
    assert_eq!(null_list_data.len(), 2);
    assert_eq!(null_list_data.scalar_at(0), 3i32.into());
    assert_eq!(null_list_data.scalar_at(1), 4i32.into());
}

// Parameterized tests for different null patterns.
#[rstest]
#[case::all_nulls(Validity::AllInvalid, vec![false, false, false])]
#[case::all_valid(Validity::AllValid, vec![true, true, true])]
#[case::mixed(Validity::from_iter([false, true, false]), vec![false, true, false])]
fn test_nullable_patterns(#[case] validity: Validity, #[case] expected_validity: Vec<bool>) {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity).unwrap();

    for (i, &expected) in expected_validity.iter().enumerate() {
        assert_eq!(listview.is_valid(i), expected);
    }
}

#[test]
fn test_nullable_elements() {
    // Test with nullable elements inside the lists.
    let elements =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
            .into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::AllValid).unwrap();

    // First list: [Some(1), None].
    let first_list = listview.list_elements_at(0);
    assert_eq!(first_list.len(), 2);
    assert!(!first_list.scalar_at(0).is_null());
    assert_eq!(first_list.scalar_at(0), 1i32.into());
    assert!(first_list.scalar_at(1).is_null());

    // Second list: [Some(3), None].
    let second_list = listview.list_elements_at(1);
    assert!(!second_list.scalar_at(0).is_null());
    assert_eq!(second_list.scalar_at(0), 3i32.into());
    assert!(second_list.scalar_at(1).is_null());

    // Third list: [Some(5), Some(6)].
    let third_list = listview.list_elements_at(2);
    assert!(!third_list.scalar_at(0).is_null());
    assert_eq!(third_list.scalar_at(0), 5i32.into());
    assert!(!third_list.scalar_at(1).is_null());
    assert_eq!(third_list.scalar_at(1), 6i32.into());

    // Check dtype of elements.
    assert!(matches!(
        listview.elements().dtype(),
        DType::Primitive(PType::I32, Nullability::Nullable)
    ));
}

#[test]
fn test_validity_length_mismatch() {
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0i32, 2].into_array();
    let sizes = buffer![2i32, 2].into_array();
    // Wrong length validity.
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());

    let result = ListViewArray::try_new(elements, offsets, sizes, validity);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("validity") && err.to_string().contains("size"),
        "Unexpected error: {err}"
    );
}
