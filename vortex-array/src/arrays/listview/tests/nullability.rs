// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::buffer;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

#[test]
fn test_nullable_listview_comprehensive() {
    // Comprehensive test for nullable ListView including scalar_at with nulls.
    // Logical lists: [[1,2], null, [5,6]]
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();
    let validity = Validity::from_iter([true, false, true]);

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, validity)
            .with_zero_copy_to_list(true)
    };

    assert_eq!(listview.len(), 3);

    // Check validity.
    assert!(
        listview
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert!(
        listview
            .is_invalid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert!(
        listview
            .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );

    // Check dtype reflects nullability.
    assert!(matches!(
        listview.dtype(),
        DType::List(_, Nullability::Nullable)
    ));

    // Test scalar_at with nulls.
    let first = listview
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(!first.is_null());
    assert_eq!(
        first,
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1i32.into(), 2i32.into()],
            Nullability::Nullable,
        )
    );

    let second = listview
        .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(second.is_null());

    let third = listview
        .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
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
    let null_list_data = listview.list_elements_at(1).unwrap();
    assert_eq!(null_list_data.len(), 2);
    assert_eq!(
        null_list_data
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        3i32.into()
    );
    assert_eq!(
        null_list_data
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        4i32.into()
    );
}

// Parameterized tests for different null patterns.
#[rstest]
#[case::all_nulls(Validity::AllInvalid, vec![false, false, false])]
#[case::all_valid(Validity::AllValid, vec![true, true, true])]
#[case::mixed(Validity::from_iter([false, true, false]), vec![false, true, false])]
fn test_nullable_patterns(#[case] validity: Validity, #[case] expected_validity: Vec<bool>) {
    // Logical lists: [[1,2], [3,4], [5,6]] with varying validity
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();

    let listview = unsafe { ListViewArray::new_unchecked(elements, offsets, sizes, validity) };

    for (i, &expected) in expected_validity.iter().enumerate() {
        assert_eq!(
            listview
                .is_valid(i, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            expected
        );
    }
}

#[test]
fn test_nullable_elements() {
    // Test with nullable elements inside the lists.
    // Logical lists: [[Some(1), None], [Some(3), None], [Some(5), Some(6)]]
    let elements =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
            .into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
            .with_zero_copy_to_list(true)
    };

    // First list: [Some(1), None].
    let first_list = listview.list_elements_at(0).unwrap();
    assert_eq!(first_list.len(), 2);
    assert!(
        !first_list
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert_eq!(
        first_list
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        1i32.into()
    );
    assert!(
        first_list
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );

    // Second list: [Some(3), None].
    let second_list = listview.list_elements_at(1).unwrap();
    assert!(
        !second_list
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert_eq!(
        second_list
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        3i32.into()
    );
    assert!(
        second_list
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );

    // Third list: [Some(5), Some(6)].
    let third_list = listview.list_elements_at(2).unwrap();
    assert!(
        !third_list
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert_eq!(
        third_list
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        5i32.into()
    );
    assert!(
        !third_list
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert_eq!(
        third_list
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        6i32.into()
    );

    // Check dtype of elements.
    assert!(matches!(
        listview.elements().dtype(),
        DType::Primitive(PType::I32, Nullability::Nullable)
    ));
}

#[test]
fn test_validity_length_mismatch() {
    // Logical lists (invalid due to validity length mismatch): [[1,2], [3,4]]
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
