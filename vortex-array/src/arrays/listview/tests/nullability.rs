// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_scalar::Scalar;

use crate::arrays::ListViewArray;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, IntoArray};

#[test]
fn test_nullable_listview() {
    // Tests ListView's handling of nullable lists where entire lists can be null.
    // Creates an array with a null list in the middle: [[1,2], null, [3,4,5]].
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0i32, 0, 2].into_array();
    let sizes = buffer![2i32, 0, 3].into_array();
    let validity = Validity::from_iter([true, false, true]);

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, validity);

    assert_eq!(listview.len(), 3);

    // Check dtype has nullable lists.
    assert!(matches!(
        listview.dtype(),
        DType::List(_, Nullability::Nullable)
    ));

    // First list: [1, 2].
    assert!(listview.validity.is_valid(0));
    let first = listview.list_elements_at(0);
    assert_eq!(first.len(), 2);
    assert_eq!(first.scalar_at(0), 1i32.into());
    assert_eq!(first.scalar_at(1), 2i32.into());

    // Second list: null.
    assert!(!listview.validity.is_valid(1));

    // Third list: [3, 4, 5].
    assert!(listview.validity.is_valid(2));
    let third = listview.list_elements_at(2);
    assert_eq!(third.len(), 3);
    assert_eq!(third.scalar_at(0), 3i32.into());
    assert_eq!(third.scalar_at(1), 4i32.into());
    assert_eq!(third.scalar_at(2), 5i32.into());
}

#[test]
fn test_all_nulls() {
    // Verifies correct behavior when all lists in a ListView are null,
    // ensuring validity tracking works properly for completely null arrays.
    let elements = buffer![1i32].into_array(); // Some elements exist but unused.
    let offsets = buffer![0i32, 0, 0].into_array();
    let sizes = buffer![0i32, 0, 0].into_array();
    let validity = Validity::from_iter([false, false, false]);

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, validity);

    assert_eq!(listview.len(), 3);

    for i in 0..3 {
        assert!(!listview.validity.is_valid(i));
    }
}

#[test]
fn test_nullable_elements() {
    // Tests nested nullability where individual elements within lists can be null,
    // distinct from list-level nullability. Creates: [[1, null, 3], [null, 5]].
    let element_validity = Validity::from_iter([true, false, true, false, true]);
    let elements_with_validity =
        crate::arrays::PrimitiveArray::new(buffer![1i32, 0, 3, 0, 5], element_validity)
            .into_array();

    let offsets = buffer![0i32, 3].into_array();
    let sizes = buffer![3i32, 2].into_array();

    let listview = ListViewArray::new(
        elements_with_validity,
        offsets,
        sizes,
        Validity::NonNullable,
    );

    assert_eq!(listview.len(), 2);

    // First list: [1, null, 3].
    let first = listview.list_elements_at(0);
    assert_eq!(first.len(), 3);

    // Check element validity through the PrimitiveArray
    let first_prim = first
        .as_opt::<crate::arrays::PrimitiveVTable>()
        .expect("Expected PrimitiveArray");
    assert!(first_prim.validity().is_valid(0));
    assert!(!first_prim.validity().is_valid(1));
    assert!(first_prim.validity().is_valid(2));

    // Second list: [null, 5].
    let second = listview.list_elements_at(1);
    assert_eq!(second.len(), 2);

    let second_prim = second
        .as_opt::<crate::arrays::PrimitiveVTable>()
        .expect("Expected PrimitiveArray");
    assert!(!second_prim.validity().is_valid(0));
    assert!(second_prim.validity().is_valid(1));
}

#[test]
fn test_scalar_at_with_nulls() {
    // Verifies that scalar_at correctly returns null scalars for null lists
    // and properly formed list scalars for valid lists in a nullable ListView.
    let elements = buffer![10i32, 20, 30].into_array();
    let offsets = buffer![0i32, 0, 2].into_array();
    let sizes = buffer![2i32, 0, 1].into_array();
    let validity = Validity::from_iter([true, false, true]);

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, validity);

    // First list: [10, 20].
    let first = listview.scalar_at(0);
    assert_eq!(
        first,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![10i32.into(), 20i32.into()],
            Nullability::Nullable,
        )
    );

    // Second list is null so scalar_at returns a null scalar.
    let second = listview.scalar_at(1);
    assert!(second.is_null());

    // Third list: [30].
    let third = listview.scalar_at(2);
    assert_eq!(
        third,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![30i32.into()],
            Nullability::Nullable,
        )
    );
}

#[test]
fn test_validity_length_mismatch() {
    // Ensures that ListView construction fails when the validity array length
    // doesn't match the number of lists, maintaining consistency in null tracking.
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0i32, 1].into_array();
    let sizes = buffer![1i32, 2].into_array();
    let validity = Validity::from_iter([true, false, true]); // Length 3 but array has 2 lists.

    let result = ListViewArray::try_new(elements.into_array(), offsets, sizes, validity);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("does not match"));
}
