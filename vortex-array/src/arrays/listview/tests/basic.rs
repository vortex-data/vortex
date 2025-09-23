// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ListArray, ListViewArray, list_view_from_list};
use crate::validity::Validity;
use crate::{Array, IntoArray};

#[test]
fn test_basic_listview() {
    // Verifies basic functionality of ListView arrays including length, dtype validation,
    // and element access for a simple list of lists: [[1,2,3], [4,5], [6,7,8,9]].
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
fn test_scalar_at() {
    // Tests the scalar_at method which returns entire lists as Scalar values,
    // verifying that lists are correctly converted to the appropriate scalar representation.
    let elements = buffer![10i32, 20, 30, 40, 50].into_array();
    let offsets = buffer![0i32, 2].into_array();
    let sizes = buffer![2i32, 3].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    // First list: [10, 20].
    let first = listview.scalar_at(0);
    assert_eq!(
        first,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![10i32.into(), 20i32.into()],
            Nullability::NonNullable,
        )
    );

    // Second list: [30, 40, 50].
    let second = listview.scalar_at(1);
    assert_eq!(
        second,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![30i32.into(), 40i32.into(), 50i32.into()],
            Nullability::NonNullable,
        )
    );
}

#[test]
fn test_out_of_order_offsets() {
    // Demonstrates that ListView supports non-sequential and overlapping offsets,
    // allowing multiple lists to reference the same underlying elements.
    // Creates lists that share elements: [[1,2], [2,3], [1,2,3]].
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0i32, 1, 0].into_array();
    let sizes = buffer![2i32, 2, 3].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 3);

    // First list: [1, 2].
    let first = listview.list_elements_at(0);
    assert_eq!(first.scalar_at(0), 1i32.into());
    assert_eq!(first.scalar_at(1), 2i32.into());

    // Second list: [2, 3].
    let second = listview.list_elements_at(1);
    assert_eq!(second.scalar_at(0), 2i32.into());
    assert_eq!(second.scalar_at(1), 3i32.into());

    // Third list: [1, 2, 3].
    let third = listview.list_elements_at(2);
    assert_eq!(third.scalar_at(0), 1i32.into());
    assert_eq!(third.scalar_at(1), 2i32.into());
    assert_eq!(third.scalar_at(2), 3i32.into());
}

#[test]
fn test_empty_listview() {
    // Verifies that an empty ListView with zero lists can be created and behaves
    // correctly, even when the underlying elements array contains data.
    let elements = buffer![1i32].into_array(); // Elements exist but no lists.
    let offsets = buffer![0i32; 0].into_array();
    let sizes = buffer![0i32; 0].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 0);
    assert!(listview.is_empty());
}

#[test]
fn test_from_list_array() {
    // Verifies conversion from a ListArray (offset-based) to ListView (offset+size based),
    // including handling of empty lists in the middle of the array.
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0i32, 2, 2, 6].into_array(); // [[1,2], [], [3,4,5,6]].

    let list_array = ListArray::new(elements.into_array(), offsets, Validity::NonNullable);
    assert_eq!(list_array.len(), 3);

    // Convert to ListView.
    let listview = list_view_from_list(list_array);

    assert_eq!(listview.len(), 3);

    // First list: [1, 2].
    let first = listview.list_elements_at(0);
    assert_eq!(first.len(), 2);
    assert_eq!(first.scalar_at(0), 1i32.into());
    assert_eq!(first.scalar_at(1), 2i32.into());

    // Second list: [].
    let second = listview.list_elements_at(1);
    assert_eq!(second.len(), 0);

    // Third list: [3, 4, 5, 6].
    let third = listview.list_elements_at(2);
    assert_eq!(third.len(), 4);
    assert_eq!(third.scalar_at(0), 3i32.into());
    assert_eq!(third.scalar_at(1), 4i32.into());
    assert_eq!(third.scalar_at(2), 5i32.into());
    assert_eq!(third.scalar_at(3), 6i32.into());
}

#[test]
fn test_validation_error_offset_size_overflow() {
    // Validates that ListView construction fails when offset + size exceeds the
    // elements array bounds, ensuring data integrity through proper validation.
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![2i32, 0].into_array();
    let sizes = buffer![3i32, 1].into_array(); // offset[0] + size[0] = 5 > elements.len() = 3.

    let result =
        ListViewArray::try_new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("exceeds elements length"));
}

#[test]
fn test_validation_error_length_mismatch() {
    // Ensures that ListView construction fails when offsets and sizes arrays have
    // different lengths, as they must have a one-to-one correspondence.
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0i32, 1].into_array();
    let sizes = buffer![1i32, 1, 1].into_array(); // Different lengths.

    let result =
        ListViewArray::try_new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("same length"));
}

#[test]
fn test_validation_error_empty_arrays() {
    // Confirms that empty offsets and sizes arrays are valid when creating a ListView
    // with zero lists, distinguishing this case from invalid empty arrays.
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0i32; 0].into_array();
    let sizes = buffer![0i32; 0].into_array();

    let result =
        ListViewArray::try_new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    // Empty arrays should be allowed for 0 lists.
    assert!(result.is_ok());
    let listview = result.unwrap();
    assert_eq!(listview.len(), 0);
    assert!(listview.is_empty());
}

#[test]
fn test_listview_with_constant_sizes() {
    // Tests the slow path of size_at when sizes is a ConstantArray instead of PrimitiveArray.
    // This forces the code to use scalar_at instead of direct primitive array access.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();

    // Create regular offsets array.
    let offsets = buffer![0i32, 3, 6].into_array();

    // Create constant sizes array - all lists have size 3.
    let sizes = ConstantArray::new(3i32, 3).into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 3);

    // Verify that size_at returns the correct constant value through the slow path.
    assert_eq!(listview.size_at(0), 3);
    assert_eq!(listview.size_at(1), 3);
    assert_eq!(listview.size_at(2), 3);

    // Verify offset_at still works correctly with primitive array.
    assert_eq!(listview.offset_at(0), 0);
    assert_eq!(listview.offset_at(1), 3);
    assert_eq!(listview.offset_at(2), 6);

    // Verify list contents work correctly.
    let first_list = listview.list_elements_at(0);
    assert_eq!(first_list.len(), 3);
    assert_eq!(first_list.scalar_at(0), 1i32.into());
    assert_eq!(first_list.scalar_at(1), 2i32.into());
    assert_eq!(first_list.scalar_at(2), 3i32.into());

    let second_list = listview.list_elements_at(1);
    assert_eq!(second_list.len(), 3);
    assert_eq!(second_list.scalar_at(0), 4i32.into());
    assert_eq!(second_list.scalar_at(1), 5i32.into());
    assert_eq!(second_list.scalar_at(2), 6i32.into());

    let third_list = listview.list_elements_at(2);
    assert_eq!(third_list.len(), 3);
    assert_eq!(third_list.scalar_at(0), 7i32.into());
    assert_eq!(third_list.scalar_at(1), 8i32.into());
    assert_eq!(third_list.scalar_at(2), 9i32.into());
}

#[test]
fn test_listview_with_constant_sizes_varied_offsets() {
    // Tests the slow path with constant sizes but non-sequential offsets.
    // This verifies that the slow path handles out-of-order and overlapping lists correctly.
    let elements = buffer![10i32, 20, 30, 40, 50, 60, 70, 80].into_array();

    // Create out-of-order offsets: lists start at 2, 0, 5.
    let offsets = buffer![2i32, 0, 5].into_array();

    // All lists have constant size of 2.
    let sizes = ConstantArray::new(2i32, 3).into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 3);

    // Verify size_at through the slow path.
    assert_eq!(listview.size_at(0), 2);
    assert_eq!(listview.size_at(1), 2);
    assert_eq!(listview.size_at(2), 2);

    // Verify offset_at.
    assert_eq!(listview.offset_at(0), 2);
    assert_eq!(listview.offset_at(1), 0);
    assert_eq!(listview.offset_at(2), 5);

    // First list: elements[2..4] = [30, 40].
    let first_list = listview.list_elements_at(0);
    assert_eq!(first_list.len(), 2);
    assert_eq!(first_list.scalar_at(0), 30i32.into());
    assert_eq!(first_list.scalar_at(1), 40i32.into());

    // Second list: elements[0..2] = [10, 20].
    let second_list = listview.list_elements_at(1);
    assert_eq!(second_list.len(), 2);
    assert_eq!(second_list.scalar_at(0), 10i32.into());
    assert_eq!(second_list.scalar_at(1), 20i32.into());

    // Third list: elements[5..7] = [60, 70].
    let third_list = listview.list_elements_at(2);
    assert_eq!(third_list.len(), 2);
    assert_eq!(third_list.scalar_at(0), 60i32.into());
    assert_eq!(third_list.scalar_at(1), 70i32.into());

    // Test scalar_at to ensure it returns the correct list scalars.
    let first_scalar = listview.scalar_at(0);
    assert_eq!(
        first_scalar,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![30i32.into(), 40i32.into()],
            Nullability::NonNullable,
        )
    );

    let second_scalar = listview.scalar_at(1);
    assert_eq!(
        second_scalar,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![10i32.into(), 20i32.into()],
            Nullability::NonNullable,
        )
    );
}

#[test]
#[allow(clippy::cognitive_complexity)]
fn test_listview_all_zero_offsets_constant() {
    // Tests where all lists start at offset 0, creating overlapping lists with shared prefixes.
    // Uses ConstantArray for offsets to test the slow path.
    let elements = buffer![100i32, 200, 300, 400, 500].into_array();

    // All lists start at offset 0.
    let offsets = ConstantArray::new(0i32, 4).into_array();

    // Different sizes: 1, 2, 3, 5.
    let sizes = buffer![1i32, 2, 3, 5].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    assert_eq!(listview.len(), 4);

    // Verify offset_at through the slow path (ConstantArray).
    assert_eq!(listview.offset_at(0), 0);
    assert_eq!(listview.offset_at(1), 0);
    assert_eq!(listview.offset_at(2), 0);
    assert_eq!(listview.offset_at(3), 0);

    // Verify sizes.
    assert_eq!(listview.size_at(0), 1);
    assert_eq!(listview.size_at(1), 2);
    assert_eq!(listview.size_at(2), 3);
    assert_eq!(listview.size_at(3), 5);

    // First list: elements[0..1] = [100].
    let first_list = listview.list_elements_at(0);
    assert_eq!(first_list.len(), 1);
    assert_eq!(first_list.scalar_at(0), 100i32.into());

    // Second list: elements[0..2] = [100, 200].
    let second_list = listview.list_elements_at(1);
    assert_eq!(second_list.len(), 2);
    assert_eq!(second_list.scalar_at(0), 100i32.into());
    assert_eq!(second_list.scalar_at(1), 200i32.into());

    // Third list: elements[0..3] = [100, 200, 300].
    let third_list = listview.list_elements_at(2);
    assert_eq!(third_list.len(), 3);
    assert_eq!(third_list.scalar_at(0), 100i32.into());
    assert_eq!(third_list.scalar_at(1), 200i32.into());
    assert_eq!(third_list.scalar_at(2), 300i32.into());

    // Fourth list: elements[0..5] = [100, 200, 300, 400, 500].
    let fourth_list = listview.list_elements_at(3);
    assert_eq!(fourth_list.len(), 5);
    assert_eq!(fourth_list.scalar_at(0), 100i32.into());
    assert_eq!(fourth_list.scalar_at(1), 200i32.into());
    assert_eq!(fourth_list.scalar_at(2), 300i32.into());
    assert_eq!(fourth_list.scalar_at(3), 400i32.into());
    assert_eq!(fourth_list.scalar_at(4), 500i32.into());

    // Verify scalar_at returns the correct list scalars.
    let first_scalar = listview.scalar_at(0);
    assert_eq!(
        first_scalar,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![100i32.into()],
            Nullability::NonNullable,
        )
    );

    let third_scalar = listview.scalar_at(2);
    assert_eq!(
        third_scalar,
        Scalar::list(
            Arc::new(PType::I32.into()),
            vec![100i32.into(), 200i32.into(), 300i32.into()],
            Nullability::NonNullable,
        )
    );
}
