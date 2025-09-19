// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::validity::Validity;
use crate::{Array, IntoArray};

#[test]
fn test_slice_basic() {
    // Tests basic slicing functionality to extract a contiguous subset of lists
    // from a ListView, verifying that the sliced view maintains correct data access.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
    let offsets = buffer![0i32, 2, 4, 6, 8].into_array();
    let sizes = buffer![2i32, 2, 2, 2, 2].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);
    assert_eq!(listview.len(), 5);

    // Slice to get middle 3 lists.
    let sliced = listview.slice(1..4);
    assert_eq!(sliced.len(), 3);

    let sliced_view = sliced
        .as_opt::<ListViewVTable>()
        .expect("Expected ListViewArray");

    // First list in slice (originally second): [3, 4].
    let first = sliced_view.list_elements_at(0);
    assert_eq!(first.len(), 2);
    assert_eq!(first.scalar_at(0), 3i32.into());
    assert_eq!(first.scalar_at(1), 4i32.into());

    // Second list in slice (originally third): [5, 6].
    let second = sliced_view.list_elements_at(1);
    assert_eq!(second.len(), 2);
    assert_eq!(second.scalar_at(0), 5i32.into());
    assert_eq!(second.scalar_at(1), 6i32.into());

    // Third list in slice (originally fourth): [7, 8].
    let third = sliced_view.list_elements_at(2);
    assert_eq!(third.len(), 2);
    assert_eq!(third.scalar_at(0), 7i32.into());
    assert_eq!(third.scalar_at(1), 8i32.into());
}

#[test]
fn test_slice_out_of_order() {
    // Verifies that slicing works correctly with non-sequential offsets,
    // ensuring that complex offset patterns are preserved through slice operations.
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0i32, 2, 1, 3, 0].into_array();
    let sizes = buffer![2i32, 2, 3, 2, 5].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);
    assert_eq!(listview.len(), 5);

    // Slice to get elements [1..3].
    let sliced = listview.slice(1..3);
    assert_eq!(sliced.len(), 2);

    let sliced_view = sliced
        .as_opt::<ListViewVTable>()
        .expect("Expected ListViewArray");

    // First list in slice: starts at offset 2, size 2 -> [3, 4].
    let first = sliced_view.list_elements_at(0);
    assert_eq!(first.len(), 2);
    assert_eq!(first.scalar_at(0), 3i32.into());
    assert_eq!(first.scalar_at(1), 4i32.into());

    // Second list in slice: starts at offset 1, size 3 -> [2, 3, 4].
    let second = sliced_view.list_elements_at(1);
    assert_eq!(second.len(), 3);
    assert_eq!(second.scalar_at(0), 2i32.into());
    assert_eq!(second.scalar_at(1), 3i32.into());
    assert_eq!(second.scalar_at(2), 4i32.into());
}

#[test]
fn test_slice_with_nulls() {
    // Tests that slicing correctly preserves validity information for null lists,
    // ensuring that null tracking is maintained in the sliced view.
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0i32, 2, 2, 3, 5].into_array();
    let sizes = buffer![2i32, 0, 1, 2, 1].into_array();
    let validity = Validity::from_iter([true, false, true, true, false]);

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, validity);
    assert_eq!(listview.len(), 5);

    // Slice to get middle elements [1..4].
    let sliced = listview.slice(1..4);
    assert_eq!(sliced.len(), 3);

    let sliced_view = sliced
        .as_opt::<ListViewVTable>()
        .expect("Expected ListViewArray");

    // Check validity is correctly sliced.
    assert!(!sliced_view.validity.is_valid(0)); // Was index 1, null.
    assert!(sliced_view.validity.is_valid(1)); // Was index 2, valid.
    assert!(sliced_view.validity.is_valid(2)); // Was index 3, valid.

    // Second list in slice (was third): [3].
    let second = sliced_view.list_elements_at(1);
    assert_eq!(second.len(), 1);
    assert_eq!(second.scalar_at(0), 3i32.into());

    // Third list in slice (was fourth): [4, 5].
    let third = sliced_view.list_elements_at(2);
    assert_eq!(third.len(), 2);
    assert_eq!(third.scalar_at(0), 4i32.into());
    assert_eq!(third.scalar_at(1), 5i32.into());
}

#[test]
fn test_slice_empty_range() {
    // Verifies that slicing with an empty range produces a valid empty ListView.
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0i32, 1, 2].into_array();
    let sizes = buffer![1i32, 1, 1].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    // Empty slice.
    let sliced = listview.slice(1..1);
    assert_eq!(sliced.len(), 0);
}

#[test]
fn test_slice_full_array() {
    // Tests that slicing the entire array range returns a view equivalent to
    // the original array, verifying data integrity for full-range slices.
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0i32, 2].into_array();
    let sizes = buffer![2i32, 2].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    // Slice the entire array.
    let sliced = listview.slice(0..2);
    assert_eq!(sliced.len(), 2);

    let sliced_view = sliced
        .as_opt::<ListViewVTable>()
        .expect("Expected ListViewArray");

    // Verify data is unchanged.
    let first = sliced_view.list_elements_at(0);
    assert_eq!(first.len(), 2);
    assert_eq!(first.scalar_at(0), 1i32.into());
    assert_eq!(first.scalar_at(1), 2i32.into());

    let second = sliced_view.list_elements_at(1);
    assert_eq!(second.len(), 2);
    assert_eq!(second.scalar_at(0), 3i32.into());
    assert_eq!(second.scalar_at(1), 4i32.into());
}

#[test]
fn test_slice_single_element() {
    // Verifies that slicing to extract a single list works correctly,
    // testing edge case of minimum non-empty slice size.
    let elements = buffer![10i32, 20, 30, 40, 50].into_array();
    let offsets = buffer![0i32, 2, 3].into_array();
    let sizes = buffer![2i32, 1, 2].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    // Slice to get single middle element.
    let sliced = listview.slice(1..2);
    assert_eq!(sliced.len(), 1);

    let sliced_view = sliced
        .as_opt::<ListViewVTable>()
        .expect("Expected ListViewArray");

    let single = sliced_view.list_elements_at(0);
    assert_eq!(single.len(), 1);
    assert_eq!(single.scalar_at(0), 30i32.into());
}

#[test]
#[should_panic(expected = "OutOfBounds")]
fn test_slice_out_of_bounds() {
    // Ensures that attempting to slice beyond array bounds triggers a panic,
    // validating bounds checking in debug builds.
    let elements = buffer![1i32, 2].into_array();
    let offsets = buffer![0i32, 1].into_array();
    let sizes = buffer![1i32, 1].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    // This should panic in debug mode.
    let _ = listview.slice(0..5);
}

#[test]
#[should_panic(expected = "start (2) must be <= stop (1)")]
#[allow(clippy::reversed_empty_ranges)]
fn test_slice_invalid_range() {
    // Validates that invalid slice ranges where start > end are rejected with
    // a clear error message, ensuring slice API consistency.
    let elements = buffer![1i32, 2].into_array();
    let offsets = buffer![0i32, 1].into_array();
    let sizes = buffer![1i32, 1].into_array();

    let listview = ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

    // Start > end should panic.
    let _ = listview.slice(2..1);
}
