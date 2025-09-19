// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use crate::IntoArray;
use crate::arrays::{ConstantArray, ListViewArray};
use crate::compute::is_constant;
use crate::validity::Validity;

#[test]
fn test_not_constant_different_sizes() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 1, 3].into_array();
    let sizes = buffer![1u32, 2, 2].into_array(); // Different sizes

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(false));
}

#[test]
fn test_not_constant_different_elements() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(false));
}

#[test]
fn test_constant_same_empty_lists() {
    let elements = buffer![42i32].into_array(); // Dummy element
    let offsets = buffer![0u32, 0, 0, 0].into_array();
    let sizes = buffer![0u32, 0, 0, 0].into_array(); // All empty

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(true));
}

#[test]
fn test_constant_single_list() {
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0u32].into_array();
    let sizes = buffer![3u32].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(true)); // Single list is constant
}

#[test]
fn test_constant_with_constant_elements() {
    // Create a ListView where all lists point to the same constant elements.
    let elements = ConstantArray::new(42i32, 6).to_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // All lists have the same content [42, 42].
    assert_eq!(is_constant(&listview).unwrap(), Some(true));
}

#[test]
fn test_not_constant_overlapping_different() {
    // Overlapping lists with different content.
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0u32, 1, 0].into_array(); // Overlapping
    let sizes = buffer![2u32, 2, 3].into_array(); // Different sizes

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(false));
}

#[test]
fn test_not_constant_with_nulls() {
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 2].into_array();
    let validity = Validity::from_iter([true, false]); // One null

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(false));
}

#[test]
fn test_constant_all_nulls() {
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 2].into_array();
    let validity = Validity::from_iter([false, false]); // All null

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(true)); // All nulls is constant
}

#[test]
fn test_constant_repeated_same_lists() {
    // Multiple lists pointing to the exact same elements.
    let elements = buffer![5i32, 6].into_array();
    let offsets = buffer![0u32, 0, 0, 0].into_array(); // All same offset
    let sizes = buffer![2u32, 2, 2, 2].into_array(); // All same size

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    assert_eq!(is_constant(&listview).unwrap(), Some(true));
}
