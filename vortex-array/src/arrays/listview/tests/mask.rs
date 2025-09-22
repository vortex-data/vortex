// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_mask::Mask;

use super::ToListView;
use crate::arrays::{BoolArray, ListViewArray, PrimitiveArray};
use crate::compute::mask;
use crate::validity::Validity;
use crate::{Array, IntoArray};

#[test]
fn test_mask_simple() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
    let offsets = buffer![0u32, 2, 4, 6].into_array();
    let sizes = buffer![2u32, 2, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Mask sets elements to null where true.
    let selection = Mask::from_iter([true, false, true, true]);
    let result = mask(&listview, &selection).unwrap();

    assert_eq!(result.len(), 4); // Length is preserved.
    let result_list = result.to_listview();

    // Check validity: true in mask means null.
    assert!(!result_list.is_valid(0)); // Masked.
    assert!(result_list.is_valid(1)); // Not masked.
    assert!(!result_list.is_valid(2)); // Masked.
    assert!(!result_list.is_valid(3)); // Masked.

    // Offsets and sizes are preserved.
    assert_eq!(result_list.offset_at(0), 0);
    assert_eq!(result_list.size_at(0), 2);
    assert_eq!(result_list.offset_at(1), 2);
    assert_eq!(result_list.size_at(1), 2);
    assert_eq!(result_list.offset_at(2), 4);
    assert_eq!(result_list.size_at(2), 2);
    assert_eq!(result_list.offset_at(3), 6);
    assert_eq!(result_list.size_at(3), 2);
}

#[test]
fn test_mask_with_nulls() {
    let elements = buffer![10i32, 20, 30, 40, 50, 60].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 2].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    // Mask sets elements to null where true.
    let selection = Mask::from_iter([true, false, false]);
    let result = mask(&listview, &selection).unwrap();

    assert_eq!(result.len(), 3);
    let result_list = result.to_listview();

    // Check validity.
    assert!(!result_list.is_valid(0)); // Was valid, now masked.
    assert!(!result_list.is_valid(1)); // Was already null, stays null.
    assert!(result_list.is_valid(2)); // Was valid, not masked.
}

#[test]
fn test_mask_empty_selection() {
    let elements = buffer![1.0f64, 2.0, 3.0, 4.0].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // No indices masked (all false = all preserved).
    let selection = Mask::from_iter([false, false]);
    let result = mask(&listview, &selection).unwrap();

    assert_eq!(result.len(), 2);
    let result_list = result.to_listview();

    // All should be valid.
    assert!(result_list.is_valid(0));
    assert!(result_list.is_valid(1));

    // Offsets and sizes preserved.
    assert_eq!(result_list.offset_at(0), 0);
    assert_eq!(result_list.size_at(0), 2);
    assert_eq!(result_list.offset_at(1), 2);
    assert_eq!(result_list.size_at(1), 2);
}

#[test]
fn test_mask_all_selected() {
    let elements = buffer![10i32, 20, 30, 40].into_array();
    let offsets = buffer![0u32, 1, 2].into_array();
    let sizes = buffer![1u32, 1, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Mask all indices (sets all to null).
    let selection = Mask::from_iter([true, true, true]);
    let result = mask(&listview, &selection).unwrap();

    assert_eq!(result.len(), 3);

    // When all elements are masked, the result might be a ConstantArray.
    // Just check that all elements are null.
    for i in 0..3 {
        assert!(!result.is_valid(i), "Element {} should be null", i);
    }
}

#[test]
fn test_mask_single_selection() {
    let elements = buffer![42i64, 84, 126].into_array();
    let offsets = buffer![0u32, 1, 2].into_array();
    let sizes = buffer![1u32, 1, 1].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Mask only index 1.
    let selection = Mask::from_iter([false, true, false]);
    let result = mask(&listview, &selection).unwrap();

    assert_eq!(result.len(), 3);
    let result_list = result.to_listview();

    // Check validity.
    assert!(result_list.is_valid(0)); // Not masked.
    assert!(!result_list.is_valid(1)); // Masked.
    assert!(result_list.is_valid(2)); // Not masked.

    // All offsets and sizes preserved.
    assert_eq!(result_list.offset_at(0), 0);
    assert_eq!(result_list.size_at(0), 1);
    assert_eq!(result_list.offset_at(1), 1);
    assert_eq!(result_list.size_at(1), 1);
    assert_eq!(result_list.offset_at(2), 2);
    assert_eq!(result_list.size_at(2), 1);
}

#[test]
fn test_mask_sparse_selection() {
    let elements = buffer![0i32..30].into_array();

    // Create 10 lists.
    let offsets = PrimitiveArray::from_iter((0..10).map(|i| (i * 3) as u32)).to_array();
    let sizes = PrimitiveArray::from_iter(vec![3u32; 10]).to_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Sparsely mask indices [2, 5, 8].
    let mut mask_vec = vec![false; 10];
    mask_vec[2] = true;
    mask_vec[5] = true;
    mask_vec[8] = true;
    let selection = Mask::from_iter(mask_vec);

    let result = mask(&listview, &selection).unwrap();

    assert_eq!(result.len(), 10);
    let result_list = result.to_listview();

    // Check validity - only indices 2, 5, 8 should be null.
    for i in 0..10 {
        if i == 2 || i == 5 || i == 8 {
            assert!(!result_list.is_valid(i), "Index {} should be masked", i);
        } else {
            assert!(result_list.is_valid(i), "Index {} should not be masked", i);
        }
    }

    // All lists should still have size 3.
    for i in 0..10 {
        assert_eq!(result_list.size_at(i), 3);
    }
}
