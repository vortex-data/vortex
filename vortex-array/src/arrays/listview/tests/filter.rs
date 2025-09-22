// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_mask::Mask;

use super::ToListView;
use crate::arrays::{BoolArray, ConstantArray, ListViewArray, PrimitiveArray};
use crate::compute::filter;
use crate::validity::Validity;
use crate::{IntoArray, ToCanonical};

#[test]
fn test_filter_comprehensive() {
    // This comprehensive test covers: basic filter, out-of-order offsets,
    // overlapping lists, various selection patterns, and element preservation.
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();

    // Out-of-order offsets with overlapping: [5,6,7], [2,3], [8,9], [0,1], [1,2,3,4]
    let offsets = buffer![5u32, 2, 8, 0, 1].into_array();
    let sizes = buffer![3u32, 2, 2, 2, 4].into_array();

    let listview = ListViewArray::try_new(elements.clone(), offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Test various selection patterns.
    // Pattern 1: Keep first and last.
    let mask1 = Mask::from_iter([true, false, false, false, true]);
    let result1 = filter(&listview, &mask1).unwrap();
    let result1_list = result1.to_listview();

    assert_eq!(result1_list.len(), 2);
    assert_eq!(result1_list.offset_at(0), 5); // List 0
    assert_eq!(result1_list.offset_at(1), 1); // List 4
    assert_eq!(result1_list.size_at(0), 3);
    assert_eq!(result1_list.size_at(1), 4);

    // Verify elements are preserved.
    let result1_elements = result1_list.elements().to_primitive();
    assert_eq!(
        result1_elements.as_slice::<i32>(),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );

    // Pattern 2: Alternating selection.
    let mask2 = Mask::from_iter([true, false, true, false, true]);
    let result2 = filter(&listview, &mask2).unwrap();
    let result2_list = result2.to_listview();

    assert_eq!(result2_list.len(), 3);
    assert_eq!(result2_list.offset_at(0), 5); // List 0
    assert_eq!(result2_list.offset_at(1), 8); // List 2
    assert_eq!(result2_list.offset_at(2), 1); // List 4

    // Pattern 3: Keep all.
    let mask_all = Mask::from_iter([true, true, true, true, true]);
    let result_all = filter(&listview, &mask_all).unwrap();
    let result_all_list = result_all.to_listview();

    assert_eq!(result_all_list.len(), 5);
    // All offsets should be preserved.
    assert_eq!(result_all_list.offset_at(0), 5);
    assert_eq!(result_all_list.offset_at(1), 2);
    assert_eq!(result_all_list.offset_at(2), 8);
    assert_eq!(result_all_list.offset_at(3), 0);
    assert_eq!(result_all_list.offset_at(4), 1);

    // Pattern 4: Keep none.
    let mask_none = Mask::from_iter([false, false, false, false, false]);
    let result_none = filter(&listview, &mask_none).unwrap();
    assert_eq!(result_none.len(), 0);
}

#[test]
fn test_filter_with_nullability() {
    // Test filter with nullable ListView.
    let elements = buffer![10i32, 20, 30, 40, 50].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 1].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    // Filter keeps all lists (including null).
    let mask = Mask::from_iter([true, true, true]);
    let result = filter(&listview, &mask).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 3);
    assert!(result_list.is_valid(0)); // First list is valid
    assert!(result_list.is_invalid(1)); // Second list is null
    assert!(result_list.is_valid(2)); // Third list is valid

    // Filter keeps only the null list.
    let mask2 = Mask::from_iter([false, true, false]);
    let result2 = filter(&listview, &mask2).unwrap();
    let result2_list = result2.to_listview();

    assert_eq!(result2_list.len(), 1);
    assert!(result2_list.is_invalid(0)); // The kept list is null
}

#[test]
fn test_filter_empty_edge_cases() {
    // Test empty lists and empty filter results.

    // Case 1: ListView with empty lists.
    let elements = buffer![99i32].into_array(); // Dummy element
    let offsets = buffer![0u32, 0, 0, 0].into_array();
    let sizes = buffer![0u32, 0, 0, 0].into_array();

    let empty_lists = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter([true, false, true, false]);
    let result = filter(&empty_lists, &mask).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 2);
    assert_eq!(result_list.size_at(0), 0);
    assert_eq!(result_list.size_at(1), 0);

    // Case 2: Mixed empty and non-empty lists.
    let elements2 = buffer![1i32, 2, 3].into_array();
    let offsets2 = buffer![0u32, 0, 1, 1].into_array();
    let sizes2 = buffer![0u32, 1, 0, 2].into_array();

    let mixed_lists = ListViewArray::try_new(elements2, offsets2, sizes2, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask2 = Mask::from_iter([true, false, true, true]);
    let result2 = filter(&mixed_lists, &mask2).unwrap();
    let result2_list = result2.to_listview();

    assert_eq!(result2_list.len(), 3);
    assert_eq!(result2_list.size_at(0), 0); // Empty list
    assert_eq!(result2_list.size_at(1), 0); // Empty list
    assert_eq!(result2_list.size_at(2), 2); // Non-empty list

    // Case 3: All-null ListView.
    let elements3 = buffer![1i32, 2, 3, 4].into_array();
    let offsets3 = buffer![0u32, 2].into_array();
    let sizes3 = buffer![2u32, 2].into_array();
    let all_null_validity = Validity::from_iter([false, false]);

    let all_null_list = ListViewArray::try_new(elements3, offsets3, sizes3, all_null_validity)
        .unwrap()
        .to_array();

    let mask3 = Mask::from_iter([true, true]);
    let null_result = filter(&all_null_list, &mask3).unwrap();
    let null_result_list = null_result.to_listview();

    assert_eq!(null_result_list.len(), 2);
    assert!(null_result_list.is_invalid(0));
    assert!(null_result_list.is_invalid(1));
}

#[test]
fn test_filter_overlapping_and_gaps() {
    // Test filtering with overlapping lists and gaps to demonstrate element preservation.

    // Elements with gaps (999 values are "gaps" between used ranges).
    let elements = buffer![1i32, 2, 3, 999, 999, 999, 7, 8, 9, 999, 11, 12].into_array();

    // Lists with overlaps and pointing to non-contiguous ranges:
    // List 0: [1,2,3] at offset 0
    // List 1: [7,8,9] at offset 6  (gap before)
    // List 2: [11,12] at offset 10 (gap before)
    // List 3: [2,3] at offset 1    (overlaps with list 0)
    // List 4: [8,9] at offset 7    (overlaps with list 1)
    let offsets = buffer![0u32, 6, 10, 1, 7].into_array();
    let sizes = buffer![3u32, 3, 2, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements.clone(), offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Filter to keep lists with gaps and overlaps.
    let mask = Mask::from_iter([false, true, true, true, false]);
    let result = filter(&listview, &mask).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 3);

    // Verify the entire elements array is preserved including gaps.
    let result_elements = result_list.elements().to_primitive();
    assert_eq!(
        result_elements.as_slice::<i32>(),
        &[1, 2, 3, 999, 999, 999, 7, 8, 9, 999, 11, 12]
    );

    // Verify offsets are unchanged.
    assert_eq!(result_list.offset_at(0), 6); // List 1: [7,8,9]
    assert_eq!(result_list.offset_at(1), 10); // List 2: [11,12]
    assert_eq!(result_list.offset_at(2), 1); // List 3: [2,3] (overlapping)

    // Verify the lists still read correctly.
    let list0 = result_list.list_elements_at(0);
    assert_eq!(list0.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 7);
    assert_eq!(list0.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 8);
    assert_eq!(list0.scalar_at(2).as_primitive().as_::<i32>().unwrap(), 9);

    let list1 = result_list.list_elements_at(1);
    assert_eq!(list1.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 11);
    assert_eq!(list1.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 12);
}

#[test]
fn test_filter_constant_arrays() {
    // Test filter with ConstantArray for offsets and sizes.

    let elements = buffer![100i32, 200, 300, 400, 500, 600, 700, 800].into_array();

    // Case 1: Constant offsets (all lists start at same position).
    let constant_offsets = ConstantArray::new(2u32, 4).into_array();
    let varying_sizes = buffer![1u32, 2, 3, 4].into_array();

    let const_offset_list = ListViewArray::try_new(
        elements.clone(),
        constant_offsets,
        varying_sizes,
        Validity::NonNullable,
    )
    .unwrap()
    .to_array();

    let mask1 = Mask::from_iter([true, false, true, false]);
    let result1 = filter(&const_offset_list, &mask1).unwrap();
    let result1_list = result1.to_listview();

    assert_eq!(result1_list.len(), 2);
    assert_eq!(result1_list.offset_at(0), 2); // Both offsets are 2
    assert_eq!(result1_list.offset_at(1), 2);
    assert_eq!(result1_list.size_at(0), 1); // Sizes: 1, 3
    assert_eq!(result1_list.size_at(1), 3);

    // Case 2: Constant sizes (all lists have same size).
    let varying_offsets = buffer![0u32, 3, 1, 5].into_array();
    let constant_sizes = ConstantArray::new(2u32, 4).into_array();

    let const_size_list = ListViewArray::try_new(
        elements.clone(),
        varying_offsets,
        constant_sizes,
        Validity::NonNullable,
    )
    .unwrap()
    .to_array();

    let mask2 = Mask::from_iter([false, true, false, true]);
    let result2 = filter(&const_size_list, &mask2).unwrap();
    let result2_list = result2.to_listview();

    assert_eq!(result2_list.len(), 2);
    assert_eq!(result2_list.offset_at(0), 3); // Offsets: 3, 5
    assert_eq!(result2_list.offset_at(1), 5);
    assert_eq!(result2_list.size_at(0), 2); // Both sizes are 2
    assert_eq!(result2_list.size_at(1), 2);

    // Case 3: Both constant (all lists are identical).
    let both_constant_offsets = ConstantArray::new(1u32, 3).into_array();
    let both_constant_sizes = ConstantArray::new(3u32, 3).into_array();

    let both_const_list = ListViewArray::try_new(
        elements,
        both_constant_offsets,
        both_constant_sizes,
        Validity::NonNullable,
    )
    .unwrap()
    .to_array();

    let mask3 = Mask::from_iter([true, false, true]);
    let result3 = filter(&both_const_list, &mask3).unwrap();
    let result3_list = result3.to_listview();

    assert_eq!(result3_list.len(), 2);
    assert_eq!(result3_list.offset_at(0), 1);
    assert_eq!(result3_list.offset_at(1), 1);
    assert_eq!(result3_list.size_at(0), 3);
    assert_eq!(result3_list.size_at(1), 3);
}

#[test]
fn test_filter_extreme_offsets() {
    // Test with very large offsets to demonstrate unreferenced element preservation.

    // Create a large elements array.
    let elements = PrimitiveArray::from_iter(0i32..10000).into_array();

    // Lists at extremes: beginning, middle, and end of the array.
    let offsets = buffer![0u32, 4999, 9995, 2500, 7500].into_array();
    let sizes = buffer![5u32, 2, 5, 3, 4].into_array();

    let listview = ListViewArray::try_new(elements.clone(), offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Filter to keep only 2 lists from the 5, demonstrating we keep all 10000 elements.
    let mask = Mask::from_iter([false, true, false, false, true]);
    let result = filter(&listview, &mask).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 2);

    // Verify offsets are preserved.
    assert_eq!(result_list.offset_at(0), 4999);
    assert_eq!(result_list.offset_at(1), 7500);

    // Verify the entire elements array is preserved.
    assert_eq!(result_list.elements().len(), 10000);

    // Verify we can still read the correct values.
    let list0 = result_list.list_elements_at(0);
    assert_eq!(
        list0.scalar_at(0).as_primitive().as_::<i32>().unwrap(),
        4999
    );
    assert_eq!(
        list0.scalar_at(1).as_primitive().as_::<i32>().unwrap(),
        5000
    );

    let list1 = result_list.list_elements_at(1);
    assert_eq!(
        list1.scalar_at(0).as_primitive().as_::<i32>().unwrap(),
        7500
    );
    assert_eq!(
        list1.scalar_at(1).as_primitive().as_::<i32>().unwrap(),
        7501
    );

    // Test sparse selection from large dataset.
    let sparse_mask = Mask::from_iter((0..5).map(|i| i == 0 || i == 4));
    let sparse_result = filter(&listview, &sparse_mask).unwrap();
    let sparse_list = sparse_result.to_listview();

    assert_eq!(sparse_list.len(), 2);
    assert_eq!(sparse_list.offset_at(0), 0); // First list
    assert_eq!(sparse_list.offset_at(1), 7500); // Last list
    assert_eq!(sparse_list.elements().len(), 10000); // Still keeps all elements
}
