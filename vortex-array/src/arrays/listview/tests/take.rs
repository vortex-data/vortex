// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;

use super::ToListView;
use crate::arrays::{BoolArray, ConstantArray, ListViewArray, PrimitiveArray};
use crate::compute::take;
use crate::validity::Validity;
use crate::{Array, IntoArray, ToCanonical};

#[test]
fn test_take_comprehensive() {
    // This comprehensive test covers: basic take, out-of-order indices, out-of-order offsets,
    // overlapping lists, and correct element preservation.
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();

    // Out-of-order offsets with overlapping: [5,6,7], [2,3], [8,9], [0,1], [1,2,3,4]
    let offsets = buffer![5u32, 2, 8, 0, 1].into_array();
    let sizes = buffer![3u32, 2, 2, 2, 4].into_array();

    let listview = ListViewArray::try_new(elements.clone(), offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Take with out-of-order indices.
    let indices = buffer![3u32, 1, 0, 4, 2].into_array();
    let result = take(&listview, &indices).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result.len(), 5);

    // Verify offsets are preserved (not adjusted).
    assert_eq!(result_list.offset_at(0), 0); // List 3
    assert_eq!(result_list.offset_at(1), 2); // List 1
    assert_eq!(result_list.offset_at(2), 5); // List 0
    assert_eq!(result_list.offset_at(3), 1); // List 4
    assert_eq!(result_list.offset_at(4), 8); // List 2

    // Verify sizes.
    assert_eq!(result_list.size_at(0), 2); // [0,1]
    assert_eq!(result_list.size_at(1), 2); // [2,3]
    assert_eq!(result_list.size_at(2), 3); // [5,6,7]
    assert_eq!(result_list.size_at(3), 4); // [1,2,3,4]
    assert_eq!(result_list.size_at(4), 2); // [8,9]

    // Verify elements are unchanged (full array preserved).
    let result_elements = result_list.elements().to_primitive();
    assert_eq!(
        result_elements.as_slice::<i32>(),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );

    // Test with nullable indices.
    let nullable_indices =
        PrimitiveArray::from_option_iter(vec![Some(0), None, Some(2)]).to_array();
    let nullable_result = take(&listview, &nullable_indices).unwrap();
    let nullable_list = nullable_result.to_listview();

    assert_eq!(nullable_list.len(), 3);
    assert!(nullable_list.is_valid(0));
    assert!(nullable_list.is_invalid(1)); // Null index
    assert!(nullable_list.is_valid(2));
}

#[test]
fn test_take_with_nullability() {
    // Test take operation with nullable ListView and nullable indices.
    let elements = buffer![10i32, 20, 30, 40, 50].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 1].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    // Take with nullable indices: take index 1 (null list), null index, and index 2.
    let indices = PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)]).to_array();
    let result = take(&listview, &indices).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result.len(), 3);
    assert!(result_list.is_invalid(0)); // Original list was null
    assert!(result_list.is_invalid(1)); // Null index
    assert!(result_list.is_valid(2)); // Valid list
    assert_eq!(result_list.size_at(2), 1);
}

#[test]
fn test_empty_edge_cases() {
    // Test empty lists, empty takes, and all-null scenarios.

    // Case 1: ListView with empty lists.
    let elements = buffer![99i32].into_array(); // Dummy element
    let offsets = buffer![0u32, 0, 0, 0].into_array();
    let sizes = buffer![0u32, 0, 0, 0].into_array();

    let empty_lists = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let indices = buffer![1u32, 3, 0].into_array();
    let result = take(&empty_lists, &indices).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 3);
    assert_eq!(result_list.size_at(0), 0);
    assert_eq!(result_list.size_at(1), 0);
    assert_eq!(result_list.size_at(2), 0);

    // Case 2: Take with empty indices array.
    let elements2 = buffer![1i32, 2, 3].into_array();
    let offsets2 = buffer![0u32].into_array();
    let sizes2 = buffer![3u32].into_array();

    let listview2 = ListViewArray::try_new(elements2, offsets2, sizes2, Validity::NonNullable)
        .unwrap()
        .to_array();

    let empty_indices = PrimitiveArray::from_iter(Vec::<u32>::new()).to_array();
    let empty_result = take(&listview2, &empty_indices).unwrap();

    assert_eq!(empty_result.len(), 0);

    // Case 3: All-null ListView.
    let elements3 = buffer![1i32, 2, 3, 4].into_array();
    let offsets3 = buffer![0u32, 2].into_array();
    let sizes3 = buffer![2u32, 2].into_array();
    let all_null_validity = Validity::from_iter([false, false]);

    let all_null_list = ListViewArray::try_new(elements3, offsets3, sizes3, all_null_validity)
        .unwrap()
        .to_array();

    let indices3 = buffer![0u32, 1].into_array();
    let null_result = take(&all_null_list, &indices3).unwrap();
    let null_result_list = null_result.to_listview();

    assert_eq!(null_result_list.len(), 2);
    assert!(null_result_list.is_invalid(0));
    assert!(null_result_list.is_invalid(1));
}

#[test]
fn test_overlapping_and_gaps() {
    // Test ListView's unique ability to handle overlapping lists and gaps in elements.
    // This demonstrates why we keep the entire elements array.

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

    // Take lists that demonstrate gaps and overlaps are preserved.
    let indices = buffer![1u32, 3, 4, 2].into_array();
    let result = take(&listview, &indices).unwrap();
    let result_list = result.to_listview();

    // Verify the entire elements array is preserved including gaps.
    let result_elements = result_list.elements().to_primitive();
    assert_eq!(
        result_elements.as_slice::<i32>(),
        &[1, 2, 3, 999, 999, 999, 7, 8, 9, 999, 11, 12]
    );

    // Verify offsets are unchanged (demonstrating gaps are maintained).
    assert_eq!(result_list.offset_at(0), 6); // List 1: [7,8,9]
    assert_eq!(result_list.offset_at(1), 1); // List 3: [2,3] (overlapping)
    assert_eq!(result_list.offset_at(2), 7); // List 4: [8,9] (overlapping)
    assert_eq!(result_list.offset_at(3), 10); // List 2: [11,12]

    // Verify the lists still read correctly despite gaps.
    let list0 = result_list.list_elements_at(0);
    assert_eq!(list0.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 7);
    assert_eq!(list0.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 8);
    assert_eq!(list0.scalar_at(2).as_primitive().as_::<i32>().unwrap(), 9);
}

#[test]
fn test_constant_arrays() {
    // Test take with ConstantArray for offsets and sizes.
    // This tests the "slow path" and is currently missing from the test suite.

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

    let indices1 = buffer![3u32, 0, 2].into_array();
    let result1 = take(&const_offset_list, &indices1).unwrap();
    let result1_list = result1.to_listview();

    assert_eq!(result1_list.len(), 3);
    assert_eq!(result1_list.offset_at(0), 2); // All offsets are 2
    assert_eq!(result1_list.offset_at(1), 2);
    assert_eq!(result1_list.offset_at(2), 2);
    assert_eq!(result1_list.size_at(0), 4); // Sizes: 4, 1, 3
    assert_eq!(result1_list.size_at(1), 1);
    assert_eq!(result1_list.size_at(2), 3);

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

    let indices2 = buffer![1u32, 3, 0].into_array();
    let result2 = take(&const_size_list, &indices2).unwrap();
    let result2_list = result2.to_listview();

    assert_eq!(result2_list.len(), 3);
    assert_eq!(result2_list.offset_at(0), 3); // Offsets: 3, 5, 0
    assert_eq!(result2_list.offset_at(1), 5);
    assert_eq!(result2_list.offset_at(2), 0);
    assert_eq!(result2_list.size_at(0), 2); // All sizes are 2
    assert_eq!(result2_list.size_at(1), 2);
    assert_eq!(result2_list.size_at(2), 2);

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

    let indices3 = buffer![2u32, 0].into_array();
    let result3 = take(&both_const_list, &indices3).unwrap();
    let result3_list = result3.to_listview();

    assert_eq!(result3_list.len(), 2);
    assert_eq!(result3_list.offset_at(0), 1);
    assert_eq!(result3_list.offset_at(1), 1);
    assert_eq!(result3_list.size_at(0), 3);
    assert_eq!(result3_list.size_at(1), 3);
}

#[test]
fn test_extreme_offsets() {
    // Test with very large offsets and extreme patterns to demonstrate
    // that we keep unreferenced elements.

    // Create a large elements array.
    let elements = PrimitiveArray::from_iter(0i32..10000).into_array();

    // Lists at extremes: beginning, middle, and end of the array.
    let offsets = buffer![0u32, 4999, 9995, 2500, 7500].into_array();
    let sizes = buffer![5u32, 2, 5, 3, 4].into_array();

    let listview = ListViewArray::try_new(elements.clone(), offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Take only 2 lists from the 5, demonstrating we keep all 10000 elements.
    let indices = buffer![1u32, 4].into_array(); // Take middle and near-end lists
    let result = take(&listview, &indices).unwrap();
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
}
