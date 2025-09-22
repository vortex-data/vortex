// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::vortex_panic;

use crate::arrays::{BoolArray, ListViewArray, ListViewVTable, PrimitiveArray};
use crate::compute::take;
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray};

// Helper trait to extract ListViewArray from ArrayRef.
trait ToListView {
    fn to_listview(&self) -> ListViewArray;
}

impl ToListView for ArrayRef {
    fn to_listview(&self) -> ListViewArray {
        self.as_opt::<ListViewVTable>()
            .unwrap_or_else(|| vortex_panic!("Expected ListViewArray"))
            .clone()
    }
}

#[test]
fn test_take_simple() {
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![0u32, 3, 5].into_array();
    let sizes = buffer![3u32, 2, 3].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let indices = buffer![1, 0, 2].into_array();
    let result = take(&listview, &indices).unwrap();

    assert_eq!(result.len(), 3);
    let result_list = result.to_listview();

    // First taken list should be [3, 4] (originally index 1).
    assert_eq!(result_list.size_at(0), 2);
    let list0 = result_list.list_elements_at(0);
    assert_eq!(list0.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 3);
    assert_eq!(list0.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 4);

    // Second taken list should be [0, 1, 2] (originally index 0).
    assert_eq!(result_list.size_at(1), 3);
    let list1 = result_list.list_elements_at(1);
    assert_eq!(list1.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 0);
    assert_eq!(list1.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 1);
    assert_eq!(list1.scalar_at(2).as_primitive().as_::<i32>().unwrap(), 2);

    // Third taken list should be [5, 6, 7] (originally index 2).
    assert_eq!(result_list.size_at(2), 3);
    let list2 = result_list.list_elements_at(2);
    assert_eq!(list2.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 5);
    assert_eq!(list2.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 6);
    assert_eq!(list2.scalar_at(2).as_primitive().as_::<i32>().unwrap(), 7);
}

#[test]
fn test_take_nullable() {
    let elements = buffer![10i32, 20, 30, 40, 50].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 1].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false]).into_array());

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    let indices = PrimitiveArray::from_option_iter(vec![Some(0), None, Some(1)]).to_array();
    let result = take(&listview, &indices).unwrap();

    assert_eq!(result.len(), 3);
    let result_list = result.to_listview();

    // First result should be valid [10, 20].
    assert!(result_list.is_valid(0));
    assert_eq!(result_list.size_at(0), 2);
    let list0 = result_list.list_elements_at(0);
    assert_eq!(list0.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 10);
    assert_eq!(list0.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 20);

    // Second result should be null (null index).
    assert!(result_list.is_invalid(1));

    // Third result should be null (original was null).
    assert!(result_list.is_invalid(2));
}

#[test]
fn test_take_out_of_order_offsets() {
    // Test with out-of-order offsets (a key feature of ListView).
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![5u32, 2, 8, 0].into_array(); // Out of order!
    let sizes = buffer![3u32, 2, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let indices = buffer![3, 1, 0, 2].into_array();
    let result = take(&listview, &indices).unwrap();

    assert_eq!(result.len(), 4);
    let result_list = result.to_listview();

    // Check that the taken lists maintain their content despite out-of-order offsets.
    // Index 3 (offset 0, size 2): [0, 1].
    assert_eq!(result_list.size_at(0), 2);
    let list0 = result_list.list_elements_at(0);
    assert_eq!(list0.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 0);
    assert_eq!(list0.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 1);

    // Index 1 (offset 2, size 2): [2, 3].
    assert_eq!(result_list.size_at(1), 2);
    let list1 = result_list.list_elements_at(1);
    assert_eq!(list1.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 2);
    assert_eq!(list1.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 3);

    // Index 0 (offset 5, size 3): [5, 6, 7].
    assert_eq!(result_list.size_at(2), 3);
    let list2 = result_list.list_elements_at(2);
    assert_eq!(list2.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 5);
    assert_eq!(list2.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 6);
    assert_eq!(list2.scalar_at(2).as_primitive().as_::<i32>().unwrap(), 7);

    // Index 2 (offset 8, size 2): [8, 9].
    assert_eq!(result_list.size_at(3), 2);
    let list3 = result_list.list_elements_at(3);
    assert_eq!(list3.scalar_at(0).as_primitive().as_::<i32>().unwrap(), 8);
    assert_eq!(list3.scalar_at(1).as_primitive().as_::<i32>().unwrap(), 9);
}

#[rstest]
#[case::empty_lists(vec![0u32, 2, 3, 1], vec![0, 0, 2, 1])]
#[case::duplicates(vec![0u32, 0, 1, 1, 0], vec![3, 3, 3, 3, 3])]
#[case::single(vec![1u32], vec![2])]
fn test_take_special_cases(#[case] indices: Vec<u32>, #[case] expected_sizes: Vec<usize>) {
    let (elements, offsets, sizes) = match indices.len() {
        1 => {
            // Single index case.
            (
                buffer![10i64, 20, 30, 40, 50, 60].into_array(),
                buffer![0u32, 2, 4].into_array(),
                buffer![2u32, 2, 2].into_array(),
            )
        }
        4 => {
            // Empty lists case.
            (
                buffer![42i32, 43, 44].into_array(),
                buffer![0u32, 0, 1, 1].into_array(),
                buffer![0u32, 1, 0, 2].into_array(),
            )
        }
        5 => {
            // Duplicates case.
            (
                buffer![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0].into_array(),
                buffer![0u32, 3].into_array(),
                buffer![3u32, 3].into_array(),
            )
        }
        _ => panic!("Unexpected test case"),
    };

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let indices_array = if indices.len() == 1 {
        buffer![1u32].into_array() // Special case for single index.
    } else {
        PrimitiveArray::from_iter(indices).to_array()
    };

    let result = take(&listview, &indices_array).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), expected_sizes.len());

    for (i, expected_size) in expected_sizes.iter().enumerate() {
        assert_eq!(result_list.size_at(i), *expected_size);
    }
}

#[rstest]
#[case::overlapping(vec![3u32, 0, 1, 2], vec![6, 3, 3, 3])]
#[case::reversed(vec![2u32, 1, 0], vec![2, 2, 1])]
fn test_take_with_special_offsets(#[case] indices: Vec<u32>, #[case] expected_sizes: Vec<usize>) {
    let is_overlapping = indices.len() == 4;

    let (elements, offsets, sizes) = if is_overlapping {
        // Overlapping case.
        (
            buffer![0i32, 1, 2, 3, 4, 5].into_array(),
            buffer![0u32, 1, 2, 0].into_array(),
            buffer![3u32, 3, 3, 6].into_array(),
        )
    } else {
        // Reversed offsets case.
        (
            buffer![10i32, 20, 30, 40, 50].into_array(),
            buffer![4u32, 2, 0].into_array(),
            buffer![1u32, 2, 2].into_array(),
        )
    };

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let indices = PrimitiveArray::from_iter(indices).to_array();
    let result = take(&listview, &indices).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), expected_sizes.len());

    for (i, expected_size) in expected_sizes.iter().enumerate() {
        assert_eq!(result_list.size_at(i), *expected_size);
    }
}

#[test]
fn test_take_with_large_indices() {
    // Test with a larger dataset.
    let elements = buffer![0i32..100].into_array();

    // Create 20 lists with varying offsets and sizes.
    let offsets = PrimitiveArray::from_iter((0..20).map(|i| i * 3)).into_array();
    let sizes = PrimitiveArray::from_iter((0..20).map(|i| (i % 4) + 1)).into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Take a subset of indices.
    let indices = buffer![19u32, 0, 10, 5, 15].into_array();
    let result = take(&listview, &indices).unwrap();

    assert_eq!(result.len(), 5);
    let result_list = result.to_listview();

    // Verify the sizes match expected pattern.
    assert_eq!(result_list.size_at(0), 4); // 19 % 4 + 1 = 4
    assert_eq!(result_list.size_at(1), 1); // 0 % 4 + 1 = 1
    assert_eq!(result_list.size_at(2), 3); // 10 % 4 + 1 = 3
    assert_eq!(result_list.size_at(3), 2); // 5 % 4 + 1 = 2
    assert_eq!(result_list.size_at(4), 4); // 15 % 4 + 1 = 4
}

#[test]
fn test_take_mixed_null_indices() {
    // Test with a mix of null and non-null indices.
    let elements = buffer![100i32, 200, 300, 400, 500, 600].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let indices = PrimitiveArray::from_option_iter(vec![
        Some(1),
        None,
        Some(2),
        Some(0),
        None,
        None,
        Some(1),
    ])
    .to_array();
    let result = take(&listview, &indices).unwrap();

    assert_eq!(result.len(), 7);
    let result_list = result.to_listview();

    assert!(result_list.is_valid(0)); // Index 1
    assert!(result_list.is_invalid(1)); // Null index
    assert!(result_list.is_valid(2)); // Index 2
    assert!(result_list.is_valid(3)); // Index 0
    assert!(result_list.is_invalid(4)); // Null index
    assert!(result_list.is_invalid(5)); // Null index
    assert!(result_list.is_valid(6)); // Index 1
}
