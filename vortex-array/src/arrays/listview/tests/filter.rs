// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, ListViewArray, ListViewVTable, PrimitiveArray};
use crate::compute::filter;
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

// TODO(connor)[ListView]: Once `ListViewArray` becomes the canonical encoding for `DType::List`, we
// can remove this and replace it with `to_list()`.

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

#[rstest]
#[case::keep_first(vec![true, false], 1, vec![0], vec![3])]
#[case::keep_second(vec![false, true, false, true], 2, vec![2, 0], vec![2, 2])]
fn test_filter_selection_patterns(
    #[case] mask_values: Vec<bool>,
    #[case] expected_len: usize,
    #[case] expected_offsets: Vec<usize>,
    #[case] expected_sizes: Vec<usize>,
) {
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let (offsets, sizes) = if mask_values.len() == 2 {
        (buffer![0u32, 3].into_array(), buffer![3u32, 3].into_array())
    } else {
        // Out-of-order offsets case.
        (
            buffer![5u32, 2, 8, 0].into_array(),
            buffer![3u32, 2, 2, 2].into_array(),
        )
    };

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter(mask_values);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), expected_len);
    let result_list = result.to_listview();

    for (i, (offset, size)) in expected_offsets
        .iter()
        .zip(expected_sizes.iter())
        .enumerate()
    {
        assert_eq!(result_list.offset_at(i), *offset);
        assert_eq!(result_list.size_at(i), *size);
    }
}

#[test]
fn test_filter_with_nulls() {
    let elements = buffer![10i32, 20, 30, 40, 50, 60].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 2].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false]).into_array());

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter([true, true]);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), 2);
    let result_list = result.to_listview();

    // First list should be valid.
    assert!(result_list.is_valid(0));
    assert_eq!(result_list.size_at(0), 2);

    // Second list should be null.
    assert!(result_list.is_invalid(1));
}

#[test]
fn test_filter_empty_lists() {
    let elements = buffer![42i32, 43].into_array();
    let offsets = buffer![0u32, 0, 1, 1].into_array();
    let sizes = buffer![0u32, 1, 0, 1].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter([true, false, true, false]);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), 2);
    let result_list = result.to_listview();

    // First list is empty.
    assert_eq!(result_list.size_at(0), 0);

    // Second list is also empty.
    assert_eq!(result_list.size_at(1), 0);
}

#[rstest]
#[case::all(vec![true, true], 2, vec![2, 2])]
#[case::none(vec![false, false], 0, vec![])]
fn test_filter_extreme_cases(
    #[case] mask_values: Vec<bool>,
    #[case] expected_len: usize,
    #[case] expected_sizes: Vec<usize>,
) {
    let elements = buffer![0i32, 1, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter(mask_values);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), expected_len);

    if expected_len > 0 {
        let result_list = result.to_listview();
        for (i, expected_size) in expected_sizes.iter().enumerate() {
            assert_eq!(result_list.size_at(i), *expected_size);
        }
    }
}

#[test]
fn test_filter_overlapping_lists() {
    // Test filtering with overlapping lists (unique to ListView).
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
    let offsets = buffer![0u32, 2, 1, 3].into_array(); // Overlapping offsets!
    let sizes = buffer![4u32, 3, 4, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter([true, false, true, false]);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), 2);
    let result_list = result.to_listview();

    // First list: offset 0, size 4.
    assert_eq!(result_list.offset_at(0), 0);
    assert_eq!(result_list.size_at(0), 4);

    // Second list: offset 1, size 4 (overlapping with first).
    // With our slice approach, offset is 1-0 = 1.
    assert_eq!(result_list.offset_at(1), 1);
    assert_eq!(result_list.size_at(1), 4);
}

#[test]
fn test_filter_reversed_offsets() {
    // Test filtering with completely reversed offsets.
    let elements = buffer![10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0].into_array();
    let offsets = buffer![5u32, 3, 1, 0].into_array(); // Reversed order!
    let sizes = buffer![1u32, 2, 2, 1].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter([false, true, true, false]);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), 2);
    let result_list = result.to_listview();

    // First result: list #1 with original offset 3, min_offset is 1, so 3-1=2
    assert_eq!(result_list.offset_at(0), 2);
    assert_eq!(result_list.size_at(0), 2);

    // Second result: list #2 with original offset 1, min_offset is 1, so 1-1=0
    assert_eq!(result_list.offset_at(1), 0);
    assert_eq!(result_list.size_at(1), 2);
}

#[test]
fn test_filter_large_dataset() {
    // Test with a larger dataset.
    let elements = buffer![0i32..200].into_array();

    // Create 50 lists with varying offsets and sizes.
    let offsets = PrimitiveArray::from_iter((0..50).map(|i| (i * 2) as u32)).to_array();
    let sizes = PrimitiveArray::from_iter((0..50).map(|i| ((i % 5) + 1) as u32)).to_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Filter every third list.
    let mask = Mask::from_iter((0..50).map(|i| i % 3 == 0));
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), 17); // 50/3 + 1
    let result_list = result.to_listview();

    // Check a few samples.
    assert_eq!(result_list.size_at(0), 1); // 0 % 5 + 1
    assert_eq!(result_list.size_at(1), 4); // 3 % 5 + 1
    assert_eq!(result_list.size_at(2), 2); // 6 % 5 + 1
}

#[test]
fn test_filter_alternating_pattern() {
    // Test with alternating true/false pattern.
    let elements = buffer![1u8, 2, 3, 4, 5, 6, 7, 8].into_array();
    let offsets = buffer![0u32, 2, 4, 6].into_array();
    let sizes = buffer![2u32, 2, 2, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let mask = Mask::from_iter([true, false, true, false]);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), 2);
    let result_list = result.to_listview();

    assert_eq!(result_list.size_at(0), 2);
    assert_eq!(result_list.size_at(1), 2);
}

#[test]
fn test_filter_with_mixed_validity() {
    // Test filtering with mixed validity in both data and filter.
    let elements = buffer![1u8, 2, 3, 4, 5, 6, 7, 8].into_array();
    let offsets = buffer![0u32, 2, 4, 6].into_array();
    let sizes = buffer![2u32, 2, 2, 2].into_array();
    let validity =
        Validity::Array(BoolArray::from_iter(vec![true, false, true, false]).into_array());

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    // Filter keeps first and third (both happen to align with validity).
    let mask = Mask::from_iter([true, false, true, false]);
    let result = filter(&listview, &mask).unwrap();

    assert_eq!(result.len(), 2);
    let result_list = result.to_listview();

    assert!(result_list.is_valid(0)); // First was valid.
    assert!(result_list.is_valid(1)); // Third was valid.
}

#[rstest]
#[case::dense_selection(true, 4, 1)] // Most selected - dense case with 5 items.
#[case::sparse_selection(false, 3, 3)] // Few selected from 30 lists.
fn test_filter_density_patterns(
    #[case] is_dense: bool,
    #[case] expected_count: usize,
    #[case] expected_size: usize,
) {
    // Handle both dense (boolean mask) and sparse (indices) patterns.
    if is_dense {
        // Dense case.
        let elements = buffer![100i64, 200, 300, 400, 500].into_array();
        let offsets = buffer![0u32, 1, 2, 3, 4].into_array();
        let sizes = buffer![1u32, 1, 1, 1, 1].into_array();

        let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
            .unwrap()
            .to_array();

        let mask = Mask::from_iter([true, true, false, true, true]);
        let result = filter(&listview, &mask).unwrap();

        assert_eq!(result.len(), expected_count);
        let result_list = result.to_listview();

        for i in 0..expected_count {
            assert_eq!(result_list.size_at(i), expected_size);
        }
    } else {
        // Sparse case.
        let elements = buffer![0i32..100].into_array();
        let offsets = PrimitiveArray::from_iter((0..30).map(|i| (i * 2) as u32)).to_array();
        let sizes = PrimitiveArray::from_iter(vec![3u32; 30]).to_array();

        let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
            .unwrap()
            .to_array();

        let mut mask_vec = vec![false; 30];
        mask_vec[5] = true;
        mask_vec[15] = true;
        mask_vec[25] = true;
        let mask = Mask::from_iter(mask_vec);

        let result = filter(&listview, &mask).unwrap();

        assert_eq!(result.len(), expected_count);
        let result_list = result.to_listview();

        for i in 0..expected_count {
            assert_eq!(result_list.size_at(i), expected_size);
        }
    }
}

#[test]
fn test_filter_with_gaps_and_out_of_order() {
    // This test specifically catches the old bad implementation.
    // Create a ListView with gaps in the elements array and out-of-order offsets.
    // Elements: [1, 2, 3, _, _, 6, 7, 8, 9, 10] (where _ are unused).
    // Lists: [[6, 7], [1, 2], [8, 9, 10]].
    // Offsets: [5, 0, 7] (out of order, with gaps!).
    // Sizes: [2, 2, 3].
    let elements = buffer![1i32, 2, 3, 999, 999, 6, 7, 8, 9, 10].into_array();
    let offsets = buffer![5u32, 0, 7].into_array();
    let sizes = buffer![2u32, 2, 3].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Filter to select all three lists.
    let mask = Mask::from_iter([true, true, true]);
    let result = filter(&listview, &mask).unwrap();
    let result_list = result.to_listview();

    // Check that we have 3 lists.
    assert_eq!(result_list.len(), 3);

    // Check the elements: with our slice-based approach, we keep gaps in the middle
    // for performance. Elements are [1, 2, 3, 999, 999, 6, 7, 8, 9, 10].
    // This is intentional - we prioritize O(1) slicing over removing unused elements.
    let filtered_elements = result_list.elements().to_primitive();
    assert_eq!(
        filtered_elements.as_slice::<i32>(),
        &[1, 2, 3, 999, 999, 6, 7, 8, 9, 10]
    );

    // Check the offsets: since min_offset=0, offsets remain [5, 0, 7].
    assert_eq!(result_list.offset_at(0), 5); // 6, 7 start at index 5 in sliced array.
    assert_eq!(result_list.offset_at(1), 0); // 1, 2 start at index 0.
    assert_eq!(result_list.offset_at(2), 7); // 8, 9, 10 start at index 7.

    // Check the sizes: should be unchanged [2, 2, 3].
    assert_eq!(result_list.size_at(0), 2);
    assert_eq!(result_list.size_at(1), 2);
    assert_eq!(result_list.size_at(2), 3);
}

#[test]
fn test_filter_high_starting_offset() {
    // This test catches issues when offsets don't start at 0.
    // Create a ListView where offsets start at 100, not 0.
    // Lists: [[103, 104], [101], [102, 103]]
    let mut elements_vec = vec![0i32; 106];
    elements_vec[100] = 100;
    elements_vec[101] = 101;
    elements_vec[102] = 102;
    elements_vec[103] = 103;
    elements_vec[104] = 104;
    elements_vec[105] = 105;
    let elements = PrimitiveArray::from_iter(elements_vec).into_array();

    // Offsets start at 103, not 0!
    let offsets = buffer![103u32, 101, 102].into_array();
    let sizes = buffer![2u32, 1, 2].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    // Filter to select the second and third lists: [[101], [102, 103]].
    let mask = Mask::from_iter([false, true, true]);
    let result = filter(&listview, &mask).unwrap();
    let result_list = result.to_listview();

    // Check that we have 2 lists.
    assert_eq!(result_list.len(), 2);

    // Check the elements: should be [101, 102, 103].
    let filtered_elements = result_list.elements().to_primitive();
    assert_eq!(filtered_elements.as_slice::<i32>(), &[101, 102, 103]);

    // Check the offsets: should be [0, 1] (normalized from min_offset=101).
    assert_eq!(result_list.offset_at(0), 0);
    assert_eq!(result_list.offset_at(1), 1);

    // Check the sizes: should be unchanged [1, 2].
    assert_eq!(result_list.size_at(0), 1);
    assert_eq!(result_list.size_at(1), 2);
}
