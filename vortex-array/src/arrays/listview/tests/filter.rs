// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_mask::Mask;

use super::common::create_basic_listview;
use super::common::create_empty_lists_listview;
use super::common::create_large_listview;
use super::common::create_nullable_listview;
use super::common::create_overlapping_listview;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::ToCanonical;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::assert_arrays_eq;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::validity::Validity;

// Conformance tests for common filter scenarios.
#[rstest]
#[case::basic(create_basic_listview())]
#[case::nullable(create_nullable_listview())]
#[case::empty_lists(create_empty_lists_listview())]
#[case::overlapping(create_overlapping_listview())]
#[case::large(create_large_listview())]
fn test_filter_listview_conformance(#[case] listview: ListViewArray) {
    test_filter_conformance(&listview.into_array());
}

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `filter`"]
#[test]
fn test_filter_preserves_unreferenced_elements() {
    // ListView-specific: Test that filter preserves the entire elements array.
    //
    // Logical list: [[5,6,7], [2,3], [8,9], [0,1], [1,2,3,4]]
    // Elements: [0,1,2,3,4,5,6,7,8,9]
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![5u32, 2, 8, 0, 1].into_array();
    let sizes = buffer![3u32, 2, 2, 2, 4].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // Filter to keep only 2 lists.
    let mask = Mask::from_iter([true, false, false, true, false]);
    let result = listview.filter(mask).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 2, "Wrong number of filtered lists");

    // Verify the entire elements array is preserved.
    assert_arrays_eq!(
        result_list.elements(),
        PrimitiveArray::from_iter([0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    );

    // Verify offsets are unchanged.
    assert_eq!(result_list.offset_at(0), 5, "Wrong offset at index 0");
    assert_eq!(result_list.offset_at(1), 0, "Wrong offset at index 3");
}

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `filter`"]
#[test]
fn test_filter_with_gaps() {
    // ListView-specific: Test filtering with gaps in elements array.
    //
    // Logical list: [[1,2,3], [7,8,9], [11,12], [2,3], [8,9]]
    // Elements: [1,2,3,999,999,999,7,8,9,999,11,12] (999 values are gaps)
    let elements = buffer![1i32, 2, 3, 999, 999, 999, 7, 8, 9, 999, 11, 12].into_array();
    let offsets = buffer![0u32, 6, 10, 1, 7].into_array();
    let sizes = buffer![3u32, 3, 2, 2, 2].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // Filter to keep lists with gaps and overlaps.
    let mask = Mask::from_iter([false, true, true, true, false]);
    let result = listview.filter(mask).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 3, "Wrong filter result length");

    // Verify the entire elements array is preserved including gaps.
    assert_arrays_eq!(
        result_list.elements(),
        PrimitiveArray::from_iter([1i32, 2, 3, 999, 999, 999, 7, 8, 9, 999, 11, 12])
    );

    // Verify offsets are unchanged.
    assert_eq!(result_list.offset_at(0), 6); // List 1: [7,8,9]
    assert_eq!(result_list.offset_at(1), 10); // List 2: [11,12]
    assert_eq!(result_list.offset_at(2), 1); // List 3: [2,3] (overlapping)

    // Verify the lists still read correctly.
    assert_arrays_eq!(
        result_list.list_elements_at(0).unwrap(),
        PrimitiveArray::from_iter([7i32, 8, 9])
    );
}

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `filter`"]
#[test]
fn test_filter_constant_arrays() {
    // ListView-specific: Test filter with ConstantArray for offsets/sizes.
    let elements = buffer![100i32, 200, 300, 400, 500, 600, 700, 800].into_array();

    // Case 1: Constant offsets (all lists start at same position).
    // Logical list: [[300], [300,400], [300,400,500], [300,400,500,600]]
    let constant_offsets = ConstantArray::new(2u32, 4).into_array();
    let varying_sizes = buffer![1u32, 2, 3, 4].into_array();

    let const_offset_list = ListViewArray::new(
        elements.clone(),
        constant_offsets,
        varying_sizes,
        Validity::NonNullable,
    )
    .into_array();

    let mask1 = Mask::from_iter([true, false, true, false]);
    let result1 = const_offset_list.filter(mask1).unwrap();
    let result1_list = result1.to_listview();

    assert_eq!(result1_list.len(), 2);
    assert_eq!(result1_list.offset_at(0), 2); // Both offsets are 2
    assert_eq!(result1_list.offset_at(1), 2);
    assert_eq!(result1_list.size_at(0), 1); // Sizes: 1, 3
    assert_eq!(result1_list.size_at(1), 3);

    // Case 2: Both constant (all lists are identical).
    // Logical list: [[200,300,400], [200,300,400], [200,300,400]]
    let both_constant_offsets = ConstantArray::new(1u32, 3).into_array();
    let both_constant_sizes = ConstantArray::new(3u32, 3).into_array();

    let both_const_list = ListViewArray::new(
        elements,
        both_constant_offsets,
        both_constant_sizes,
        Validity::NonNullable,
    )
    .into_array();

    let mask2 = Mask::from_iter([true, false, true]);
    let result2 = both_const_list.filter(mask2).unwrap();
    let result2_list = result2.to_listview();

    assert_eq!(result2_list.len(), 2);
    assert_eq!(result2_list.offset_at(0), 1);
    assert_eq!(result2_list.offset_at(1), 1);
    assert_eq!(result2_list.size_at(0), 3);
    assert_eq!(result2_list.size_at(1), 3);
}

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `filter`"]
#[test]
fn test_filter_extreme_offsets() {
    // ListView-specific: Test with very large offsets.
    let elements = PrimitiveArray::from_iter(0i32..10000).into_array();

    // Lists at extremes: beginning, middle, and end of the array.
    // Logical list: [[0..5], [4999..5001], [9995..10000], [2500..2503], [7500..7504]]
    let offsets = buffer![0u32, 4999, 9995, 2500, 7500].into_array();
    let sizes = buffer![5u32, 2, 5, 3, 4].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // Filter to keep only 2 lists, demonstrating we keep all 10000 elements.
    let mask = Mask::from_iter([false, true, false, false, true]);
    let result = listview.filter(mask).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 2);

    // Verify offsets are preserved.
    assert_eq!(result_list.offset_at(0), 4999);
    assert_eq!(result_list.offset_at(1), 7500);

    // Verify the entire elements array is preserved.
    assert_eq!(result_list.elements().len(), 10000);

    // Verify we can still read the correct values.
    let list0 = result_list.list_elements_at(0).unwrap();
    assert_eq!(
        list0
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        4999
    );
    assert_eq!(
        list0
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .as_primitive()
            .as_::<i32>()
            .unwrap(),
        5000
    );

    // Test sparse selection from large dataset.
    let sparse_mask = Mask::from_iter((0..5).map(|i| i == 0 || i == 4));
    let sparse_result = listview.filter(sparse_mask).unwrap();
    let sparse_list = sparse_result.to_listview();

    assert_eq!(sparse_list.len(), 2);
    assert_eq!(sparse_list.offset_at(0), 0); // First list
    assert_eq!(sparse_list.offset_at(1), 7500); // Last list
    assert_eq!(sparse_list.elements().len(), 10000); // Still keeps all elements
}
