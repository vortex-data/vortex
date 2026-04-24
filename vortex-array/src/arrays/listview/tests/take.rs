// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;

use super::common::create_basic_listview;
use super::common::create_empty_lists_listview;
use super::common::create_large_listview;
use super::common::create_nullable_listview;
use super::common::create_overlapping_listview;
use crate::IntoArray;
use crate::LEGACY_SESSION;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::assert_arrays_eq;
use crate::compute::conformance::take::test_take_conformance;
use crate::validity::Validity;

// Conformance tests for common take scenarios.
#[rstest]
#[case::basic(create_basic_listview())]
#[case::nullable(create_nullable_listview())]
#[case::empty_lists(create_empty_lists_listview())]
#[case::overlapping(create_overlapping_listview())]
#[case::large(create_large_listview())]
fn test_take_listview_conformance(#[case] listview: ListViewArray) {
    test_take_conformance(&listview.into_array());
}

// ListView-specific tests that aren't covered by conformance.

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `take`"]
#[test]
fn test_take_preserves_unreferenced_elements() {
    // ListView-specific: Test that take preserves the entire elements array
    // even when taking only a subset of lists.
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let offsets = buffer![5u32, 2, 8, 0, 1].into_array();
    let sizes = buffer![3u32, 2, 2, 2, 4].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // Take only 2 lists.
    let indices = buffer![1u32, 3].into_array();
    let result = listview.take(indices).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 2);

    // Verify the entire elements array is preserved.
    assert_arrays_eq!(
        result_list.elements(),
        PrimitiveArray::from_iter([0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    );

    // Verify offsets are preserved.
    assert_eq!(result_list.offset_at(0), 2); // List 1
    assert_eq!(result_list.offset_at(1), 0); // List 3
}

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `take`"]
#[test]
fn test_take_with_gaps() {
    // ListView-specific: Test with gaps in elements array.
    // Elements with gaps (999 values are "gaps" between used ranges).
    let elements = buffer![1i32, 2, 3, 999, 999, 999, 7, 8, 9, 999, 11, 12].into_array();
    let offsets = buffer![0u32, 6, 10, 1, 7].into_array();
    let sizes = buffer![3u32, 3, 2, 2, 2].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    let indices = buffer![1u32, 3, 4, 2].into_array();
    let result = listview.take(indices).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    // Verify the entire elements array is preserved including gaps.
    assert_arrays_eq!(
        result_list.elements(),
        PrimitiveArray::from_iter([1i32, 2, 3, 999, 999, 999, 7, 8, 9, 999, 11, 12])
    );

    // Verify the lists still read correctly despite gaps.
    assert_arrays_eq!(
        result_list.list_elements_at(0).unwrap(),
        PrimitiveArray::from_iter([7i32, 8, 9])
    );
}

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `take`"]
#[test]
fn test_take_constant_arrays() {
    // ListView-specific: Test with ConstantArray for offsets/sizes.
    let elements = buffer![100i32, 200, 300, 400, 500, 600, 700, 800].into_array();

    // Case 1: Constant offsets (all lists start at same position).
    let constant_offsets = ConstantArray::new(2u32, 4).into_array();
    let varying_sizes = buffer![1u32, 2, 3, 4].into_array();

    let const_offset_list = ListViewArray::new(
        elements.clone(),
        constant_offsets,
        varying_sizes,
        Validity::NonNullable,
    )
    .into_array();

    let indices = buffer![3u32, 0, 2].into_array();
    let result = const_offset_list.take(indices).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 3);
    assert_eq!(result_list.offset_at(0), 2); // All offsets are 2
    assert_eq!(result_list.offset_at(1), 2);
    assert_eq!(result_list.offset_at(2), 2);
    assert_eq!(result_list.size_at(0), 4); // Sizes: 4, 1, 3
    assert_eq!(result_list.size_at(1), 1);
    assert_eq!(result_list.size_at(2), 3);

    // Case 2: Both constant (all lists are identical).
    let both_constant_offsets = ConstantArray::new(1u32, 3).into_array();
    let both_constant_sizes = ConstantArray::new(3u32, 3).into_array();

    let both_const_list = ListViewArray::new(
        elements,
        both_constant_offsets,
        both_constant_sizes,
        Validity::NonNullable,
    )
    .into_array();

    let indices2 = buffer![2u32, 0].into_array();
    let result2 = both_const_list.take(indices2).unwrap();
    #[expect(deprecated)]
    let result2_list = result2.to_listview();

    assert_eq!(result2_list.len(), 2);
    assert_eq!(result2_list.offset_at(0), 1);
    assert_eq!(result2_list.offset_at(1), 1);
    assert_eq!(result2_list.size_at(0), 3);
    assert_eq!(result2_list.size_at(1), 3);
}

#[ignore = "TODO(connor)[ListView]: Don't rebuild ListView after every `take`"]
#[test]
fn test_take_extreme_offsets() {
    // ListView-specific: Test with very large offsets to demonstrate
    // that we keep unreferenced elements.
    let elements = PrimitiveArray::from_iter(0i32..10000).into_array();

    // Lists at extremes: beginning, middle, and end of the array.
    let offsets = buffer![0u32, 4999, 9995, 2500, 7500].into_array();
    let sizes = buffer![5u32, 2, 5, 3, 4].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // Take only 2 lists, demonstrating we keep all 10000 elements.
    let indices = buffer![1u32, 4].into_array();
    let result = listview.take(indices).unwrap();
    #[expect(deprecated)]
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
}
