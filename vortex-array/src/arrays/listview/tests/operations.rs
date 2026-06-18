// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::common::create_basic_listview;
use super::common::create_large_listview;
use super::common::create_nullable_listview;
use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Slice tests
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_slice_comprehensive() {
    // Comprehensive test for basic slicing, full array, and single element cases.
    // Logical lists: [[1,2,3], [4,5], [6,7,8], [9,10]]
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
    let offsets = buffer![0i32, 3, 5, 7].into_array();
    let sizes = buffer![3i32, 2, 3, 2].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // Test basic slice [1..3] - middle portion.
    let sliced = listview.slice(1..3).unwrap();
    let sliced_list = sliced.as_::<ListView>();
    assert_eq!(sliced_list.len(), 2, "Wrong slice length");
    assert_eq!(sliced_list.offset_at(0), 3, "Wrong offset for list[1]");
    assert_eq!(sliced_list.size_at(0), 2, "Wrong size for list[1]");
    assert_eq!(sliced_list.offset_at(1), 5, "Wrong offset for list[2]");
    assert_eq!(sliced_list.size_at(1), 3, "Wrong size for list[2]");

    // Test full array slice [0..4].
    let full = listview.slice(0..4).unwrap();
    let full_list = full.as_::<ListView>();
    assert_eq!(full_list.len(), 4, "Full slice should preserve length");
    for i in 0..4 {
        // Compare the sliced elements
        assert_eq!(
            full_list
                .array()
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            listview
                .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            "Mismatch at index {}",
            i
        );
    }

    // Test single element slice [2..3].
    let single = listview.slice(2..3).unwrap();
    let single_list = single.as_::<ListView>();
    assert_eq!(single_list.len(), 1, "Single element slice failed");
    assert_eq!(single_list.offset_at(0), 5, "Wrong offset for single slice");
    assert_eq!(single_list.size_at(0), 3, "Wrong size for single slice");
}

#[test]
fn test_slice_out_of_order() {
    // ListView-specific: Test slicing with out-of-order offsets.
    // Logical lists: [[70,80], [10,20,30], [40,50,60], [90], [30]]
    let elements = buffer![10i32, 20, 30, 40, 50, 60, 70, 80, 90].into_array();
    let offsets = buffer![6i32, 0, 3, 8, 2].into_array(); // Out of order.
    let sizes = buffer![2i32, 3, 3, 1, 1].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // Slice [1..4] should maintain the out-of-order offsets.
    let sliced = listview.slice(1..4).unwrap();
    let sliced_list = sliced.as_::<ListView>();

    assert_eq!(
        sliced_list.len(),
        3,
        "Slice [1..4] of out-of-order ListView should produce 3 lists"
    );
    assert_eq!(
        sliced_list.offset_at(0),
        0,
        "First list should have offset 0 (from original index 1)"
    );
    assert_eq!(sliced_list.size_at(0), 3, "First list should have size 3");
    assert_eq!(
        sliced_list.offset_at(1),
        3,
        "Second list should have offset 3 (from original index 2)"
    );
    assert_eq!(sliced_list.size_at(1), 3, "Second list should have size 3");
    assert_eq!(
        sliced_list.offset_at(2),
        8,
        "Third list should have offset 8 (from original index 3)"
    );
    assert_eq!(sliced_list.size_at(2), 1, "Third list should have size 1");

    // Verify the actual list contents are correct.
    assert_arrays_eq!(
        sliced_list.list_elements_at(0).unwrap(),
        PrimitiveArray::from_iter([10i32, 20, 30])
    );
    assert_arrays_eq!(
        sliced_list.list_elements_at(1).unwrap(),
        PrimitiveArray::from_iter([40i32, 50, 60])
    );
    assert_arrays_eq!(
        sliced_list.list_elements_at(2).unwrap(),
        PrimitiveArray::from_iter([90i32])
    );
}

#[test]
fn test_slice_with_nulls() {
    // Test slicing with nullable ListView.
    // Logical lists: [[1,2], null, [5,6], null]
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
    let offsets = buffer![0i32, 2, 4, 6].into_array();
    let sizes = buffer![2i32, 2, 2, 2].into_array();
    let validity =
        Validity::Array(BoolArray::from_iter(vec![true, false, true, false]).into_array());

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, validity)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    // Slice [1..3] should preserve nulls.
    let sliced = listview.slice(1..3).unwrap();
    let sliced_list = sliced.as_::<ListView>();

    assert_eq!(sliced_list.len(), 2);
    assert!(
        sliced_list
            .array()
            .is_invalid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Original index 1 was null.
    assert!(
        sliced_list
            .array()
            .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Original index 2 was valid.

    // Verify offsets and sizes are preserved.
    assert_eq!(sliced_list.offset_at(0), 2);
    assert_eq!(sliced_list.size_at(0), 2);
    assert_eq!(sliced_list.offset_at(1), 4);
    assert_eq!(sliced_list.size_at(1), 2);
}

// Parameterized edge case tests.
#[rstest]
#[case::empty_range(2, 2, Some(0))] // Empty range [2..2]
#[case::out_of_bounds(10, 15, None)] // Out of bounds [10..15]
#[case::invalid_range(3, 1, None)] // Invalid range where start > stop
fn test_slice_edge_cases(
    #[case] start: usize,
    #[case] stop: usize,
    #[case] expected_len: Option<usize>,
) {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    match expected_len {
        Some(len) => {
            let sliced = listview.slice(start..stop).unwrap();
            assert_eq!(sliced.len(), len);
        }
        None => {
            // slice will panic or return empty for invalid ranges
            if start < stop && stop <= listview.len() {
                let sliced = listview.slice(start..stop).unwrap();
                assert_eq!(sliced.len(), 0);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Cast tests
////////////////////////////////////////////////////////////////////////////////////////////////////

#[rstest]
#[case::i32_to_i64(PType::I32, PType::I64)]
#[case::f32_to_f64(PType::F32, PType::F64)]
#[case::u8_to_u16(PType::U8, PType::U16)]
fn test_cast_numeric_types(#[case] from_ptype: PType, #[case] to_ptype: PType) {
    let elements = match from_ptype {
        PType::I32 => buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        PType::F32 => buffer![1.0f32, 2.0, 3.0, 4.0].into_array(),
        PType::U8 => buffer![1u8, 2, 3, 4, 5, 6, 7, 8].into_array(),
        _ => panic!("Unexpected type"),
    };

    let (offsets, sizes) = match from_ptype {
        PType::I32 => (
            buffer![0u32, 2, 4].into_array(),
            buffer![2u32, 2, 2].into_array(),
        ),
        PType::F32 => (buffer![0u32, 2].into_array(), buffer![2u32, 2].into_array()),
        PType::U8 => (
            buffer![0u32, 3, 5].into_array(),
            buffer![3u32, 2, 3].into_array(),
        ),
        _ => panic!("Unexpected type"),
    };

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    let target_dtype = DType::List(
        Arc::new(DType::Primitive(to_ptype, Nullability::NonNullable)),
        Nullability::NonNullable,
    );

    let result = listview.cast(target_dtype.clone()).unwrap();
    assert_eq!(result.dtype(), &target_dtype);

    #[expect(deprecated)]
    let result_list = result.to_listview();
    assert!(
        result_list.len() == 3 || result_list.len() == 2,
        "Expected 2 or 3 lists"
    );

    // Check that elements were properly cast.
    let elements = result_list.elements();
    assert_eq!(
        elements.dtype(),
        &DType::Primitive(to_ptype, Nullability::NonNullable)
    );
}

#[test]
fn test_cast_with_nulls() {
    // Logical lists: [[10,20], null]
    let elements = buffer![10i32, 20, 30, 40].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 2].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false]).into_array());

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, validity)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    let target_dtype = DType::List(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        Nullability::Nullable,
    );

    let result = listview.cast(target_dtype.clone()).unwrap();
    assert_eq!(result.dtype(), &target_dtype);

    #[expect(deprecated)]
    let result_list = result.to_listview();
    assert!(
        result_list
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert!(
        result_list
            .is_invalid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
}

#[rstest]
#[case::empty_lists(vec![0, 1, 0, 1], 4)]
#[case::overlapping(vec![3, 3, 5], 3)]
fn test_cast_special_patterns(#[case] expected_sizes: Vec<usize>, #[case] list_count: usize) {
    let is_empty_case = list_count == 4;

    let (elements, offsets, sizes) = if is_empty_case {
        // Empty lists case.
        (
            buffer![42i32, 43].into_array(),
            buffer![0u32, 0, 1, 1].into_array(),
            buffer![0u32, 1, 0, 1].into_array(),
        )
    } else {
        // Overlapping case.
        (
            buffer![1.0f32, 2.0, 3.0, 4.0, 5.0].into_array(),
            buffer![0u32, 1, 0].into_array(),
            buffer![3u32, 3, 5].into_array(),
        )
    };

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    let target_dtype = if is_empty_case {
        DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::NonNullable,
        )
    } else {
        DType::List(
            Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
            Nullability::NonNullable,
        )
    };

    let result = listview.cast(target_dtype).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), list_count);

    for (i, expected_size) in expected_sizes.iter().enumerate() {
        assert_eq!(result_list.size_at(i), *expected_size);
    }
}

#[test]
fn test_cast_large_dataset() {
    // Test with larger data.
    // Logical lists: [[0..4], [4..8], [8..12], ..., [76..80]] (20 lists of size 4)
    let elements = buffer![0u16..100].into_array();
    let offsets = buffer![
        0u32, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76
    ]
    .into_array();
    let sizes = buffer![4u32; 20].into_array();

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    let target_dtype = DType::List(
        Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
        Nullability::NonNullable,
    );

    let result = listview.cast(target_dtype).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 20);
    for i in 0..20 {
        assert_eq!(result_list.size_at(i), 4);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Zip tests
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_zip_widens_false_element_nullability() -> VortexResult<()> {
    // [[1, 2], [3], [4]]
    let if_true = ListViewArray::new(
        buffer![1i32, 2, 3, 4].into_array(),
        buffer![0u32, 2, 3].into_array(),
        buffer![2u32, 1, 1].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    // [[10, null], [30], [40]]
    let if_false = ListViewArray::new(
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30), Some(40)]).into_array(),
        buffer![0u32, 2, 3].into_array(),
        buffer![2u32, 1, 1].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    let mask = Mask::from_iter([false, true, false]);

    let result = mask
        .into_array()
        .zip(if_true, if_false)?
        .execute::<ArrayRef>(&mut LEGACY_SESSION.create_execution_ctx())?;
    assert!(result.is::<ListView>());
    assert_eq!(
        result.dtype(),
        &DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::NonNullable,
        )
    );

    // [[10, null], [3], [40]]
    let expected = ListViewArray::new(
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(3), Some(40)]).into_array(),
        buffer![0u32, 2, 3].into_array(),
        buffer![2u32, 1, 1].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    assert_arrays_eq!(result, expected);
    Ok(())
}

#[test]
fn test_zip_widens_true_element_nullability() -> VortexResult<()> {
    // [[1, null], [3], [4]]
    let if_true = ListViewArray::new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array(),
        buffer![0u32, 2, 3].into_array(),
        buffer![2u32, 1, 1].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    // [[10], [20], [30]]
    let if_false = ListViewArray::new(
        buffer![10i32, 20, 30].into_array(),
        buffer![0u32, 1, 2].into_array(),
        buffer![1u32, 1, 1].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    let mask = Mask::from_iter([true, false, true]);

    let result = mask
        .into_array()
        .zip(if_true, if_false)?
        .execute::<ArrayRef>(&mut LEGACY_SESSION.create_execution_ctx())?;
    assert!(result.is::<ListView>());
    assert_eq!(
        result.dtype(),
        &DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::NonNullable,
        )
    );

    // [[1, null], [20], [4]]
    let expected = ListViewArray::new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(20), Some(4)]).into_array(),
        buffer![0u32, 2, 3].into_array(),
        buffer![2u32, 1, 1].into_array(),
        Validity::NonNullable,
    )
    .into_array();
    assert_arrays_eq!(result, expected);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Constant tests
////////////////////////////////////////////////////////////////////////////////////////////////////

// Parameterized tests for is_constant scenarios.
#[rstest]
#[case::different_sizes(
    buffer![1i32, 2, 3, 4],
    buffer![0i32, 1, 2],
    buffer![1i32, 1, 2], // Different sizes
    Validity::NonNullable,
    false
)]
#[case::different_elements(
    buffer![1i32, 2, 3, 4],
    buffer![0i32, 2],
    buffer![2i32, 2], // Same size, different elements
    Validity::NonNullable,
    false
)]
#[case::same_empty_lists(
    buffer![99i32], // Dummy element
    buffer![0i32, 0, 0],
    buffer![0i32, 0, 0], // All empty lists
    Validity::NonNullable,
    true
)]
#[case::single_list(
    buffer![1i32, 2, 3],
    buffer![0i32],
    buffer![3i32],
    Validity::NonNullable,
    true
)]
#[case::overlapping_different(
    buffer![1i32, 2, 3, 4],
    buffer![0i32, 1, 2], // Overlapping but different
    buffer![2i32, 2, 2],
    Validity::NonNullable,
    false
)]
fn test_is_constant_basic(
    #[case] elements: vortex_buffer::Buffer<i32>,
    #[case] offsets: vortex_buffer::Buffer<i32>,
    #[case] sizes: vortex_buffer::Buffer<i32>,
    #[case] validity: Validity,
    #[case] expected: bool,
) {
    let listview = ListViewArray::new(
        elements.into_array(),
        offsets.into_array(),
        sizes.into_array(),
        validity,
    )
    .into_array();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    assert_eq!(is_constant(&listview, &mut ctx).unwrap(), expected);
}

#[test]
fn test_constant_with_constant_elements() {
    // Test with ConstantArray as elements - all lists point to same constant value.
    // Logical lists: [[42,42], [42,42], [42,42]]
    let elements = ConstantArray::new(42i32, 10).into_array();
    let offsets = buffer![0i32, 2, 4].into_array();
    let sizes = buffer![2i32, 2, 2].into_array();

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    // All lists contain [42, 42] so should be constant.
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    assert!(is_constant(&listview, &mut ctx).unwrap());
}

#[test]
fn test_constant_with_nulls() {
    // Test nullable ListView scenarios.
    // Logical lists: [[1,2], [3,4]] (validity varies by case)
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let offsets = buffer![0i32, 2].into_array();
    let sizes = buffer![2i32, 2].into_array();

    // Case 1: Mixed valid and null - not constant.
    let validity_mixed = Validity::Array(BoolArray::from_iter(vec![true, false]).into_array());
    let listview_mixed = unsafe {
        ListViewArray::new_unchecked(
            elements.clone(),
            offsets.clone(),
            sizes.clone(),
            validity_mixed,
        )
        .with_zero_copy_to_list(true)
    }
    .into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    assert!(!is_constant(&listview_mixed, &mut ctx).unwrap());

    // Case 2: All nulls - should be constant.
    let validity_all_null = Validity::AllInvalid;
    let listview_all_null = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, validity_all_null)
            .with_zero_copy_to_list(true)
    }
    .into_array();
    let mut ctx2 = LEGACY_SESSION.create_execution_ctx();
    assert!(is_constant(&listview_all_null, &mut ctx2).unwrap());
}

#[test]
fn test_constant_repeated_same_lists() {
    // Test multiple lists that are identical (overlapping).
    // Logical lists: [[10,20,30], [10,20,30], [10,20,30], [10,20,30]]
    let elements = buffer![10i32, 20, 30].into_array();
    let offsets = buffer![0i32, 0, 0, 0].into_array(); // All point to same start.
    let sizes = buffer![3i32, 3, 3, 3].into_array(); // All same size.

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    // All lists are [10, 20, 30] so should be constant.
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    assert!(is_constant(&listview, &mut ctx).unwrap());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mask tests
////////////////////////////////////////////////////////////////////////////////////////////////////

// Conformance tests for common mask scenarios.
#[rstest]
#[case::basic(create_basic_listview())]
#[case::nullable(create_nullable_listview())]
#[case::large(create_large_listview())]
fn test_mask_listview_conformance(#[case] listview: ListViewArray) {
    test_mask_conformance(&listview.into_array());
}

#[test]
fn test_mask_preserves_structure() {
    // ListView-specific: Verify mask preserves offsets and sizes.
    // Logical lists: [[1,2], [3,4], [5,6], [7,8]]
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
    let offsets = buffer![0u32, 2, 4, 6].into_array();
    let sizes = buffer![2u32, 2, 2, 2].into_array();

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::NonNullable)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    // Mask sets elements to null where true.
    let selection = Mask::from_iter([true, false, true, true]);
    let result = listview.mask((!&selection).into_array()).unwrap();

    assert_eq!(result.len(), 4); // Length is preserved.
    #[expect(deprecated)]
    let result_list = result.to_listview();

    // Check validity: true in selection means null.
    assert!(
        !result_list
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Masked.
    assert!(
        result_list
            .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Not masked.
    assert!(
        !result_list
            .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Masked.
    assert!(
        !result_list
            .is_valid(3, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Masked.

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
fn test_mask_with_existing_nulls() {
    // ListView-specific: Test interaction between existing nulls and mask.
    // Logical lists: [[10,20], null, [50,60]]
    let elements = buffer![10i32, 20, 30, 40, 50, 60].into_array();
    let offsets = buffer![0u32, 2, 4].into_array();
    let sizes = buffer![2u32, 2, 2].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array());

    let listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, validity)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    // Mask additional elements.
    let selection = Mask::from_iter([false, true, true]);
    let result = listview.mask((!&selection).into_array()).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    // Check combined validity:
    assert!(
        result_list
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Was valid, mask is false -> valid.
    assert!(
        !result_list
            .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Was invalid, mask is true -> invalid.
    assert!(
        !result_list
            .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Was valid, mask is true -> invalid.
}

#[test]
fn test_mask_with_gaps() {
    // ListView-specific: Mask with gaps in elements.
    // Logical lists: [[1,2], [5,6], [9,10]] (999 values are gaps)
    let elements = buffer![1i32, 2, 999, 999, 5, 6, 999, 999, 9, 10].into_array();
    let offsets = buffer![0u32, 4, 8].into_array();
    let sizes = buffer![2u32, 2, 2].into_array();

    let listview = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    let selection = Mask::from_iter([true, false, false]);
    let result = listview.mask((!&selection).into_array()).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 3);
    assert!(
        !result_list
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Masked
    assert!(
        result_list
            .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Not masked
    assert!(
        result_list
            .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Not masked

    // Offsets and sizes still preserved
    assert_eq!(result_list.offset_at(1), 4);
    assert_eq!(result_list.size_at(1), 2);
}

#[test]
fn test_mask_constant_arrays() {
    // ListView-specific: Test mask with ConstantArray offsets/sizes.
    // Logical lists: [[200,300], [200,300], [200,300]]
    let elements = buffer![100i32, 200, 300, 400, 500, 600].into_array();

    // All lists start at offset 1 and have size 2.
    let constant_offsets = ConstantArray::new(1u32, 3).into_array();
    let constant_sizes = ConstantArray::new(2u32, 3).into_array();

    let const_list = ListViewArray::new(
        elements,
        constant_offsets,
        constant_sizes,
        Validity::NonNullable,
    )
    .into_array();

    let selection = Mask::from_iter([false, true, false]);
    let result = const_list.mask((!&selection).into_array()).unwrap();
    #[expect(deprecated)]
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 3);
    assert!(
        result_list
            .is_valid(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert!(
        !result_list
            .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    ); // Masked
    assert!(
        result_list
            .is_valid(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );

    // All offsets and sizes remain constant
    assert_eq!(result_list.offset_at(0), 1);
    assert_eq!(result_list.offset_at(1), 1);
    assert_eq!(result_list.offset_at(2), 1);
    assert_eq!(result_list.size_at(0), 2);
    assert_eq!(result_list.size_at(1), 2);
    assert_eq!(result_list.size_at(2), 2);
}
