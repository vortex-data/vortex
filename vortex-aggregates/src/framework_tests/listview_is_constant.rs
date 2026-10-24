// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `is_constant` over list view arrays.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_session::VortexSession;

use crate::is_constant::is_constant;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

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
        &mut SESSION.create_execution_ctx(),
    )
    .into_array();

    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
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
    let mut ctx = SESSION.create_execution_ctx();
    assert!(!is_constant(&listview_mixed, &mut ctx).unwrap());

    // Case 2: All nulls - should be constant.
    let validity_all_null = Validity::AllInvalid;
    let listview_all_null = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, validity_all_null)
            .with_zero_copy_to_list(true)
    }
    .into_array();
    let mut ctx2 = SESSION.create_execution_ctx();
    assert!(is_constant(&listview_all_null, &mut ctx2).unwrap());
}

#[test]
fn test_constant_repeated_same_lists() {
    // Test multiple lists that are identical (overlapping).
    // Logical lists: [[10,20,30], [10,20,30], [10,20,30], [10,20,30]]
    let elements = buffer![10i32, 20, 30].into_array();
    let offsets = buffer![0i32, 0, 0, 0].into_array(); // All point to same start.
    let sizes = buffer![3i32, 3, 3, 3].into_array(); // All same size.

    let mut ctx = SESSION.create_execution_ctx();
    let listview =
        ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx).into_array();

    // All lists are [10, 20, 30] so should be constant.
    assert!(is_constant(&listview, &mut ctx).unwrap());
}
