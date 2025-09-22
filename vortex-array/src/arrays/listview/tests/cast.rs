// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_dtype::{DType, Nullability, PType};

use super::ToListView;
use crate::arrays::{BoolArray, ListViewArray};
use crate::compute::cast;
use crate::validity::Validity;
use crate::{Array, IntoArray};

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

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let target_dtype = DType::List(
        Arc::new(DType::Primitive(to_ptype, Nullability::NonNullable)),
        Nullability::NonNullable,
    );

    let result = cast(&listview, &target_dtype).unwrap();
    assert_eq!(result.dtype(), &target_dtype);

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
    let elements = buffer![10i32, 20, 30, 40].into_array();
    let offsets = buffer![0u32, 2].into_array();
    let sizes = buffer![2u32, 2].into_array();
    let validity = Validity::Array(BoolArray::from_iter(vec![true, false]).into_array());

    let listview = ListViewArray::try_new(elements, offsets, sizes, validity)
        .unwrap()
        .to_array();

    let target_dtype = DType::List(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        Nullability::Nullable,
    );

    let result = cast(&listview, &target_dtype).unwrap();
    assert_eq!(result.dtype(), &target_dtype);

    let result_list = result.to_listview();
    assert!(result_list.is_valid(0));
    assert!(result_list.is_invalid(1));
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

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

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

    let result = cast(&listview, &target_dtype).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), list_count);

    for (i, expected_size) in expected_sizes.iter().enumerate() {
        assert_eq!(result_list.size_at(i), *expected_size);
    }
}

#[test]
fn test_cast_large_dataset() {
    // Test with larger data.
    let elements = buffer![0u16..100].into_array();
    let offsets = buffer![
        0u32, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76
    ]
    .into_array();
    let sizes = buffer![4u32; 20].into_array();

    let listview = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
        .unwrap()
        .to_array();

    let target_dtype = DType::List(
        Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
        Nullability::NonNullable,
    );

    let result = cast(&listview, &target_dtype).unwrap();
    let result_list = result.to_listview();

    assert_eq!(result_list.len(), 20);
    for i in 0..20 {
        assert_eq!(result_list.size_at(i), 4);
    }
}
