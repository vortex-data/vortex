// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::{Buffer, buffer};
use vortex_dtype::PType::I32;
use vortex_dtype::{DType, NativePType, Nullability, PType};

use crate::IntoArray;
use crate::accessor::ArrayAccessor;
use crate::array::Array;
use crate::arrays::{ChunkedArray, ChunkedVTable, ListArray, StructArray, VarBinViewArray};
use crate::canonical::ToCanonical;
use crate::validity::Validity;

fn chunked_array() -> ChunkedArray {
    ChunkedArray::try_new(
        vec![
            buffer![1u64, 2, 3].into_array(),
            buffer![4u64, 5, 6].into_array(),
            buffer![7u64, 8, 9].into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .unwrap()
}

fn assert_equal_slices<T: NativePType>(arr: &dyn Array, slice: &[T]) {
    let mut values = Vec::with_capacity(arr.len());
    if let Some(arr) = arr.as_opt::<ChunkedVTable>() {
        arr.chunks()
            .iter()
            .map(|a| a.to_primitive())
            .for_each(|a| values.extend_from_slice(a.as_slice::<T>()));
    } else {
        values.extend_from_slice(arr.to_primitive().as_slice::<T>());
    }
    assert_eq!(values, slice);
}

#[test]
fn slice_middle() {
    assert_equal_slices(&chunked_array().slice(2..5), &[3u64, 4, 5])
}

#[test]
fn slice_begin() {
    assert_equal_slices(&chunked_array().slice(1..3), &[2u64, 3]);
}

#[test]
fn slice_aligned() {
    assert_equal_slices(&chunked_array().slice(3..6), &[4u64, 5, 6]);
}

#[test]
fn slice_many_aligned() {
    assert_equal_slices(&chunked_array().slice(0..6), &[1u64, 2, 3, 4, 5, 6]);
}

#[test]
fn slice_end() {
    assert_equal_slices(&chunked_array().slice(7..8), &[8u64]);
}

#[test]
fn slice_exactly_end() {
    assert_equal_slices(&chunked_array().slice(6..9), &[7u64, 8, 9]);
}

#[test]
fn slice_empty() {
    let chunked = ChunkedArray::try_new(vec![], PType::U32.into()).unwrap();
    let sliced = chunked.slice(0..0);

    assert!(sliced.is_empty());
}

#[test]
fn scalar_at_empty_children_both_sides() {
    let array = ChunkedArray::try_new(
        vec![
            Buffer::<u64>::empty().into_array(),
            Buffer::<u64>::empty().into_array(),
            buffer![1u64, 2].into_array(),
            Buffer::<u64>::empty().into_array(),
            Buffer::<u64>::empty().into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .unwrap();
    assert_eq!(array.scalar_at(0), 1u64.into());
    assert_eq!(array.scalar_at(1), 2u64.into());
}

#[test]
fn scalar_at_empty_children_trailing() {
    let array = ChunkedArray::try_new(
        vec![
            buffer![1u64, 2].into_array(),
            Buffer::<u64>::empty().into_array(),
            Buffer::<u64>::empty().into_array(),
            buffer![3u64, 4].into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .unwrap();
    assert_eq!(array.scalar_at(0), 1u64.into());
    assert_eq!(array.scalar_at(1), 2u64.into());
    assert_eq!(array.scalar_at(2), 3u64.into());
    assert_eq!(array.scalar_at(3), 4u64.into());
}

#[test]
fn scalar_at_empty_children_leading() {
    let array = ChunkedArray::try_new(
        vec![
            Buffer::<u64>::empty().into_array(),
            Buffer::<u64>::empty().into_array(),
            buffer![1u64, 2].into_array(),
            buffer![3u64, 4].into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .unwrap();
    assert_eq!(array.scalar_at(0), 1u64.into());
    assert_eq!(array.scalar_at(1), 2u64.into());
    assert_eq!(array.scalar_at(2), 3u64.into());
    assert_eq!(array.scalar_at(3), 4u64.into());
}

#[test]
pub fn pack_nested_structs() {
    let struct_array = StructArray::try_new(
        ["a"].into(),
        vec![VarBinViewArray::from_iter_str(["foo", "bar", "baz", "quak"]).into_array()],
        4,
        Validity::NonNullable,
    )
    .unwrap();
    let dtype = struct_array.dtype().clone();
    let chunked = ChunkedArray::try_new(
        vec![
            ChunkedArray::try_new(vec![struct_array.to_array()], dtype.clone())
                .unwrap()
                .into_array(),
        ],
        dtype,
    )
    .unwrap()
    .into_array();
    let canonical_struct = chunked.to_struct();
    let canonical_varbin = canonical_struct.fields()[0].to_varbinview();
    let original_varbin = struct_array.fields()[0].to_varbinview();
    let orig_values =
        original_varbin.with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
    let canon_values =
        canonical_varbin.with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
    assert_eq!(orig_values, canon_values);
}

#[test]
pub fn pack_nested_lists() {
    let l1 = ListArray::try_new(
        buffer![1, 2, 3, 4].into_array(),
        buffer![0, 3].into_array(),
        Validity::NonNullable,
    )
    .unwrap();

    let l2 = ListArray::try_new(
        buffer![5, 6].into_array(),
        buffer![0, 2].into_array(),
        Validity::NonNullable,
    )
    .unwrap();

    let chunked_list = ChunkedArray::try_new(
        vec![l1.clone().into_array(), l2.clone().into_array()],
        DType::List(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        ),
    );

    let canon_values = chunked_list.unwrap().to_listview();

    assert_eq!(l1.scalar_at(0), canon_values.scalar_at(0));
    assert_eq!(l2.scalar_at(0), canon_values.scalar_at(1));
}
