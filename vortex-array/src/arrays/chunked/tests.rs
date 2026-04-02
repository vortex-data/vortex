// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::buffer;

use crate::IntoArray;
use crate::accessor::ArrayAccessor;
use crate::array::VTable;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::PType::I32;
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

#[test]
fn slice_middle() {
    assert_arrays_eq!(
        chunked_array().slice(2..5).unwrap(),
        PrimitiveArray::from_iter([3u64, 4, 5])
    );
}

#[test]
fn slice_begin() {
    assert_arrays_eq!(
        chunked_array().slice(1..3).unwrap(),
        PrimitiveArray::from_iter([2u64, 3])
    );
}

#[test]
fn slice_aligned() {
    assert_arrays_eq!(
        chunked_array().slice(3..6).unwrap(),
        PrimitiveArray::from_iter([4u64, 5, 6])
    );
}

#[test]
fn slice_many_aligned() {
    assert_arrays_eq!(
        chunked_array().slice(0..6).unwrap(),
        PrimitiveArray::from_iter([1u64, 2, 3, 4, 5, 6])
    );
}

#[test]
fn slice_end() {
    assert_arrays_eq!(
        chunked_array().slice(7..8).unwrap(),
        PrimitiveArray::from_iter([8u64])
    );
}

#[test]
fn slice_exactly_end() {
    assert_arrays_eq!(
        chunked_array().slice(6..9).unwrap(),
        PrimitiveArray::from_iter([7u64, 8, 9])
    );
}

#[test]
fn slice_empty() {
    let chunked = ChunkedArray::try_new(vec![], PType::U32.into()).unwrap();
    let sliced = chunked.slice(0..0).unwrap();

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
    assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2]));
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
    assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2, 3, 4]));
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
    assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2, 3, 4]));
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
            ChunkedArray::try_new(vec![struct_array.clone().into_array()], dtype.clone())
                .unwrap()
                .into_array(),
        ],
        dtype,
    )
    .unwrap()
    .into_array();
    let canonical_struct = chunked.to_struct();
    let canonical_varbin = canonical_struct.unmasked_fields()[0].to_varbinview();
    let original_varbin = struct_array.unmasked_fields()[0].to_varbinview();
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

    let canon_values = chunked_list.unwrap().as_array().to_listview();

    assert_eq!(l1.scalar_at(0).unwrap(), canon_values.scalar_at(0).unwrap());
    assert_eq!(l2.scalar_at(0).unwrap(), canon_values.scalar_at(1).unwrap());
}

#[test]
fn with_slots_updates_nchunks_len_and_offsets() {
    let orig = chunked_array();
    let slots = vec![
        Some(buffer![0u64, 4, 9].into_array()),
        Some(buffer![10u64, 11, 12, 13].into_array()),
        Some(buffer![14u64, 15, 16, 17, 18].into_array()),
    ];
    let expected_nchunks = slots.len() - 1;
    let expected_len = orig.len();

    let mut data = orig.into_data();
    <Chunked as VTable>::with_slots(&mut data, slots).unwrap();
    let array = ChunkedArray::try_from_data(data).unwrap();

    assert_eq!(array.nchunks(), expected_nchunks);
    assert_eq!(array.len(), expected_len);
    assert_eq!(array.chunk_offsets(), buffer![0u64, 4, 9]);
    assert_arrays_eq!(
        array.chunk(0).clone(),
        PrimitiveArray::from_iter([10u64, 11, 12, 13])
    );
    assert_arrays_eq!(
        array.chunk(1).clone(),
        PrimitiveArray::from_iter([14u64, 15, 16, 17, 18])
    );
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_iter([10u64, 11, 12, 13, 14, 15, 16, 17, 18])
    );
}
