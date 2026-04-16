// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::buffer;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::accessor::ArrayAccessor;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::struct_::StructArrayExt;
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
fn with_slot_rewrites_chunk_and_offsets() {
    let array = chunked_array().into_array();

    let replacement = buffer![10u64, 11, 12].into_array();
    let array = array.with_slot(1, replacement).unwrap();
    let array = array.as_::<Chunked>();

    assert_eq!(array.nchunks(), 3);
    assert_eq!(array.chunk_offsets(), [0, 3, 6, 9]);
    assert_arrays_eq!(
        array.chunk(0).clone(),
        PrimitiveArray::from_iter([10u64, 11, 12])
    );
    assert_arrays_eq!(
        array.array().clone(),
        PrimitiveArray::from_iter([10u64, 11, 12, 4, 5, 6, 7, 8, 9])
    );
}

#[test]
fn with_slot_rejects_len_mismatch() {
    let err = chunked_array()
        .into_array()
        .with_slot(1, buffer![10u64, 11].into_array())
        .unwrap_err();

    assert!(err.to_string().contains("physical rewrite"));
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

    assert_eq!(
        l1.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        canon_values
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert_eq!(
        l2.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        canon_values
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
}
