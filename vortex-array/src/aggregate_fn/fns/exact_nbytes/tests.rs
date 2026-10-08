// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::exact_nbytes;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::validity::Validity;

const LONG: &str = "a string that is much longer than twelve bytes";

fn exact(array: &ArrayRef) -> VortexResult<u64> {
    let mut ctx = array_session().create_execution_ctx();
    exact_nbytes(array, &mut ctx)
}

fn long_strings(n: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..n).map(|i| format!("{LONG} #{i:06}"))).into_array()
}

/// A freshly built array references every byte it owns, so exact and approximate sizes agree.
#[rstest]
#[case::null(NullArray::new(7).into_array())]
#[case::primitive(PrimitiveArray::new(buffer![1i64, 2, 3, 4], Validity::NonNullable).into_array())]
#[case::nullable_primitive(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array())]
#[case::bool(BoolArray::from_iter([true, false, true, true, false, true, false, true]).into_array())]
#[case::decimal(DecimalArray::new(buffer![1i64, -2, 3], DecimalDType::new(10, 2), Validity::NonNullable).into_array())]
#[case::short_strings(VarBinViewArray::from_iter_str(["a", "bb", "ccc"]).into_array())]
#[case::long_strings(long_strings(100))]
#[case::fixed_size_list(FixedSizeListArray::new(
    PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array(),
    3,
    Validity::NonNullable,
    2,
).into_array())]
fn compact_arrays_match_nbytes(#[case] array: ArrayRef) -> VortexResult<()> {
    assert_eq!(exact(&array)?, array.nbytes());
    Ok(())
}

#[test]
fn sliced_strings_count_only_referenced_data() -> VortexResult<()> {
    let array = long_strings(10_000);
    let slice = array.slice(10..13)?;

    let value_len = (LONG.len() + " #000010".len()) as u64;
    assert_eq!(exact(&slice)?, 3 * 16 + 3 * value_len);
    // The slice still holds the parent's entire data buffer.
    assert!(slice.nbytes() > 10_000 * value_len);
    Ok(())
}

#[test]
fn overlapping_views_count_shared_bytes_once() -> VortexResult<()> {
    let data = ByteBuffer::copy_from(vec![b'x'; 100]);
    let view = BinaryView::make_view(&data[0..50], 0, 0);
    let views: Buffer<BinaryView> = [view, view, view].into_iter().collect();
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::try_new(
        views,
        Arc::from([data]),
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();

    assert_eq!(exact(&array)?, 3 * 16 + 50);
    assert_eq!(array.nbytes(), 3 * 16 + 100);
    Ok(())
}

#[test]
fn same_allocation_through_two_buffers_counts_once() -> VortexResult<()> {
    let data = ByteBuffer::copy_from(vec![b'y'; 100]);
    let views: Buffer<BinaryView> = [
        BinaryView::make_view(&data[0..50], 0, 0),
        BinaryView::make_view(&data[0..50], 1, 0),
    ]
    .into_iter()
    .collect();
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::try_new(
        views,
        Arc::from([data.clone(), data]),
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();

    assert_eq!(exact(&array)?, 2 * 16 + 50);
    assert_eq!(array.nbytes(), 2 * 16 + 200);
    Ok(())
}

#[test]
fn null_views_do_not_reference_data() -> VortexResult<()> {
    let data = ByteBuffer::copy_from(vec![b'z'; 40]);
    let views: Buffer<BinaryView> = [
        BinaryView::make_view(&data[0..20], 0, 0),
        BinaryView::make_view(&data[20..40], 0, 20),
    ]
    .into_iter()
    .collect();
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::try_new(
        views,
        Arc::from([data]),
        DType::Utf8(Nullability::Nullable),
        Validity::from_iter([true, false]),
        &mut ctx,
    )?
    .into_array();

    // Two views, the valid value's 20 bytes, and a one-byte validity bitmap.
    assert_eq!(exact(&array)?, 2 * 16 + 20 + 1);
    Ok(())
}

#[test]
fn sliced_list_view_counts_only_referenced_elements() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter(0i64..1000).into_array();
    let offsets = PrimitiveArray::from_iter((0u32..100).map(|i| i * 10)).into_array();
    let sizes = PrimitiveArray::from_iter(std::iter::repeat_n(10u32, 100)).into_array();
    let array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();
    let slice = array.slice(5..7)?;

    // Two u32 offsets, two u32 sizes, and twenty referenced i64 elements.
    assert_eq!(exact(&slice)?, 2 * 4 + 2 * 4 + 20 * 8);
    assert_eq!(slice.nbytes(), 2 * 4 + 2 * 4 + 1000 * 8);
    Ok(())
}

#[test]
fn overlapping_and_disjoint_lists() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter(0i32..100).into_array();
    // [0, 10) and [5, 15) overlap; [50, 60) is disjoint; the rest of the elements are unused.
    let offsets = buffer![0u32, 5, 50].into_array();
    let sizes = buffer![10u32, 10, 10].into_array();
    let array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable).into_array();

    assert_eq!(exact(&array)?, 3 * 4 + 3 * 4 + 25 * 4);
    Ok(())
}

#[test]
fn null_lists_do_not_reference_elements() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter(0i32..20).into_array();
    let offsets = buffer![0u32, 10].into_array();
    let sizes = buffer![10u32, 10].into_array();
    let array = ListViewArray::new(elements, offsets, sizes, Validity::from_iter([true, false]))
        .into_array();

    assert_eq!(exact(&array)?, 2 * 4 + 2 * 4 + 10 * 4 + 1);
    Ok(())
}

#[test]
fn constant_string_value_is_stored_once() -> VortexResult<()> {
    let array = ConstantArray::new(LONG, 1000).into_array();
    assert_eq!(exact(&array)?, 1000 * 16 + LONG.len() as u64);
    Ok(())
}

#[test]
fn constant_primitive_counts_every_slot() -> VortexResult<()> {
    let array = ConstantArray::new(7i32, 1000).into_array();
    assert_eq!(exact(&array)?, 1000 * 4);
    Ok(())
}

#[test]
fn struct_sums_sliced_fields() -> VortexResult<()> {
    let strings = long_strings(1000);
    let ints = PrimitiveArray::from_iter(0i64..1000).into_array();
    let array = StructArray::try_new(
        FieldNames::from(["s", "i"]),
        vec![strings.clone(), ints.clone()],
        1000,
        Validity::NonNullable,
    )?
    .into_array();
    let slice = array.slice(100..200)?;

    let expected = exact(&strings.slice(100..200)?)? + exact(&ints.slice(100..200)?)?;
    assert_eq!(exact(&slice)?, expected);
    assert!(slice.nbytes() > expected);
    Ok(())
}

#[test]
fn chunked_slices_of_one_buffer_count_each_referenced_range() -> VortexResult<()> {
    let strings = long_strings(1000);
    let chunked = ChunkedArray::try_new(
        vec![strings.slice(0..10)?, strings.slice(500..510)?],
        strings.dtype().clone(),
    )?
    .into_array();

    let expected = exact(&strings.slice(0..10)?)? + exact(&strings.slice(500..510)?)?;
    assert_eq!(exact(&chunked)?, expected);
    Ok(())
}
