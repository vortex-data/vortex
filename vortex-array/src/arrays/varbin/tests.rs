// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::fixture;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::offsets_tile_utf8;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::Scalar;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[fixture]
fn binary_array() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = Buffer::copy_from("hello worldhello world this is a long string".as_bytes());
    let offsets = buffer![0, 11, 44].into_array();

    VarBinArray::try_new(
        offsets.into_array(),
        values,
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    )
    .unwrap()
    .into_array()
}

#[rstest]
pub fn test_scalar_at(binary_array: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    for (index, value) in ["hello world", "hello world this is a long string"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            binary_array.execute_scalar(index, &mut ctx).unwrap(),
            Scalar::from(value)
        );
    }
    assert_arrays_eq!(
        binary_array,
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]),
        &mut ctx
    );
}

#[rstest]
pub fn slice_array(binary_array: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    let binary_arr = binary_array.slice(1..2).unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world this is a long string"]),
        &mut ctx
    );
}

#[rstest]
#[case::tiles_multibyte_strings(        vec![0, 1, 3, 6], "héllo".as_bytes(),  true)]
#[case::ignores_bytes_outside_the_range(vec![1, 2, 3],    &[0xff, b'a', b'b'], true)]
#[case::splits_a_char(                  vec![0, 2, 6],    "héllo".as_bytes(),  false)]
#[case::invalid_byte(                   vec![0, 1, 2],    &[b'a', 0xff],       false)]
#[case::decreasing(                     vec![0, 3, 1, 6], "héllo".as_bytes(),  false)]
#[case::past_the_end(                   vec![0, 7],       "héllo".as_bytes(),  false)]
#[case::no_offsets(                     vec![],           b"",                 false)]
fn test_offsets_tile_utf8(#[case] offsets: Vec<u32>, #[case] bytes: &[u8], #[case] expected: bool) {
    assert_eq!(offsets_tile_utf8(&offsets, bytes), expected);
}

#[test]
fn test_offsets_tile_utf8_rejects_negative_offsets() {
    assert!(!offsets_tile_utf8(&[-1i32, 0], b"a"));
}

#[rstest]
#[case::decreasing(buffer![0i32, 5, 3].into_array())]
#[case::negative_first(buffer![-1i32, 2, 3].into_array())]
#[case::beyond_bytes(buffer![0i32, 2, 6].into_array())]
fn try_new_rejects_invalid_offsets(#[case] offsets: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    let values = Buffer::copy_from("hello".as_bytes());

    let result = VarBinArray::try_new(
        offsets,
        values,
        DType::Binary(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    );

    assert!(result.is_err());
}

#[test]
fn try_new_allows_nonzero_first_offset() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = Buffer::copy_from("hello".as_bytes());
    let offsets = buffer![1i32, 3, 5].into_array();

    let array = VarBinArray::try_new(
        offsets,
        values,
        DType::Binary(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    )?;

    assert_arrays_eq!(
        array,
        VarBinViewArray::from_iter_bin([b"el", b"lo"]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn try_new_rejects_unsorted_utf8_offsets_at_null_row() {
    let mut ctx = SESSION.create_execution_ctx();
    // Every non-null string is valid UTF-8, so only the offsets check rejects the null row 3..1.
    let values = Buffer::copy_from("hello".as_bytes());
    let offsets = buffer![0i32, 3, 1, 5].into_array();

    let result = VarBinArray::try_new(
        offsets,
        values,
        DType::Utf8(Nullability::Nullable),
        Validity::from_iter([true, false, true]),
        &mut ctx,
    );

    assert!(result.is_err());
}

#[test]
fn try_new_rejects_invalid_utf8_at_null_row() {
    let mut ctx = SESSION.create_execution_ctx();
    // The offsets are valid and each non-null string is valid UTF-8. Only the null row holds
    // bytes that are not valid UTF-8.
    let values = Buffer::copy_from([b'a', 0xFF, b'b']);
    let offsets = buffer![0i32, 1, 2, 3].into_array();

    let result = VarBinArray::try_new(
        offsets,
        values,
        DType::Utf8(Nullability::Nullable),
        Validity::from_iter([true, false, true]),
        &mut ctx,
    );

    assert!(result.is_err_and(|err| err.to_string().contains("invalid utf-8")));
}

/// Sets statistics that say `offsets` are sorted and in the range `0..=max`.
fn forge_offsets_statistics(offsets: &ArrayRef, max: i32) {
    offsets.statistics().set(Stat::IsSorted, Precision::exact(true));
    offsets.statistics().set(Stat::Min, Precision::exact(0i32));
    offsets.statistics().set(Stat::Max, Precision::exact(max));
}

#[test]
fn try_new_ignores_forged_offsets_statistics() {
    let mut ctx = SESSION.create_execution_ctx();
    let values = Buffer::copy_from("abcd".as_bytes());
    let offsets = buffer![0i32, 1000, 2].into_array();
    forge_offsets_statistics(&offsets, 4);

    let result = VarBinArray::try_new(
        offsets,
        values,
        DType::Binary(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    );

    assert!(result.is_err_and(|err| err.to_string().contains("sorted")));
}

#[test]
fn try_new_replaces_forged_offsets_statistics_of_valid_utf8() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let offsets = buffer![0i32, 2, 4].into_array();
    forge_offsets_statistics(&offsets, 1);

    VarBinArray::try_new(
        offsets.clone(),
        Buffer::copy_from("abcd".as_bytes()),
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
        &mut ctx,
    )?;

    assert_eq!(
        offsets.statistics().get_as::<i32>(Stat::Max),
        Precision::exact(4i32)
    );
    Ok(())
}

#[test]
fn decode_rejects_offsets_with_forged_statistics() -> VortexResult<()> {
    let dtype = DType::Binary(Nullability::NonNullable);
    let offsets = buffer![0i32, 1000, 2].into_array();
    forge_offsets_statistics(&offsets, 4);

    // SAFETY: the offsets are not valid on purpose. The test only serializes this array.
    let array = unsafe {
        VarBinArray::new_unchecked(
            offsets,
            Buffer::copy_from("abcd".as_bytes()),
            dtype.clone(),
            Validity::NonNullable,
        )
    }
    .into_array();

    let array_ctx = ArrayContext::empty();
    let mut blob = ByteBufferMut::empty();
    for buffer in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        blob.extend_from_slice(buffer.as_ref());
    }
    let serialized = SerializedArray::try_from(blob.freeze())?;

    let result = serialized.decode(
        &dtype,
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    );

    assert!(result.is_err_and(|err| err.to_string().contains("sorted")));
    Ok(())
}
