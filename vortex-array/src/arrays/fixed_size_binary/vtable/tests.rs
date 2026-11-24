// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::compute::conformance::consistency::test_array_consistency;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::patches::Patches;
use crate::scalar::Scalar;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn nullable_array() -> FixedSizeBinaryArray {
    FixedSizeBinaryArray::from_option_iter(
        [Some(b"aaa"), None, Some(b"ccc"), Some(b"ddd"), None],
        3,
    )
}

#[test]
fn try_new_infers_length() -> VortexResult<()> {
    let array =
        FixedSizeBinaryArray::try_new(ByteBuffer::copy_from(b"abcdef"), 3, Validity::NonNullable)?;
    assert_eq!(array.len(), 2);
    assert_eq!(
        array.dtype(),
        &DType::FixedSizeBinary(3, Nullability::NonNullable)
    );
    assert_eq!(array.value(0), b"abc");
    assert_eq!(array.value(1), b"def");
    Ok(())
}

#[test]
fn try_new_rejects_partial_rows() {
    let error =
        FixedSizeBinaryArray::try_new(ByteBuffer::copy_from(b"abcde"), 3, Validity::NonNullable)
            .unwrap_err()
            .to_string();
    assert!(
        error.contains("not a whole number of 3-byte rows"),
        "{error}"
    );
}

#[test]
fn try_new_handle_rejects_wrong_buffer_length() {
    let result = FixedSizeBinaryArray::try_new_handle(
        BufferHandle::new_host(ByteBuffer::copy_from(b"abcd")),
        2,
        3,
        Validity::NonNullable,
    );
    assert!(result.is_err());
}

#[test]
fn try_new_rejects_validity_of_another_length() {
    let result = FixedSizeBinaryArray::try_new(
        ByteBuffer::copy_from(b"abcd"),
        2,
        Validity::from_iter([true, false, true]),
    );
    assert!(result.is_err());
}

#[test]
fn zero_width_length_comes_from_validity() -> VortexResult<()> {
    let array = FixedSizeBinaryArray::try_new(
        ByteBuffer::empty(),
        0,
        Validity::from_iter([true, false, true]),
    )?;
    assert_eq!(array.len(), 3);

    let array = FixedSizeBinaryArray::try_new_handle(
        BufferHandle::new_host(ByteBuffer::empty()),
        0,
        5,
        Validity::NonNullable,
    )?;
    assert_eq!(array.len(), 5);
    Ok(())
}

#[test]
fn scalar_at() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = nullable_array().into_array();
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        Scalar::fixed_size_binary(b"aaa".to_vec(), Nullability::Nullable)
    );
    assert_eq!(
        array.execute_scalar(1, &mut ctx)?,
        Scalar::null(DType::FixedSizeBinary(3, Nullability::Nullable))
    );
    Ok(())
}

#[test]
fn take() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let taken = nullable_array()
        .into_array()
        .take(buffer![3u32, 0, 1, 3].into_array())?;
    let expected =
        FixedSizeBinaryArray::from_option_iter([Some(b"ddd"), Some(b"aaa"), None, Some(b"ddd")], 3);
    assert_arrays_eq!(taken, expected, &mut ctx);
    Ok(())
}

#[test]
fn filter() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let filtered = nullable_array()
        .into_array()
        .filter(Mask::from_iter([true, true, false, true, false]))?;
    let expected = FixedSizeBinaryArray::from_option_iter([Some(b"aaa"), None, Some(b"ddd")], 3);
    assert_arrays_eq!(filtered, expected, &mut ctx);
    Ok(())
}

#[test]
fn slice() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let sliced = nullable_array().into_array().slice(1..4)?;
    let expected = FixedSizeBinaryArray::from_option_iter([None, Some(b"ccc"), Some(b"ddd")], 3);
    assert_arrays_eq!(sliced, expected, &mut ctx);
    Ok(())
}

#[test]
fn dict_values_take() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dict = DictArray::try_new(
        buffer![2u8, 0, 2].into_array(),
        FixedSizeBinaryArray::from_iter([b"xy", b"zw", b"uv"], 2).into_array(),
    )?;
    let expected = FixedSizeBinaryArray::from_iter([b"uv", b"xy", b"uv"], 2);
    assert_arrays_eq!(
        dict.into_array()
            .execute::<FixedSizeBinaryArray>(&mut ctx)?,
        expected,
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::value(
    Scalar::fixed_size_binary(b"ab".to_vec(), Nullability::NonNullable),
    FixedSizeBinaryArray::from_iter([b"ab"; 3], 2),
)]
#[case::nullable_value(
    Scalar::fixed_size_binary(b"ab".to_vec(), Nullability::Nullable),
    FixedSizeBinaryArray::from_option_iter([Some(b"ab"); 3], 2),
)]
#[case::null(
    Scalar::null(DType::FixedSizeBinary(2, Nullability::Nullable)),
    FixedSizeBinaryArray::from_option_iter([None::<&[u8]>; 3], 2),
)]
fn constant_canonicalizes(
    #[case] scalar: Scalar,
    #[case] expected: FixedSizeBinaryArray,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = ConstantArray::new(scalar.clone(), 3)
        .into_array()
        .execute::<FixedSizeBinaryArray>(&mut ctx)?;
    assert_eq!(array.dtype(), scalar.dtype());
    assert_arrays_eq!(array, expected, &mut ctx);
    Ok(())
}

#[test]
fn patch() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let patches = Patches::new(
        5,
        0,
        buffer![1u32, 4].into_array(),
        FixedSizeBinaryArray::from_option_iter([Some(b"xyz"), None], 3).into_array(),
        None,
        &mut ctx,
    )?;
    let patched = nullable_array().patch(&patches, &mut ctx)?;
    let expected = FixedSizeBinaryArray::from_option_iter(
        [Some(b"aaa"), Some(b"xyz"), Some(b"ccc"), Some(b"ddd"), None],
        3,
    );
    assert_arrays_eq!(patched, expected, &mut ctx);
    Ok(())
}

#[test]
fn masked_validity() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let masked = nullable_array()
        .into_array()
        .mask(BoolArray::from_iter([true, true, false, true, true]).into_array())?;
    let expected =
        FixedSizeBinaryArray::from_option_iter([Some(b"aaa"), None, None, Some(b"ddd"), None], 3);
    assert_arrays_eq!(masked, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::non_nullable(FixedSizeBinaryArray::from_iter(
    [b"aaaa", b"bbbb", b"cccc", b"dddd", b"eeee"],
    4,
).into_array())]
#[case::nullable(nullable_array().into_array())]
#[case::width_3(FixedSizeBinaryArray::from_iter(
    (0u8..50).map(|i| [i, i.wrapping_mul(3), 7]),
    3,
).into_array())]
#[case::width_16(FixedSizeBinaryArray::from_option_iter(
    (0u8..40).map(|i| (i % 5 != 0).then_some([i; 16])),
    16,
).into_array())]
#[case::zero_width(FixedSizeBinaryArray::from_option_iter(
    [Some(b""), None, Some(b""), Some(b"")],
    0,
).into_array())]
fn structural_conformance(#[case] array: ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    test_take_conformance(&array, &mut ctx);
    test_filter_conformance(&array, &mut ctx);
    test_mask_conformance(&array, &mut ctx);
}

#[rstest]
#[case::non_nullable(FixedSizeBinaryArray::from_iter(
    [b"aaaa", b"bbbb", b"aaab", b"dddd", b"eeee"],
    4,
).into_array())]
#[case::nullable(nullable_array().into_array())]
#[case::width_3(FixedSizeBinaryArray::from_iter(
    (0u8..50).map(|i| [i % 7, i.wrapping_mul(3), 7]),
    3,
).into_array())]
#[case::width_16(FixedSizeBinaryArray::from_option_iter(
    (0u8..40).map(|i| (i % 5 != 0).then_some([i % 9; 16])),
    16,
).into_array())]
fn consistency(#[case] array: ArrayRef) {
    test_array_consistency(&array, &mut SESSION.create_execution_ctx());
}
