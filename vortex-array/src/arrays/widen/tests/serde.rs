// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the `Widen` wire formats.
//!
//! A decimal leaf can use the `vortex.decimal` wire ID while preserving the values of the original
//! `Widen`. The tests check the wire ID and the decoded values separately.

use rstest::rstest;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::SESSION;
use crate::ArrayContext;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::SliceArray;
use crate::arrays::StructArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::WidenPlugin;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::session::ArraySessionExt;
use crate::stats::Precision;
use crate::stats::Stat;
use crate::stats::StatsProvider;
use crate::validity::Validity;

fn serialize_blob(
    array: &ArrayRef,
    ctx: &ArrayContext,
    session: &VortexSession,
) -> VortexResult<ByteBuffer> {
    let mut blob = ByteBufferMut::empty();
    for buffer in array.serialize(ctx, session, &SerializeOptions::default())? {
        blob.extend_from_slice(buffer.as_ref());
    }

    Ok(blob.freeze())
}

/// Returns a `Widen` of the logical dtype `logical` over `values`, with the nullability of
/// `values`.
fn widen_over(values: impl IntoArray, logical: DType) -> VortexResult<ArrayRef> {
    let values = values.into_array();
    let logical = logical.with_nullability(values.dtype().nullability());

    Ok(WidenArray::try_new(values, logical)?.into_array())
}

/// Serializes `array` with an unrestricted context and decodes it again. Returns the decoded
/// array and the IDs that the context interned.
fn serialize_round_trip(
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<(ArrayRef, Vec<ArrayId>)> {
    let ctx = ArrayContext::empty();
    let serialized = SerializedArray::try_from(serialize_blob(array, &ctx, session)?)?;
    let ids = ctx.to_ids();
    let decoded = serialized.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(ids.clone()),
        session,
    )?;

    Ok((decoded, ids))
}

#[track_caller]
fn assert_widen_decode_error(parts: ArrayDeserialization<'_>, expected_error: &str) {
    let error = WidenPlugin
        .deserialize(parts, &SESSION)
        .expect_err("invalid Widen nodes must fail before decoding the child");
    assert!(
        error.to_string().contains(expected_error),
        "expected {expected_error}, got {error}"
    );
}

#[test]
fn decimal_widen_writes_narrow_bytes() -> VortexResult<()> {
    let values = [-999_999_999i32, 0, 999_999_999];
    let child = DecimalArray::from_iter(values, DecimalDType::new(9, 2));
    let child_ptr = child.buffer::<i32>().as_ptr();
    let logical_dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
    let widen = widen_over(child, logical_dtype.clone())?;
    let serialization = SESSION
        .array_serialize(&widen)?
        .vortex_expect("WidenPlugin serializes a decimal Widen over a Decimal leaf");
    assert_eq!(serialization.buffers[0].as_ptr(), child_ptr.cast::<u8>());

    let ctx =
        ArrayContext::empty().with_allowed_ids([ArrayVTable::id(&Decimal)].into_iter().collect());
    let serialized = SerializedArray::try_from(serialize_blob(&widen, &ctx, &SESSION)?)?;
    let read_ctx = ReadContext::new(ctx.to_ids());

    assert_eq!(
        read_ctx.resolve(serialized.encoding_id()),
        Some(ArrayVTable::id(&Decimal))
    );
    assert_eq!(serialized.nbuffers(), 1);
    let buffer = serialized.buffer(0)?;
    assert_eq!(buffer.len(), 4 * widen.len());
    let expected_bytes = values
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(buffer.as_host().as_slice(), expected_bytes);
    let decoded = serialized.decode(widen.dtype(), widen.len(), &read_ctx, &SESSION)?;
    assert_eq!(decoded.dtype(), &logical_dtype);
    assert_arrays_eq!(decoded, widen, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[rstest]
#[case::nullable(DecimalArray::from_option_iter(
    [Some(-999_999_999i32), None, Some(42)], DecimalDType::new(9, 2),
), 38)]
#[case::all_null(DecimalArray::new(
    buffer![i32::MIN, i32::MAX], DecimalDType::new(9, 2), Validity::AllInvalid,
    &mut SESSION.create_execution_ctx(),
), 38)]
#[case::empty(DecimalArray::from_iter(Vec::<i32>::new(), DecimalDType::new(9, 2)), 38)]
#[case::i128_child(DecimalArray::from_iter(
    [i128::MIN / 2, 0, i128::MAX / 2], DecimalDType::new(38, 0),
), 76)]
fn decimal_widen_round_trip(
    #[case] child: DecimalArray,
    #[case] precision: u8,
) -> VortexResult<()> {
    let scale = child.decimal_dtype().scale();
    let widen = widen_over(
        child,
        DType::Decimal(
            DecimalDType::new(precision, scale),
            Nullability::NonNullable,
        ),
    )?;
    let (decoded, ids) = serialize_round_trip(&widen, &SESSION)?;

    assert!(!ids.contains(&ArrayVTable::id(&Widen)));
    assert_arrays_eq!(decoded, widen, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn decimal_widen_nested_round_trip() -> VortexResult<()> {
    let child = DecimalArray::from_option_iter(
        [Some(-999_999_999i32), None, Some(42)],
        DecimalDType::new(9, 2),
    );
    let widen = widen_over(
        child,
        DType::Decimal(DecimalDType::new(38, 2), Nullability::Nullable),
    )?;
    let nested = StructArray::from_fields(&[("decimal", widen.clone())])?.into_array();
    let chunked = ChunkedArray::try_new(vec![widen.clone(), widen.clone()], widen.dtype().clone())?
        .into_array();

    for array in [nested, chunked] {
        let (decoded, ids) = serialize_round_trip(&array, &SESSION)?;
        assert!(!ids.contains(&ArrayVTable::id(&Widen)));
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
    }

    Ok(())
}

#[test]
fn serialization_keeps_parent_statistics() -> VortexResult<()> {
    let child = DecimalArray::from_iter([-3i32, 5, 1], DecimalDType::new(9, 2));
    let dtype = DecimalDType::new(38, 2);
    let widen = widen_over(child, DType::Decimal(dtype, Nullability::NonNullable))?;
    let min = Scalar::decimal((-3i128).into(), dtype, Nullability::NonNullable);
    let max = Scalar::decimal(5i128.into(), dtype, Nullability::NonNullable);

    for (stat, value) in [(Stat::Min, &min), (Stat::Max, &max)] {
        let value = value
            .value()
            .cloned()
            .vortex_expect("extrema are non-null decimal scalars");
        widen.statistics().set(stat, Precision::Exact(value));
    }

    let (decoded, _) = serialize_round_trip(&widen, &SESSION)?;
    assert_eq!(decoded.statistics().get(Stat::Min), Precision::Exact(min));
    assert_eq!(decoded.statistics().get(Stat::Max), Precision::Exact(max));

    Ok(())
}

#[rstest]
#[case::zero_ptype(
    buffer![0u8, u8::MAX].into_array(),
    DType::from(PType::U64),
    [8, 0], // The zero enum value for U8 still needs its field tag.
)]
#[case::encoded_decimal(
    DictArray::try_new(
        buffer![2u8, 0, 1, 2].into_array(),
        DecimalArray::from_option_iter([Some(-5i32), None, Some(8)], DecimalDType::new(9, 2))
            .into_array(),
    )?.into_array(),
    DType::Decimal(DecimalDType::new(38, 2), Nullability::Nullable),
    [16, 9],
)]
fn test_widen_wire_round_trip(
    #[case] child: ArrayRef,
    #[case] dtype: DType,
    #[case] metadata: [u8; 2],
) -> VortexResult<()> {
    let widen = widen_over(child.clone(), dtype)?;
    let ctx = ArrayContext::empty();
    let serialized = SerializedArray::try_from(serialize_blob(&widen, &ctx, &SESSION)?)?;
    let read_ctx = ReadContext::new(ctx.to_ids());

    assert_eq!(
        read_ctx.resolve(serialized.encoding_id()),
        Some(ArrayVTable::id(&Widen))
    );
    assert_eq!(serialized.metadata(), metadata.as_slice());
    assert_eq!(serialized.nbuffers(), 0);
    assert_eq!(serialized.nchildren(), 1);

    let decoded = serialized.decode(widen.dtype(), widen.len(), &read_ctx, &SESSION)?;
    let decoded_widen = decoded.as_::<Widen>();
    let values = decoded_widen.values();
    assert_eq!(values.dtype(), child.dtype());
    assert_eq!(values.encoding_id(), child.encoding_id());
    assert_arrays_eq!(decoded, widen, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn test_widen_wire_id_requires_permission() -> VortexResult<()> {
    let widen = widen_over(buffer![1i8, 2, 3], DType::from(PType::I64))?;
    let restricted =
        ArrayContext::empty().with_allowed_ids([ArrayVTable::id(&Primitive)].into_iter().collect());
    let error = serialize_blob(&widen, &restricted, &SESSION)
        .expect_err("the target must permit the Widen wire ID");
    assert!(error.to_string().contains("vortex.widen not permitted"));

    Ok(())
}

#[rstest]
#[case::missing(      PType::I64.into(), &[],        "must name the child type")]
#[case::unknown_ptype(PType::I64.into(), &[8, 127],  "unknown enumeration value 127")]
#[case::wrong_parent( PType::I64.into(), &[16, 9],   "does not match the parent")]
#[case::incompatible_child(
    PType::I64.into(),
    &[8, PType::U8 as u8],
    "signedness",
)]
#[case::precision_would_truncate(
    DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
    &[16, 137, 2], // A wrapping cast from 265 to u8 would accept precision 9.
    "out of range",
)]
#[case::precision_below_scale(
    DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
    &[16, 1],
    "scale",
)]
fn test_widen_wire_rejects_invalid_child_metadata(
    #[case] dtype: DType,
    #[case] metadata: &[u8],
    #[case] expected_error: &str,
) {
    let children = [buffer![1u32].into_array()];
    let parts =
        ArrayDeserialization::new(ArrayVTable::id(&Widen), &dtype, 1, metadata, &[], &children);

    assert_widen_decode_error(parts, expected_error);
}

#[test]
fn test_widen_wire_rejects_invalid_node() {
    let dtype = DType::from(PType::I64);
    let metadata = [8, PType::I8 as u8];
    let children = [buffer![1i8].into_array()];
    let cases = [
        (ArrayVTable::id(&Decimal), &children[..], "wire ID"),
        (
            ArrayVTable::id(&Widen),
            &children[..0],
            "at least one child",
        ),
    ];

    for (id, children, expected_error) in cases {
        let parts = ArrayDeserialization::new(id, &dtype, 1, &metadata, &[], &children);
        assert_widen_decode_error(parts, expected_error);
    }
}

/// A later writer may append children and node buffers that do not change the decoded values, so
/// a reader ignores everything after the first child.
#[test]
fn test_widen_wire_ignores_trailing_children_and_node_buffers() -> VortexResult<()> {
    let dtype = DType::from(PType::I64);
    let metadata = [8, PType::I8 as u8];
    let buffers = [BufferHandle::new_host(ByteBuffer::copy_from([1]))];
    // In-memory children assert the requested dtype, so decoding this `u32` child would panic.
    let children = [
        buffer![1i8, -2, 3].into_array(),
        buffer![7u32, 8, 9].into_array(),
    ];
    let parts = ArrayDeserialization::new(
        ArrayVTable::id(&Widen),
        &dtype,
        3,
        &metadata,
        &buffers,
        &children,
    );

    let decoded = WidenPlugin.deserialize(parts, &SESSION)?;
    let expected = widen_over(buffer![1i8, -2, 3], dtype)?;
    assert_arrays_eq!(decoded, expected, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn serialization_does_not_execute_validity() -> VortexResult<()> {
    let validity =
        SliceArray::new(BoolArray::from_iter([true, false, true]).into_array(), 0..2).into_array();
    let child = DecimalArray::new(
        buffer![1i32, 2],
        DecimalDType::new(9, 2),
        Validity::Array(validity),
        &mut SESSION.create_execution_ctx(),
    );
    let widen = widen_over(
        child,
        DType::Decimal(DecimalDType::new(38, 2), Nullability::Nullable),
    )?;
    let error = serialize_blob(&widen, &ArrayContext::empty(), &SESSION)
        .expect_err("serialization requires a ready, serializable Decimal leaf");
    assert!(error.to_string().contains("vortex.slice"));

    Ok(())
}
