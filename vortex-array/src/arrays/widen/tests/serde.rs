// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization through existing wire forms, including nested arrays and stored statistics.
//!
//! Wire IDs and logical values are checked separately because a `Widen` can read back as another
//! encoding.

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
use crate::ArrayId;
use crate::ArrayRef;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::SliceArray;
use crate::arrays::StructArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::session::ArraySessionExt;
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

    let ctx = ArrayContext::empty();
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

#[test]
fn serialization_requires_a_ready_decimal_leaf() -> VortexResult<()> {
    let child = DecimalArray::from_iter([1i32, 2, 3], DecimalDType::new(9, 2)).into_array();
    let dict = DictArray::try_new(buffer![0u8, 1, 2].into_array(), child)?;
    let widen = widen_over(
        dict,
        DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
    )?;
    let error = serialize_blob(&widen, &ArrayContext::empty(), &SESSION)
        .expect_err("serialization requires a ready, serializable Decimal leaf");
    assert!(error.to_string().contains("requires a Decimal leaf"));

    let integer = widen_over(buffer![1i8, 2, 3], DType::from(PType::I64))?;
    let error = serialize_blob(&integer, &ArrayContext::empty(), &SESSION)
        .expect_err("WidenPlugin only serializes decimal arrays");
    assert!(error.to_string().contains("requires a decimal dtype"));

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
