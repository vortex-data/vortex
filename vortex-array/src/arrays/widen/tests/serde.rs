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
use crate::arrays::Bool;
use crate::arrays::ChunkedArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenPlugin;
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
use crate::session::ArraySession;
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

#[rstest]
#[case::integer(widen_over(buffer![-1i8, 0, 100], DType::from(PType::I64)))]
#[case::integer_nullable(widen_over(
    PrimitiveArray::from_option_iter([Some(-300i16), None, Some(300)]),
    DType::from(PType::I32),
))]
#[case::empty(widen_over(
    PrimitiveArray::from_iter(Vec::<i16>::new()),
    DType::from(PType::I64),
))]
#[case::all_null(widen_over(
    PrimitiveArray::new(buffer![i8::MIN, i8::MAX], Validity::AllInvalid),
    DType::from(PType::I64),
))]
fn serialization_writes_widen(#[case] widen: VortexResult<ArrayRef>) -> VortexResult<()> {
    let widen = widen?;
    assert!(widen.is::<Widen>());

    let (decoded, ids) = serialize_round_trip(&widen, &SESSION)?;

    assert!(decoded.is::<Primitive>(), "{}", decoded.encoding_id());
    assert!(!ids.contains(&ArrayVTable::id(&Widen)), "{ids:?}");
    assert_arrays_eq!(decoded, widen, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn serialization_writes_nested_widen() -> VortexResult<()> {
    let integer = widen_over(
        PrimitiveArray::from_option_iter([Some(1i8), None, Some(-7)]),
        DType::from(PType::I64),
    )?;
    let nested = StructArray::from_fields(&[("integer", integer.clone())])?.into_array();
    let chunked = ChunkedArray::try_new(
        vec![
            integer,
            PrimitiveArray::from_option_iter([Some(1000i64)]).into_array(),
        ],
        DType::Primitive(PType::I64, Nullability::Nullable),
    )?
    .into_array();

    for array in [nested, chunked] {
        let (decoded, ids) = serialize_round_trip(&array, &SESSION)?;
        assert!(!ids.contains(&ArrayVTable::id(&Widen)), "{ids:?}");
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
    }

    Ok(())
}

#[test]
fn serialization_keeps_widen_statistics() -> VortexResult<()> {
    let widen = widen_over(buffer![-3i8, 5, 1], DType::from(PType::I64))?;
    let min = Scalar::primitive(-3i64, Nullability::NonNullable);
    let max = Scalar::primitive(5i64, Nullability::NonNullable);

    for (stat, value) in [(Stat::Min, &min), (Stat::Max, &max)] {
        let value = value.value().cloned();
        let value = value.vortex_expect("the planted statistic is not null");
        widen.statistics().set(stat, Precision::Exact(value));
    }

    let (decoded, _) = serialize_round_trip(&widen, &SESSION)?;

    assert_eq!(decoded.statistics().get(Stat::Min), Precision::Exact(min));
    assert_eq!(decoded.statistics().get(Stat::Max), Precision::Exact(max));

    Ok(())
}

#[test]
fn serialization_writes_widen_with_restricted_context() -> VortexResult<()> {
    let ctx = ArrayContext::new(vec![ArrayVTable::id(&Primitive)])
        .with_allowed_ids([ArrayVTable::id(&Primitive)].into_iter().collect());
    let widen = widen_over(buffer![-1i8, 0, 100], DType::from(PType::I64))?;

    let serialized = SerializedArray::try_from(serialize_blob(&widen, &ctx, &SESSION)?)?;
    let read_ctx = ReadContext::new(ctx.to_ids());
    let decoded = serialized.decode(widen.dtype(), widen.len(), &read_ctx, &SESSION)?;

    assert!(decoded.is::<Primitive>());
    assert_arrays_eq!(decoded, widen, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn empty_session_can_register_widen_writer() -> VortexResult<()> {
    let session = VortexSession::empty().with_some(ArraySession::empty());
    session.arrays().register(Primitive);
    session.arrays().register(Bool);
    session.arrays().register(WidenPlugin);
    let widen = widen_over(buffer![-1i8, 0, 100], DType::from(PType::I64))?;

    let (decoded, ids) = serialize_round_trip(&widen, &session)?;

    assert!(
        !session
            .arrays()
            .registry()
            .contains_key(&ArrayVTable::id(&Widen))
    );
    assert!(!ids.contains(&ArrayVTable::id(&Widen)));
    assert_arrays_eq!(decoded, widen, &mut session.create_execution_ctx());

    Ok(())
}

#[test]
fn decimal_widen_writes_narrow_bytes() -> VortexResult<()> {
    let values = [-999_999_999i32, 0, 999_999_999];
    let child = DecimalArray::from_iter(values, DecimalDType::new(9, 2));
    let widen = widen_over(
        child,
        DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
    )?;
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
    let scale = child.dtype().as_decimal_opt().unwrap().scale();
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
