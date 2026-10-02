// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::Narrow;
use super::NarrowArray;
use super::NarrowArraySlotsExt;
use crate::ArrayContext;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

fn roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let buffers = array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(buffer.as_ref());
    }

    SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )
}

#[rstest]
#[case::i8_limits(vec![-128i64, 127], PType::I8)]
#[case::i16_lower(vec![-129i64, 0], PType::I16)]
#[case::i16_upper(vec![0i64, 128], PType::I16)]
#[case::i32_lower(vec![-32769i64, 0], PType::I32)]
#[case::i32_upper(vec![0i64, 32768], PType::I32)]
#[case::i64_lower(vec![i64::from(i32::MIN) - 1, 0], PType::I64)]
#[case::i64_upper(vec![0i64, i64::from(i32::MAX) + 1], PType::I64)]
#[case::positive_signed(vec![0i64, 255], PType::I16)]
fn test_signed_encoding(#[case] values: Vec<i64>, #[case] storage: PType) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter(values);
    let encoded = NarrowArray::encode(input.clone(), &mut ctx)?;
    let values = encoded.as_opt::<Narrow>().map_or(&encoded, |array| array.values());

    assert_eq!(values.dtype().as_ptype(), storage);
    assert_eq!(encoded.dtype(), input.dtype());
    assert_arrays_eq!(encoded, input.into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::u8(vec![0u64, 255], PType::U8)]
#[case::u16(vec![256u64, 65535], PType::U16)]
#[case::u32(vec![65536u64, u64::from(u32::MAX)], PType::U32)]
#[case::u64(vec![u64::from(u32::MAX) + 1, u64::MAX], PType::U64)]
fn test_unsigned_encoding(#[case] values: Vec<u64>, #[case] storage: PType) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter(values);
    let encoded = NarrowArray::encode(input.clone(), &mut ctx)?;
    let values = encoded.as_opt::<Narrow>().map_or(&encoded, |array| array.values());

    assert_eq!(values.dtype().as_ptype(), storage);
    assert_eq!(encoded.dtype(), input.dtype());
    assert_arrays_eq!(encoded, input.into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::empty(PrimitiveArray::new(Buffer::<i64>::empty(), Validity::NonNullable))]
#[case::all_null(PrimitiveArray::new(buffer![i64::MIN, i64::MAX], Validity::AllInvalid))]
#[case::null_payloads(PrimitiveArray::new(
    buffer![i64::MIN, 42i64, i64::MAX],
    Validity::from_iter([false, true, false]),
))]
fn test_null_payloads_and_empty(#[case] input: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = NarrowArray::encode(input.clone(), &mut ctx)?;

    assert_eq!(encoded.as_::<Narrow>().values().dtype().as_ptype(), PType::I8);
    assert_arrays_eq!(encoded, input.into_array(), &mut ctx);

    Ok(())
}

#[rstest]
#[case::equal_width(PType::I8, PType::I8)]
#[case::wider_storage(PType::I16, PType::I8)]
#[case::signedness(PType::I8, PType::U64)]
#[case::float_storage(PType::F32, PType::I64)]
#[case::float_logical(PType::I8, PType::F64)]
fn test_reject_invalid_dtypes(#[case] storage: PType, #[case] logical: PType) {
    let child = crate::Canonical::empty(&storage.into()).into_array();
    assert!(NarrowArray::try_new(child, logical.into()).is_err());
}

#[test]
fn test_validate_slots() {
    assert!(
        NarrowArray::try_new(
            buffer![1i8].into_array(),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .is_err()
    );
    for slots in [
        vec![],
        vec![None],
        vec![Some(buffer![1i8].into_array()), None],
    ] {
        assert!(
            NarrowArray::try_from_parts(
                ArrayParts::new(Narrow, PType::I64.into(), 1, EmptyArrayData)
                    .with_slots(slots.into()),
            )
            .is_err()
        );
    }
    assert!(
        NarrowArray::try_from_parts(
            ArrayParts::new(Narrow, PType::I64.into(), 2, EmptyArrayData)
                .with_slots(vec![Some(buffer![1i8].into_array())].into()),
        )
        .is_err()
    );
}

#[test]
fn test_flatten_and_scalar_dtype() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = PrimitiveArray::from_option_iter([Some(-128i8), None, Some(127)]).into_array();
    let inner =
        NarrowArray::try_new(child, DType::Primitive(PType::I16, Nullability::Nullable))?;
    let outer = NarrowArray::try_new(
        inner.into_array(),
        DType::Primitive(PType::I64, Nullability::Nullable),
    )?;

    assert_eq!(outer.values().dtype().as_ptype(), PType::I8);
    assert_eq!(
        outer.as_ref().execute_scalar(0, &mut ctx)?,
        Scalar::primitive(-128i64, Nullability::Nullable)
    );
    assert_eq!(
        outer.as_ref().execute_scalar(1, &mut ctx)?,
        Scalar::null(outer.dtype().clone())
    );

    Ok(())
}

#[test]
fn test_selection_and_fill_null() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = PrimitiveArray::from_option_iter([Some(-12i8), None, Some(34), Some(56)]).into_array();
    let array =
        NarrowArray::try_new(child, DType::Primitive(PType::I64, Nullability::Nullable))?
            .into_array();
    let selected = array
        .slice(1..4)?
        .take(buffer![2u32, 0, 1].into_array())?
        .filter(Mask::from_iter([true, false, true]))?
        .optimize()?;

    assert!(selected.is::<Narrow>());
    assert_eq!(selected.as_::<Narrow>().values().dtype().as_ptype(), PType::I8);
    assert_arrays_eq!(
        selected,
        buffer![56i64, 34].into_array().cast(array.dtype().clone())?,
        &mut ctx
    );

    let masked = array
        .clone()
        .mask(buffer![true, true, false, true].into_array())?
        .optimize()?;
    assert!(masked.is::<Narrow>());
    assert_arrays_eq!(
        masked,
        PrimitiveArray::from_option_iter([Some(-12i64), None, None, Some(56)]),
        &mut ctx
    );

    let filled = array.fill_null(Scalar::from(100i64))?;
    assert!(filled.is::<Narrow>());
    assert_arrays_eq!(filled, buffer![-12i64, 100, 34, 56].into_array(), &mut ctx);
    assert_arrays_eq!(
        array.fill_null(Scalar::from(1000i64))?,
        buffer![-12i64, 1000, 34, 56].into_array(),
        &mut ctx
    );

    Ok(())
}

#[test]
fn test_cast_preserves_checked_nullability() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = PrimitiveArray::from_option_iter([Some(1i8), None]).into_array();
    let array =
        NarrowArray::try_new(child, DType::Primitive(PType::I32, Nullability::Nullable))?
            .into_array();
    let widened = array.cast(DType::Primitive(PType::I64, Nullability::Nullable))?;
    assert!(widened.is::<Narrow>());
    assert!(
        array
            .cast(PType::I64.into())?
            .execute::<PrimitiveArray>(&mut ctx)
            .is_err()
    );
    assert_arrays_eq!(
        array.cast(DType::Primitive(PType::I8, Nullability::Nullable))?,
        PrimitiveArray::from_option_iter([Some(1i8), None]),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case(Nullability::NonNullable)]
#[case(Nullability::Nullable)]
fn test_fill_null_preserves_result_dtype(#[case] nullability: Nullability) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = PrimitiveArray::from_option_iter([Some(1i8), None]).into_array();
    let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    let array = NarrowArray::try_new(child, dtype)?.into_array();
    let filled = array.fill_null(Scalar::primitive(2i64, nullability))?;
    let expected_dtype = DType::Primitive(PType::I64, nullability);

    assert_eq!(filled.dtype(), &expected_dtype);
    assert!(filled.is::<Narrow>());
    assert_arrays_eq!(
        filled,
        buffer![1i64, 2].into_array().cast(expected_dtype)?,
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::eq(Operator::Eq)]
#[case::ne(Operator::NotEq)]
#[case::lt(Operator::Lt)]
#[case::le(Operator::Lte)]
#[case::gt(Operator::Gt)]
#[case::ge(Operator::Gte)]
fn test_comparison(#[case] operator: Operator) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter([Some(-128i64), None, Some(0), Some(127)]).into_array();
    let array = NarrowArray::encode(
        input.clone().execute::<PrimitiveArray>(&mut ctx)?,
        &mut ctx,
    )?;
    for value in [-129i64, -128, 0, 127, 128] {
        let constant = ConstantArray::new(Scalar::from(value), input.len()).into_array();
        assert_arrays_eq!(
            array.binary(constant.clone(), operator)?,
            input.binary(constant.clone(), operator)?,
            &mut ctx
        );
        assert_arrays_eq!(
            constant.binary(array.clone(), operator)?,
            constant.binary(input.clone(), operator)?,
            &mut ctx
        );
    }

    let rhs = NarrowArray::try_new(
        buffer![0i16, 300, -300, 127].into_array(),
        PType::I64.into(),
    )?
    .into_array();
    let expected_rhs = rhs.clone().execute::<PrimitiveArray>(&mut ctx)?.into_array();
    assert_arrays_eq!(
        array.binary(rhs, operator)?,
        input.binary(expected_rhs, operator)?,
        &mut ctx
    );

    Ok(())
}

#[test]
fn test_aggregate_partial_states() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let batches = [
        vec![Some(-128i64), None, Some(42)],
        vec![Some(127i64), None],
        vec![None, None],
    ];
    let fns = [
        MinMax.bind(NumericalAggregateOpts::default()),
        Sum.bind(NumericalAggregateOpts::default()),
        SumV2.bind(NumericalAggregateOpts::default()),
        IsConstant.bind(crate::aggregate_fn::EmptyOptions),
        IsSorted.bind(IsSortedOptions::default()),
    ];
    for aggregate in fns {
        let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let mut expected = aggregate.accumulator(&dtype)?;
        let mut actual = aggregate.accumulator(&dtype)?;
        for values in &batches {
            let input = PrimitiveArray::from_option_iter(values.clone());
            let encoded = NarrowArray::encode(input.clone(), &mut ctx)?;
            expected.accumulate(&input.into_array(), &mut ctx)?;
            actual.accumulate(&encoded, &mut ctx)?;
            assert_eq!(actual.partial_scalar()?, expected.partial_scalar()?);
        }
        assert_eq!(actual.final_scalar()?, expected.final_scalar()?);
    }

    Ok(())
}

#[test]
fn test_encoded_child_serde() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = DictArray::try_new(
        buffer![2u8, 0, 1, 2].into_array(),
        buffer![-1i8, 0, 1].into_array(),
    )?
    .into_array();
    let array = NarrowArray::try_new(values, PType::I64.into())?.into_array();
    let decoded = roundtrip(&array)?;
    assert!(decoded.is::<Narrow>());
    assert_eq!(decoded.as_::<Narrow>().values().dtype().as_ptype(), PType::I8);
    assert_arrays_eq!(decoded.clone(), array, &mut ctx);
    assert_eq!(
        decoded.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<i64>(),
        &[1, -1, 0, 1]
    );

    Ok(())
}

#[test]
fn test_arithmetic_uses_logical_width() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = NarrowArray::try_new(buffer![120i8, -120].into_array(), PType::I64.into())?
        .into_array();
    let rhs = ConstantArray::new(Scalar::from(20i64), 2).into_array();

    assert_arrays_eq!(
        array.binary(rhs, Operator::Add)?,
        buffer![140i64, -100].into_array(),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::missing(vec![])]
#[case::extra(vec![PType::I8 as u8, 0])]
#[case::unknown(vec![255])]
#[case::float(vec![PType::F32 as u8])]
#[case::unsigned(vec![PType::U8 as u8])]
#[case::equal_width(vec![PType::I64 as u8])]
fn test_reject_malformed_metadata(#[case] metadata: Vec<u8>) {
    let values = [buffer![1i8].into_array()];
    assert!(
        Narrow
            .deserialize(&PType::I64.into(), 1, &metadata, &[], &values, &SESSION)
            .is_err()
    );
}
