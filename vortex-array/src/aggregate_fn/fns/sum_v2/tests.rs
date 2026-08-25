// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_proto::expr as pb;

use super::SumV2;
use super::sum_v2;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::sum::Sum;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

fn sum_with_options(array: &ArrayRef, options: NumericalAggregateOpts) -> VortexResult<Scalar> {
    let mut accumulator = Accumulator::try_new(SumV2, options, array.dtype().clone())?;
    accumulator.accumulate(array, &mut array_session().create_execution_ctx())?;
    accumulator.finish()
}

#[test]
fn distinct_id_and_struct_partial() -> VortexResult<()> {
    let options = NumericalAggregateOpts::default();
    let aggregate = SumV2.bind(options);
    assert_eq!(aggregate.id().as_ref(), "vortex.sum_v2");

    let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let partial_dtype = SumV2
        .partial_dtype(&options, &input_dtype)
        .expect("supported sum_v2 dtype");
    assert_eq!(partial_dtype.nullability(), Nullable);
    let fields = partial_dtype.as_struct_fields();
    assert_eq!(fields.names().as_ref(), &["sum", "is_overflow", "is_empty"]);
    assert_eq!(
        fields.field("sum"),
        Some(DType::Primitive(PType::I64, Nullability::NonNullable))
    );

    let proto = aggregate.serialize_proto()?;
    let decoded = pb::AggregateFn::decode(proto.encode_to_vec().as_slice())?;
    let round_tripped = AggregateFnRef::from_proto(&decoded, &array_session())?;
    assert_eq!(round_tripped, aggregate);
    Ok(())
}

#[test]
fn legacy_sum_is_unchanged() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let options = NumericalAggregateOpts::default();
    assert_eq!(Sum.bind(options).id().as_ref(), "vortex.sum");
    assert!(matches!(
        Sum.partial_dtype(&options, &dtype),
        Some(DType::Primitive(..))
    ));

    let mut legacy = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype.clone())?;
    let mut v2 = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype)?;

    assert_eq!(
        legacy.finish()?.as_primitive().typed_value::<i64>(),
        Some(0)
    );
    assert!(v2.finish()?.is_null());
    Ok(())
}

#[rstest]
#[case::primitive(PrimitiveArray::from_option_iter([None::<i32>, None]).into_array())]
#[case::float(PrimitiveArray::from_option_iter::<f64, _>([None, None]).into_array())]
#[case::bool(BoolArray::from_iter([None::<bool>, None]).into_array())]
#[case::constant(
    ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 4).into_array()
)]
#[case::decimal(
    DecimalArray::new(
        buffer![0i64, 0],
        DecimalDType::new(10, 2),
        Validity::AllInvalid,
    )
    .into_array()
)]
fn all_null_is_null(#[case] array: ArrayRef) -> VortexResult<()> {
    assert!(sum_v2(&array, &mut array_session().create_execution_ctx())?.is_null());
    Ok(())
}

#[test]
fn valid_zero_is_not_empty() -> VortexResult<()> {
    let zeros = PrimitiveArray::from_iter([0i32, 0]).into_array();
    assert_eq!(
        sum_v2(&zeros, &mut array_session().create_execution_ctx())?
            .as_primitive()
            .typed_value::<i64>(),
        Some(0)
    );

    let false_values = BoolArray::from_iter([false, false]).into_array();
    assert_eq!(
        sum_v2(&false_values, &mut array_session().create_execution_ctx())?
            .as_primitive()
            .typed_value::<u64>(),
        Some(0)
    );
    Ok(())
}

#[test]
fn all_nan_is_nonempty_zero_when_skipped() -> VortexResult<()> {
    let array =
        PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
    assert_eq!(
        sum_v2(&array, &mut array_session().create_execution_ctx())?
            .as_primitive()
            .typed_value::<f64>(),
        Some(0.0)
    );

    let included = sum_with_options(&array, NumericalAggregateOpts::include_nans())?;
    assert!(
        included
            .as_primitive()
            .typed_value::<f64>()
            .expect("non-null sum")
            .is_nan()
    );
    Ok(())
}

#[test]
fn ignores_ambiguous_legacy_sum_stat() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([None::<i32>, None]).into_array();
    array
        .statistics()
        .set(Stat::Sum, Precision::Exact(ScalarValue::from(42i64)));

    assert!(sum_v2(&array, &mut array_session().create_execution_ctx())?.is_null());
    Ok(())
}

#[test]
fn overflow_is_explicit_and_absorbing() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let mut accumulator = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype)?;
    let batch = PrimitiveArray::new(buffer![i64::MAX, 1, 7], Validity::NonNullable).into_array();
    accumulator.accumulate(&batch, &mut array_session().create_execution_ctx())?;
    assert!(accumulator.is_saturated());

    let partial = accumulator.partial_scalar()?;
    let fields = partial.as_struct();
    assert_eq!(
        fields
            .field("sum")
            .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
        Some(i64::MAX)
    );
    assert_eq!(
        fields
            .field("is_overflow")
            .and_then(|flag| flag.as_bool().value()),
        Some(true)
    );
    assert_eq!(
        fields
            .field("is_empty")
            .and_then(|flag| flag.as_bool().value()),
        Some(false)
    );
    assert!(accumulator.finish()?.is_null());
    Ok(())
}

#[test]
fn empty_chunk_is_a_merge_identity() -> VortexResult<()> {
    let value = PrimitiveArray::from_option_iter([Some(7i32)]).into_array();
    let empty = PrimitiveArray::from_option_iter([None::<i32>]).into_array();
    let chunked =
        ChunkedArray::try_new(vec![value, empty], DType::Primitive(PType::I32, Nullable))?
            .into_array();

    assert_eq!(
        sum_v2(&chunked, &mut array_session().create_execution_ctx())?
            .as_primitive()
            .typed_value::<i64>(),
        Some(7)
    );
    Ok(())
}

#[test]
fn combine_partials_empty_is_identity() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let mut empty = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype.clone())?;
    let empty_partial = empty.partial_scalar()?;

    empty.combine_partials(empty_partial.clone())?;
    assert!(empty.final_scalar()?.is_null());

    let mut value = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype.clone())?;
    let batch = PrimitiveArray::from_iter([7i64]).into_array();
    value.accumulate(&batch, &mut array_session().create_execution_ctx())?;
    let value_partial = value.partial_scalar()?;

    empty.combine_partials(value_partial.clone())?;
    assert_eq!(
        empty.final_scalar()?.as_primitive().typed_value::<i64>(),
        Some(7)
    );

    let mut value_then_empty =
        Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype)?;
    value_then_empty.combine_partials(value_partial)?;
    value_then_empty.combine_partials(empty_partial)?;
    assert_eq!(
        value_then_empty
            .final_scalar()?
            .as_primitive()
            .typed_value::<i64>(),
        Some(7)
    );
    Ok(())
}

#[test]
fn combine_partials_overflow_is_absorbing() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let mut max = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype.clone())?;
    let max_batch = PrimitiveArray::from_iter([i64::MAX]).into_array();
    max.accumulate(&max_batch, &mut array_session().create_execution_ctx())?;

    let mut one = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype.clone())?;
    let one_batch = PrimitiveArray::from_iter([1i64]).into_array();
    one.accumulate(&one_batch, &mut array_session().create_execution_ctx())?;

    let mut combined =
        Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype.clone())?;
    combined.combine_partials(max.partial_scalar()?)?;
    combined.combine_partials(one.partial_scalar()?)?;
    assert!(combined.is_saturated());
    assert!(combined.final_scalar()?.is_null());

    combined.combine_partials(max.partial_scalar()?)?;
    let overflow_partial = combined.partial_scalar()?;
    let fields = overflow_partial.as_struct();
    assert_eq!(
        fields
            .field("sum")
            .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
        Some(i64::MAX)
    );
    assert_eq!(
        fields
            .field("is_overflow")
            .and_then(|flag| flag.as_bool().value()),
        Some(true)
    );
    assert_eq!(
        fields
            .field("is_empty")
            .and_then(|flag| flag.as_bool().value()),
        Some(false)
    );

    let mut propagated = Accumulator::try_new(SumV2, NumericalAggregateOpts::default(), dtype)?;
    propagated.combine_partials(overflow_partial)?;
    assert!(propagated.is_saturated());
    assert!(propagated.final_scalar()?.is_null());
    Ok(())
}

#[test]
fn grouped_primitive_tracks_empty_overflow_and_null_groups() -> VortexResult<()> {
    let elements =
        PrimitiveArray::from_option_iter([Some(5i64), None, Some(i64::MAX), Some(1)]).into_array();
    let groups = ListViewArray::try_new(
        elements,
        buffer![0i32, 0, 1, 2, 0].into_array(),
        buffer![1i32, 0, 1, 2, 1].into_array(),
        Validity::from_iter([true, true, true, true, false]),
    )?
    .into_array();
    let mut accumulator = GroupedAccumulator::try_new(
        SumV2,
        NumericalAggregateOpts::default(),
        DType::Primitive(PType::I64, Nullable),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    accumulator.accumulate_list(&groups, &mut ctx)?;

    let result = accumulator.finish()?;
    let expected =
        PrimitiveArray::from_option_iter([Some(5i64), None, None, None, None]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_bool_fallback_tracks_empty_groups() -> VortexResult<()> {
    let elements = BoolArray::from_iter([Some(true), Some(true), None, None]).into_array();
    let groups = ListViewArray::try_new(
        elements,
        buffer![0i32, 2, 2].into_array(),
        buffer![2i32, 0, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let mut accumulator = GroupedAccumulator::try_new(
        SumV2,
        NumericalAggregateOpts::default(),
        DType::Bool(Nullable),
    )?;
    let mut ctx = array_session().create_execution_ctx();
    accumulator.accumulate_list(&groups, &mut ctx)?;

    let result = accumulator.finish()?;
    let expected = PrimitiveArray::from_option_iter([Some(2u64), None, None]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::all_valid(
    Validity::AllValid,
    [Some(10i64), None, None, Some(40)]
)]
#[case::all_invalid(Validity::AllInvalid, [None, None, None, None])]
#[case::some_invalid(
    Validity::from_iter([true, true, true, false]),
    [Some(10i64), None, None, None]
)]
fn finalize_struct_applies_partial_and_struct_validity(
    #[case] validity: Validity,
    #[case] expected: [Option<i64>; 4],
) -> VortexResult<()> {
    let partials = StructArray::try_new(
        FieldNames::from(["sum", "is_overflow", "is_empty"]),
        vec![
            PrimitiveArray::from_iter([10i64, 20, 30, 40]).into_array(),
            BoolArray::from_iter([false, true, false, false]).into_array(),
            BoolArray::from_iter([false, false, true, false]).into_array(),
        ],
        4,
        validity,
    )?
    .into_array();

    let result = SumV2.finalize(partials)?;
    let expected = PrimitiveArray::from_option_iter(expected).into_array();
    assert_arrays_eq!(
        &result,
        &expected,
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}
