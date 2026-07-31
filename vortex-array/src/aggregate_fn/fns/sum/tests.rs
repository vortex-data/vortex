// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the default [`Sum`] behavior: the `{sum, is_overflow, is_empty}` state algebra, the
//! null-for-zero-valid-values rule across input kinds, NaN and overflow handling,
//! cached-statistic consumption, and grouped aggregation.

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::Sum;
use super::SumAggregateOpts;
use super::sum;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

/// Sum an array with explicit [`SumAggregateOpts`] (test-only helper).
fn sum_with_options(arr: &ArrayRef, options: SumAggregateOpts) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(Sum, options, arr.dtype().clone())?;
    acc.accumulate(arr, &mut array_session().create_execution_ctx())?;
    acc.finish()
}

#[test]
fn sum_uses_new_partial_shape_by_default() {
    let options = SumAggregateOpts::default();
    let sum = Sum.bind(options);
    assert_eq!(sum.id().as_ref(), "vortex.sum");
    let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let partial_dtype = Sum.partial_dtype(&options, &input_dtype).unwrap();
    assert_eq!(partial_dtype.nullability(), Nullable);
    let fields = partial_dtype.as_struct_fields();
    assert_eq!(fields.names().as_ref(), &["sum", "is_overflow", "is_empty"]);
    assert_eq!(
        fields.field("sum"),
        Some(DType::Primitive(PType::I64, Nullability::NonNullable))
    );
    assert_eq!(
        fields.field("is_overflow"),
        Some(DType::Bool(Nullability::NonNullable))
    );
    assert_eq!(
        fields.field("is_empty"),
        Some(DType::Bool(Nullability::NonNullable))
    );
}

#[test]
fn legacy_options_use_scalar_partial_and_zero_on_empty() -> VortexResult<()> {
    let options = SumAggregateOpts::deserialize(&NumericalAggregateOpts::skip_nans().serialize())?;
    assert_eq!(
        options,
        SumAggregateOpts {
            skip_nans: true,
            struct_partial: false,
        }
    );

    let input_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    assert_eq!(
        Sum.partial_dtype(&options, &input_dtype),
        Some(DType::Primitive(PType::I64, Nullable))
    );

    let mut acc = Accumulator::try_new(Sum, options, input_dtype)?;
    assert_eq!(
        acc.partial_scalar()?.as_primitive().typed_value::<i64>(),
        Some(0)
    );
    assert_eq!(acc.finish()?.as_primitive().typed_value::<i64>(), Some(0));
    Ok(())
}

// State algebra: the `{sum, is_overflow, is_empty}` monoid.

#[test]
fn sum_state_empty_is_null() -> VortexResult<()> {
    // A state that never saw a valid value finalizes to null, and combining empty states
    // stays empty.
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut state = Sum.empty_partial(&SumAggregateOpts::default(), &dtype)?;
    let empty = Sum.to_scalar(&state)?;
    let fields = empty.as_struct();
    assert_eq!(
        fields
            .field("sum")
            .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
        Some(0)
    );
    assert_eq!(
        fields
            .field("is_overflow")
            .and_then(|is_overflow| is_overflow.as_bool().value()),
        Some(false)
    );
    assert_eq!(
        fields
            .field("is_empty")
            .and_then(|is_empty| is_empty.as_bool().value()),
        Some(true)
    );
    Sum.combine_partials(&mut state, empty)?;
    assert!(Sum.finalize_scalar(&state)?.is_null());
    Ok(())
}

#[test]
fn sum_state_empty_is_identity() -> VortexResult<()> {
    // Combining an empty state into a non-empty state changes nothing.
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut state = Sum.empty_partial(&SumAggregateOpts::default(), &dtype)?;
    Sum.combine_partials(&mut state, Scalar::primitive(100i64, Nullable))?;

    let empty = Sum.to_scalar(&Sum.empty_partial(&SumAggregateOpts::default(), &dtype)?)?;
    Sum.combine_partials(&mut state, empty)?;

    let result = Sum.finalize_scalar(&state)?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(100));
    Ok(())
}

#[test]
fn sum_state_overflow_sets_flag_and_poisons() -> VortexResult<()> {
    // Overflow sets the flag and poisons the merge even when combined with later values.
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let mut overflowed = Sum.empty_partial(&SumAggregateOpts::default(), &dtype)?;
    Sum.combine_partials(&mut overflowed, Scalar::primitive(i64::MAX, Nullable))?;
    Sum.combine_partials(&mut overflowed, Scalar::primitive(1i64, Nullable))?;
    let overflowed = Sum.to_scalar(&overflowed)?;
    let fields = overflowed.as_struct();
    assert_eq!(
        fields
            .field("sum")
            .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
        Some(i64::MAX)
    );
    assert_eq!(
        fields
            .field("is_overflow")
            .and_then(|is_overflow| is_overflow.as_bool().value()),
        Some(true)
    );
    assert_eq!(
        fields
            .field("is_empty")
            .and_then(|is_empty| is_empty.as_bool().value()),
        Some(false)
    );

    let mut state = Sum.empty_partial(&SumAggregateOpts::default(), &dtype)?;
    Sum.combine_partials(&mut state, Scalar::primitive(5i64, Nullable))?;
    Sum.combine_partials(&mut state, overflowed)?;
    Sum.combine_partials(&mut state, Scalar::primitive(7i64, Nullable))?;

    assert!(Sum.finalize_scalar(&state)?.is_null());
    Ok(())
}

// The null-for-zero-valid-values rule.

#[rstest]
#[case::i32(DType::Primitive(PType::I32, Nullability::NonNullable))]
#[case::f64(DType::Primitive(PType::F64, Nullability::NonNullable))]
#[case::bool(DType::Bool(Nullability::NonNullable))]
fn sum_empty_is_null(#[case] dtype: DType) -> VortexResult<()> {
    let mut acc = Accumulator::try_new(Sum, SumAggregateOpts::default(), dtype)?;
    assert!(acc.finish()?.is_null());
    Ok(())
}

#[rstest]
#[case::primitive(PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array())]
#[case::float(PrimitiveArray::from_option_iter::<f32, _>([None, None, None]).into_array())]
#[case::bool(BoolArray::from_iter([None::<bool>, None, None]).into_array())]
#[case::constant_primitive(
    ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 10).into_array()
)]
#[case::constant_bool(ConstantArray::new(Scalar::null(DType::Bool(Nullable)), 10).into_array())]
#[case::constant_decimal(
    ConstantArray::new(Scalar::null(DType::Decimal(DecimalDType::new(10, 2), Nullable)), 10)
        .into_array()
)]
fn sum_all_null_is_null(#[case] array: ArrayRef) -> VortexResult<()> {
    let result = sum(&array, &mut array_session().create_execution_ctx())?;
    assert!(result.is_null());
    Ok(())
}

#[test]
fn sum_all_nan_is_zero_not_null() -> VortexResult<()> {
    // NaNs are valid values: with the default `skip_nans` they contribute nothing, but
    // the sum is a genuine `0.0`, unlike an all-null array whose sum is null.
    let arr = PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
    let result = sum(&arr, &mut array_session().create_execution_ctx())?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
    Ok(())
}

#[test]
fn legacy_scalar_partial_preserves_zero_on_empty() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
    assert!(sum(&arr, &mut ctx)?.is_null());

    // A scalar `Stat::Sum` is an old partial. Its zero identity cannot encode emptiness, so its
    // historical zero-on-empty result is preserved when it is encountered.
    arr.statistics()
        .set(Stat::Sum, Precision::Exact(ScalarValue::from(0i64)));
    assert_eq!(
        sum(&arr, &mut ctx)?.as_primitive().typed_value::<i64>(),
        Some(0)
    );

    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut state = Sum.empty_partial(&SumAggregateOpts::default(), &dtype)?;
    Sum.combine_partials(&mut state, Scalar::primitive(0i64, Nullable))?;
    assert_eq!(
        Sum.finalize_scalar(&state)?
            .as_primitive()
            .typed_value::<i64>(),
        Some(0)
    );

    let mut overflowed = Sum.empty_partial(&SumAggregateOpts::default(), &dtype)?;
    Sum.combine_partials(
        &mut overflowed,
        Scalar::null(DType::Primitive(PType::I64, Nullable)),
    )?;
    let overflowed = Sum.to_scalar(&overflowed)?;
    let fields = overflowed.as_struct();
    assert_eq!(
        fields
            .field("sum")
            .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
        Some(0)
    );
    assert_eq!(
        fields
            .field("is_overflow")
            .and_then(|is_overflow| is_overflow.as_bool().value()),
        Some(true)
    );
    assert_eq!(
        fields
            .field("is_empty")
            .and_then(|is_empty| is_empty.as_bool().value()),
        Some(false)
    );
    Ok(())
}

// Return dtype widening (mirrors `Sum`'s rules; the result is always nullable).

#[rstest]
#[case::bool(
    DType::Bool(Nullability::NonNullable),
    DType::Primitive(PType::U64, Nullable)
)]
#[case::i32(
    DType::Primitive(PType::I32, Nullability::NonNullable),
    DType::Primitive(PType::I64, Nullable)
)]
#[case::u8(
    DType::Primitive(PType::U8, Nullability::NonNullable),
    DType::Primitive(PType::U64, Nullable)
)]
#[case::f32(
    DType::Primitive(PType::F32, Nullability::NonNullable),
    DType::Primitive(PType::F64, Nullable)
)]
#[case::decimal(
    DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable),
    DType::Decimal(DecimalDType::new(20, 2), Nullable)
)]
fn sum_return_dtype_widens(#[case] input: DType, #[case] expected: DType) {
    let dtype = Sum
        .return_dtype(&SumAggregateOpts::default(), &input)
        .unwrap();
    assert_eq!(dtype, expected);
}

// One value smoke test per accumulate branch; summation arithmetic is pinned by the
// shared kernels' tests in the `sum` module.

#[test]
fn sum_primitive_with_nulls() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
    let result = sum(&arr, &mut array_session().create_execution_ctx())?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
    Ok(())
}

#[test]
fn sum_bool_with_nulls() -> VortexResult<()> {
    let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]);
    let result = sum(
        &arr.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
    Ok(())
}

#[test]
fn sum_decimal_with_nulls() -> VortexResult<()> {
    let decimal = DecimalArray::new(
        buffer![100i32, 200i32, 300i32, 400i32],
        DecimalDType::new(4, 2),
        Validity::from_iter([true, false, true, true]),
    );
    let result = sum(
        &decimal.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    let expected = Scalar::try_new(
        DType::Decimal(DecimalDType::new(14, 2), Nullable),
        Some(ScalarValue::from(DecimalValue::from(800i32))),
    )?;
    assert_eq!(result, expected);
    Ok(())
}

#[test]
fn sum_constant() -> VortexResult<()> {
    let array = ConstantArray::new(5u64, 10).into_array();
    let result = sum(&array, &mut array_session().create_execution_ctx())?;
    assert_eq!(result, 50u64.into());
    Ok(())
}

#[test]
fn sum_constant_false_is_zero_not_null() -> VortexResult<()> {
    let array = ConstantArray::new(false, 10).into_array();
    let result = sum(&array, &mut array_session().create_execution_ctx())?;
    assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
    Ok(())
}

#[test]
fn sum_multi_batch_and_finish_resets() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(Sum, SumAggregateOpts::default(), dtype)?;

    let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
    acc.accumulate(&batch1, &mut ctx)?;
    let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
    acc.accumulate(&batch2, &mut ctx)?;
    let result = acc.finish()?;
    assert_eq!(result.as_primitive().typed_value::<i64>(), Some(48));

    // finish resets the state: an untouched accumulator is empty again.
    assert!(acc.finish()?.is_null());
    let batch3 = PrimitiveArray::new(buffer![1i32], Validity::NonNullable).into_array();
    acc.accumulate(&batch3, &mut ctx)?;
    assert_eq!(acc.finish()?.as_primitive().typed_value::<i64>(), Some(1));
    Ok(())
}

// Chunked accumulation: the nullable sum must merge across chunks.

#[test]
fn sum_chunked_floats_with_nulls() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter(vec![Some(1.5f64), None, Some(3.2), Some(4.8)]);
    let chunk2 = PrimitiveArray::from_option_iter(vec![Some(2.1f64), Some(5.7), None]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

    let result = sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(result.as_primitive().as_::<f64>(), Some(17.3));
    Ok(())
}

#[test]
fn sum_chunked_all_nulls_is_null() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
    let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
    let result = sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert!(result.is_null());
    Ok(())
}

#[test]
fn sum_chunked_empty_chunks() -> VortexResult<()> {
    let chunk1 = PrimitiveArray::from_option_iter(vec![Some(10.5f64), Some(20.3)]);
    let chunk2 = ConstantArray::new(Scalar::primitive(0f64, Nullable), 0);
    let chunk3 = PrimitiveArray::from_option_iter(vec![Some(5.2f64)]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(
        vec![
            chunk1.into_array(),
            chunk2.into_array(),
            chunk3.into_array(),
        ],
        dtype,
    )?;

    let result = sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(result.as_primitive().as_::<f64>(), Some(36.0));
    Ok(())
}

#[test]
fn sum_chunked_value_survives_empty_chunk() -> VortexResult<()> {
    // One valid value in one chunk, followed by an all-null chunk: the value must survive merging
    // with the second chunk's null identity.
    let chunk1 = PrimitiveArray::from_option_iter::<u32, _>(vec![Some(1)]);
    let chunk2 = PrimitiveArray::from_option_iter::<u32, _>(vec![None]);
    let dtype = chunk1.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

    let result = sum(
        &chunked.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(result.as_primitive().as_::<u64>(), Some(1));
    Ok(())
}

// NaN handling and its interplay with the nullable sum and cached statistics.

#[test]
fn sum_f64_with_nan_and_nulls() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(f64::NAN), Some(3.0)])
        .into_array();
    let result = sum(&arr, &mut array_session().create_execution_ctx())?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(4.0));
    Ok(())
}

#[test]
fn sum_f64_with_nan_not_skipping() -> VortexResult<()> {
    let arr =
        PrimitiveArray::new(buffer![1.0f64, f64::NAN, 2.0], Validity::NonNullable).into_array();
    let result = sum_with_options(&arr, SumAggregateOpts::include_nans())?;
    assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
    Ok(())
}

#[test]
fn sum_not_skipping_shortcircuits_on_exact_nan_count_stat() -> VortexResult<()> {
    // The array has no NaNs; a planted exact NaNCount stat proves the NaN poisoning came
    // from the stat rather than a scan.
    let arr = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
    arr.statistics()
        .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(1u64)));
    let result = sum_with_options(&arr, SumAggregateOpts::include_nans())?;
    assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
    Ok(())
}

#[test]
fn sum_uses_cached_stat_sum() -> VortexResult<()> {
    // A planted exact `Stat::Sum` with a known null count is consumed instead of a scan
    // (the planted value differs from the actual data to prove it).
    let arr = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
    arr.statistics()
        .set(Stat::Sum, Precision::Exact(ScalarValue::from(42.0f64)));
    arr.statistics()
        .set(Stat::NullCount, Precision::Exact(ScalarValue::from(0u64)));
    let result = sum(&arr, &mut array_session().create_execution_ctx())?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(42.0));
    Ok(())
}

#[test]
fn sum_not_skipping_uses_cached_sum_when_nan_free() -> VortexResult<()> {
    // With an exact NaNCount of zero, the planted exact Sum stat is usable as-is.
    let arr = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
    arr.statistics()
        .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(0u64)));
    arr.statistics()
        .set(Stat::Sum, Precision::Exact(ScalarValue::from(42.0f64)));
    arr.statistics()
        .set(Stat::NullCount, Precision::Exact(ScalarValue::from(0u64)));
    let result = sum_with_options(&arr, SumAggregateOpts::include_nans())?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(42.0));
    Ok(())
}

#[test]
fn sum_constant_nan() -> VortexResult<()> {
    let arr = ConstantArray::new(f64::NAN, 4).into_array();
    // NaN constants are skipped by default (a non-empty zero sum) and poison the sum otherwise.
    let result = sum_with_options(&arr, SumAggregateOpts::default())?;
    assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));

    let result = sum_with_options(&arr, SumAggregateOpts::include_nans())?;
    assert!(result.as_primitive().typed_value::<f64>().unwrap().is_nan());
    Ok(())
}

#[test]
fn sum_f64_with_infinity() -> VortexResult<()> {
    let batch = PrimitiveArray::new(
        buffer![1.0f64, f64::INFINITY, f64::NEG_INFINITY, 2.0],
        Validity::NonNullable,
    )
    .into_array();
    let acc = sum(&batch, &mut array_session().create_execution_ctx())?;
    // INFINITY + NEG_INFINITY = NaN, which is treated as saturated
    assert!(acc.as_primitive().typed_value::<f64>().unwrap().is_nan());

    let mut acc = Accumulator::try_new(
        Sum,
        SumAggregateOpts::default(),
        DType::Primitive(PType::F64, Nullability::NonNullable),
    )?;
    acc.accumulate(&batch, &mut array_session().create_execution_ctx())?;
    assert!(acc.is_saturated());
    Ok(())
}

// Overflow: a null sum value plus an explicit saturation flag.

#[test]
fn sum_checked_overflow_is_null_and_saturates() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let mut acc = Accumulator::try_new(Sum, SumAggregateOpts::default(), dtype)?;
    assert!(!acc.is_saturated());

    let batch = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
    acc.accumulate(&batch, &mut array_session().create_execution_ctx())?;
    assert!(acc.is_saturated());
    let result = acc.finish()?;
    assert!(result.is_null());

    // finish resets state, clearing saturation
    assert!(!acc.is_saturated());
    Ok(())
}

#[test]
fn sum_decimal_i256_overflow() -> VortexResult<()> {
    let decimal_dtype = DecimalDType::new(76, 0);
    let decimal = DecimalArray::new(
        buffer![i256::MAX, i256::MAX, i256::MAX],
        decimal_dtype,
        Validity::AllValid,
    );

    let result = sum(
        &decimal.into_array(),
        &mut array_session().create_execution_ctx(),
    )?;
    assert_eq!(
        result,
        Scalar::null(DType::Decimal(decimal_dtype, Nullable))
    );
    Ok(())
}

#[test]
fn sum_decimal_near_precision_boundary() -> VortexResult<()> {
    // Input precision 4 → return precision min(76, 4+10) = 14.
    // Native type for precision 14 is I64 (max precision 18), so 14 < 18.
    // Use combine_partials to push state near (but under) 10^14.
    let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
    let mut state = Sum.empty_partial(&SumAggregateOpts::default(), &input_dtype)?;

    let near_limit = Scalar::decimal(
        DecimalValue::from(99_999_999_999_990i64),
        DecimalDType::new(14, 0),
        Nullable,
    );
    Sum.combine_partials(&mut state, near_limit)?;

    // Add a small value that keeps us just under 10^14.
    let small = Scalar::decimal(DecimalValue::from(9i64), DecimalDType::new(14, 0), Nullable);
    Sum.combine_partials(&mut state, small)?;

    let result = Sum.finalize_scalar(&state)?;
    assert!(!result.is_null());
    assert_eq!(
        result.as_decimal().decimal_value(),
        Some(DecimalValue::I256(i256::from_i128(99_999_999_999_999)))
    );
    Ok(())
}

#[rstest]
#[case::positive(99_999_999_999_999i64, 1i64)]
#[case::negative(-99_999_999_999_999i64, -1i64)]
fn sum_decimal_precision_overflow_within_i256(
    #[case] near_limit: i64,
    #[case] one_more: i64,
) -> VortexResult<()> {
    // Input precision 4 → return precision 14. Native I64 (max 18).
    // The max representable magnitude for precision 14 is 10^14 - 1: pushing the sum to
    // exactly ±10^14 fails fits_in_precision even though i256 arithmetic does not
    // overflow. This tests the precision-based saturation path in combine_partials.
    let input_dtype = DType::Decimal(DecimalDType::new(4, 0), Nullability::NonNullable);
    let mut state = Sum.empty_partial(&SumAggregateOpts::default(), &input_dtype)?;

    let near_limit = Scalar::decimal(
        DecimalValue::from(near_limit),
        DecimalDType::new(14, 0),
        Nullable,
    );
    Sum.combine_partials(&mut state, near_limit)?;

    let one_more = Scalar::decimal(
        DecimalValue::from(one_more),
        DecimalDType::new(14, 0),
        Nullable,
    );
    Sum.combine_partials(&mut state, one_more)?;

    let result = Sum.finalize_scalar(&state)?;
    assert!(result.is_null());
    assert_eq!(
        result.dtype(),
        &DType::Decimal(DecimalDType::new(14, 0), Nullable)
    );
    Ok(())
}

#[test]
fn sum_decimal_accumulate_precision_overflow() -> VortexResult<()> {
    // Test precision overflow via the accumulate_decimal path (not combine_partials).
    // Input precision 27 → return precision 37. Native for 37 is I128 (max 38), so 37 < 38.
    // Use combine_partials to get the state close to 10^37, then accumulate a real array
    // that pushes it over.
    let input_dtype = DType::Decimal(DecimalDType::new(27, 0), Nullability::NonNullable);
    let return_dtype = DecimalDType::new(37, 0);
    let mut state = Sum.empty_partial(&SumAggregateOpts::default(), &input_dtype)?;

    // Set state to 10^37 - 1 via combine_partials.
    let near_limit_val: i128 = 10i128.pow(37) - 1;
    let near_limit = Scalar::decimal(DecimalValue::from(near_limit_val), return_dtype, Nullable);
    Sum.combine_partials(&mut state, near_limit)?;

    // Now accumulate a real i128 array with a single element = 1 to overflow precision.
    let decimal = DecimalArray::new(buffer![1i128], DecimalDType::new(27, 0), Validity::AllValid);
    let columnar = crate::Columnar::Canonical(crate::Canonical::Decimal(decimal));
    let mut ctx = array_session().create_execution_ctx();
    Sum.accumulate(&mut state, &columnar, &mut ctx)?;

    let result = Sum.finalize_scalar(&state)?;
    assert!(result.is_null());
    Ok(())
}

// Grouped aggregation: empty, all-null, and null groups through the lazy finalize.

fn run_grouped_sum(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
    let mut acc =
        GroupedAccumulator::try_new(Sum, SumAggregateOpts::default(), elem_dtype.clone())?;
    let mut ctx = array_session().create_execution_ctx();
    acc.accumulate_list(groups, &mut ctx)?;
    acc.finish()
}

#[test]
fn grouped_sum_partial_distinguishes_empty_overflow_and_null_group() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::from_option_iter([Some(5i64), None, Some(i64::MAX), Some(1)]).into_array();
    let groups = ListViewArray::try_new(
        elements,
        buffer![0i32, 0, 1, 2, 0].into_array(),
        buffer![1i32, 0, 1, 2, 1].into_array(),
        Validity::from_iter([true, true, true, true, false]),
    )?
    .into_array();
    let mut acc = GroupedAccumulator::try_new(
        Sum,
        SumAggregateOpts::default(),
        DType::Primitive(PType::I64, Nullable),
    )?;
    acc.accumulate_list(&groups, &mut ctx)?;
    let partials = acc.flush()?;

    let value = partials.execute_scalar(0, &mut ctx)?;
    let fields = value.as_struct();
    assert_eq!(
        fields
            .field("sum")
            .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
        Some(5)
    );
    assert_eq!(
        fields
            .field("is_overflow")
            .and_then(|is_overflow| is_overflow.as_bool().value()),
        Some(false)
    );
    assert_eq!(
        fields
            .field("is_empty")
            .and_then(|is_empty| is_empty.as_bool().value()),
        Some(false)
    );

    for index in [1, 2] {
        let empty = partials.execute_scalar(index, &mut ctx)?;
        let fields = empty.as_struct();
        assert_eq!(
            fields
                .field("sum")
                .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
            Some(0)
        );
        assert_eq!(
            fields
                .field("is_overflow")
                .and_then(|is_overflow| is_overflow.as_bool().value()),
            Some(false)
        );
        assert_eq!(
            fields
                .field("is_empty")
                .and_then(|is_empty| is_empty.as_bool().value()),
            Some(true)
        );
    }

    let overflow = partials.execute_scalar(3, &mut ctx)?;
    let fields = overflow.as_struct();
    assert_eq!(
        fields
            .field("sum")
            .and_then(|sum| sum.as_primitive().typed_value::<i64>()),
        Some(i64::MAX)
    );
    assert_eq!(
        fields
            .field("is_overflow")
            .and_then(|is_overflow| is_overflow.as_bool().value()),
        Some(true)
    );
    assert_eq!(
        fields
            .field("is_empty")
            .and_then(|is_empty| is_empty.as_bool().value()),
        Some(false)
    );
    assert!(partials.execute_scalar(4, &mut ctx)?.is_null());
    Ok(())
}

#[test]
fn grouped_sum_fallback_empty_and_all_null_groups() -> VortexResult<()> {
    // Bool elements are rejected by the primitive grouped kernel, forcing the generic
    // per-group fallback: empty and all-null groups have null sums there too.
    let mut ctx = array_session().create_execution_ctx();
    let elements = BoolArray::from_iter([Some(true), Some(true), None, None]).into_array();
    let groups = ListViewArray::try_new(
        elements,
        buffer![0i32, 2, 2].into_array(),
        buffer![2i32, 0, 2].into_array(),
        Validity::NonNullable,
    )?
    .into_array();

    let result = run_grouped_sum(&groups, &DType::Bool(Nullable))?;
    let expected = PrimitiveArray::from_option_iter([Some(2u64), None, None]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_fixed_size_list() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
    let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(6i64), Some(15i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_with_null_elements() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
            .into_array();
    let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(4i64), Some(11i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_with_null_group() -> VortexResult<()> {
    // A null group must become a null row through the lazy finalize's validity handling.
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9], Validity::NonNullable)
            .into_array();
    let validity = Validity::from_iter([true, false, true]);
    let groups = FixedSizeListArray::try_new(elements, 3, validity, 3)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(6i64), None, Some(24i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_all_null_elements_in_group() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::from_option_iter([None::<i32>, None, Some(3), Some(4)]).into_array();
    let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::I32, Nullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    // The all-null group has a null sum
    let expected = PrimitiveArray::from_option_iter([None, Some(7i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_all_nan_is_zero_not_null() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![f64::NAN, f64::NAN, 3.0, 4.0], Validity::NonNullable)
            .into_array();
    let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

    let elem_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

    let expected = PrimitiveArray::from_option_iter([Some(0.0f64), Some(7.0)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_finish_resets() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut acc = GroupedAccumulator::try_new(Sum, SumAggregateOpts::default(), elem_dtype)?;

    let elements1 = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
    let groups1 = FixedSizeListArray::try_new(elements1, 2, Validity::NonNullable, 2)?;
    acc.accumulate_list(&groups1.into_array(), &mut ctx)?;
    let result1 = acc.finish()?;

    let expected1 = PrimitiveArray::from_option_iter([Some(3i64), Some(7i64)]).into_array();
    assert_arrays_eq!(&result1, &expected1, &mut ctx);

    let elements2 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
    let groups2 = FixedSizeListArray::try_new(elements2, 2, Validity::NonNullable, 1)?;
    acc.accumulate_list(&groups2.into_array(), &mut ctx)?;
    let result2 = acc.finish()?;

    let expected2 = PrimitiveArray::from_option_iter([Some(30i64)]).into_array();
    assert_arrays_eq!(&result2, &expected2, &mut ctx);
    Ok(())
}

#[test]
fn grouped_sum_listview_out_of_order_offsets_with_null_group() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        PrimitiveArray::new(buffer![100i32, 200, 300], Validity::NonNullable).into_array();
    let offsets = PrimitiveArray::new(buffer![2i32, 0, 1], Validity::NonNullable).into_array();
    let sizes = PrimitiveArray::new(buffer![1i32, 1, 1], Validity::NonNullable).into_array();
    let validity = Validity::from_iter([true, false, true]);
    let groups = ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array();

    let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let result = run_grouped_sum(&groups, &elem_dtype)?;

    // group 0 -> elements[2..3] = 300; group 1 -> null; group 2 -> elements[1..2] = 200.
    let expected =
        PrimitiveArray::from_option_iter([Some(300i64), None, Some(200i64)]).into_array();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}
