// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal aggregate results and overflow behavior.
//!
//! Compares decimal [`Widen`] arrays with canonical inputs and checks sum precision and statistics.

use rstest::rstest;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::aggregates::assert_aggregate_batches;
use super::aggregates::kernel_partial;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::first::First;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::last::Last;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

/// Decimal aggregates paired with whether the [`Widen`] kernel must return a partial.
fn aggregates() -> Vec<(AggregateFnRef, bool)> {
    let options = NumericalAggregateOpts::default();

    vec![
        (MinMax.bind(options), true),
        (Min.bind(options), true),
        (Max.bind(options), true),
        (IsConstant.bind(EmptyOptions), true),
        (IsSorted.bind(IsSortedOptions { strict: false }), true),
        (IsSorted.bind(IsSortedOptions { strict: true }), true),
        (Sum.bind(options), true),
        (Sum.bind(NumericalAggregateOpts { skip_nans: false }), false),
        (SumV2.bind(options), false),
        (First.bind(EmptyOptions), false),
        (Last.bind(EmptyOptions), false),
    ]
}

fn canonical_batch(values: &[Option<i128>], dtype: &DType) -> DecimalArray {
    let decimal = *dtype
        .as_decimal_opt()
        .vortex_expect("decimal fixture has a decimal dtype");

    match_each_decimal_value_type!(decimal.canonical_values_type(), |T| {
        let values = values.iter().map(|value| {
            value.map(|value| {
                <T as BigCast>::from(value)
                    .vortex_expect("fixture values fit the parent canonical width")
            })
        });
        match dtype.nullability() {
            Nullable => DecimalArray::from_option_iter(values, decimal),
            NonNullable => DecimalArray::from_iter(
                values.map(|value| {
                    value.vortex_expect("non-nullable fixture contains only valid values")
                }),
                decimal,
            ),
        }
    })
}

/// Encodes a batch and asserts that it exercises the [`Widen`] kernel.
#[track_caller]
fn encode_batch(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let encoded = WidenArray::encode_decimal(array, ctx)?;
    assert!(encoded.is::<Widen>());

    Ok(encoded)
}

/// Builds matching encoded and canonical sum inputs with a non-nullable `Decimal(4, 0)` dtype.
fn sum_batch(values: &[i8]) -> VortexResult<(ArrayRef, ArrayRef)> {
    let child = DecimalArray::from_iter(values.iter().copied(), DecimalDType::new(2, 0));
    let widen = WidenArray::try_new(
        child.into_array(),
        DType::Decimal(DecimalDType::new(4, 0), NonNullable),
    )?;
    let canonical = DecimalArray::from_iter(
        values.iter().map(|&value| -> i16 { value.into() }),
        DecimalDType::new(4, 0),
    );

    Ok((widen.into_array(), canonical.into_array()))
}

#[rstest]
#[case::extrema(
    vec![
        vec![Some(-99), None, Some(99)], // i8 values with a null.
        vec![Some(-9_999), Some(9_999)], // i16 extrema.
        vec![Some(-999_999_999), Some(999_999_999)], // i32 extrema.
        vec![Some(-999_999_999_999_999_999), Some(999_999_999_999_999_999)], // i64 extrema.
        vec![Some(-(10i128.pow(38) - 1)), Some(10i128.pow(38) - 1)], // i128 extrema.
    ],
    Nullable,
    -3,
)]
#[case::ascending(
    vec![vec![Some(-99), Some(-1)], vec![Some(0), Some(99)]],
    NonNullable,
    2,
)]
#[case::descending_boundary(
    vec![vec![Some(0), Some(99)], vec![Some(-99), Some(-1)]],
    Nullable,
    0,
)]
#[case::unsorted_batch(vec![vec![Some(99), Some(-99)]], NonNullable, 0)]
#[case::sum_exceeds_child_width(
    vec![
        vec![Some(99), Some(98)], // The batch sum exceeds i8::MAX.
        vec![Some(-99), Some(-98)], // The batch sum is below i8::MIN.
    ],
    NonNullable,
    0,
)]
#[case::equal_boundary(vec![vec![Some(7)], vec![Some(7)]], Nullable, 0)]
#[case::constant(vec![vec![Some(7), Some(7)], vec![Some(7)]], NonNullable, 0)]
#[case::changed_constant(vec![vec![Some(7), Some(7)], vec![Some(8)]], Nullable, 0)]
#[case::constant_with_null(vec![vec![Some(7), None, Some(7)]], Nullable, 0)]
#[case::empty(vec![vec![], vec![]], Nullable, 0)]
#[case::all_null(vec![vec![None], vec![None, None]], Nullable, 0)]
#[case::single_null(vec![vec![None]], Nullable, 0)]
#[case::single_null_then_sorted(vec![vec![None], vec![Some(1), Some(2)]], Nullable, 0)]
#[case::empty_then_constant(vec![vec![], vec![Some(7), Some(7)]], Nullable, 0)]
#[case::leading_and_trailing_nulls(
    vec![vec![None, Some(-7)], vec![Some(7), None]],
    Nullable,
    0,
)]
fn test_decimal_aggregate_partials(
    #[case] batches: Vec<Vec<Option<i128>>>,
    #[case] nullability: Nullability,
    #[case] scale: i8,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Decimal(DecimalDType::new(76, scale), nullability);

    for (aggregate, served) in aggregates() {
        assert_aggregate_batches(
            &aggregate,
            served,
            &batches,
            &dtype,
            |values, dtype, _ctx| Ok(canonical_batch(values, dtype)),
            encode_batch,
            &mut ctx,
        )?;
    }

    Ok(())
}

/// Overflow is checked after combining each batch with the running sum, for both signs.
#[rstest]
fn test_decimal_sum_running_overflow(#[values(1, -1)] sign: i8) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let aggregate = Sum.bind(NumericalAggregateOpts { skip_nans: true });
    let dtype = DType::Decimal(DecimalDType::new(4, 0), NonNullable);
    let sum_dtype = DecimalDType::new(14, 0);
    let unit: i64 = sign.into();
    let seed = unit * (10i64.pow(14) - 100);
    let seed = Scalar::decimal(DecimalValue::I64(seed), sum_dtype, Nullable);

    let mut actual = aggregate.accumulator(&dtype)?;
    let mut expected = aggregate.accumulator(&dtype)?;
    actual.combine_partials(&seed)?;
    expected.combine_partials(&seed)?;

    let fits = unit * (10i64.pow(14) - 1);
    let batches = [
        (
            sign * 99,
            Scalar::decimal(DecimalValue::I64(fits), sum_dtype, Nullable),
        ),
        (sign, Scalar::null(DType::Decimal(sum_dtype, Nullable))),
    ];
    for (value, result) in batches {
        let (widen, canonical) = sum_batch(&[value])?;
        assert!(kernel_partial(&aggregate, &widen, &mut ctx)?.is_some());
        actual.accumulate(&widen, &mut ctx)?;
        expected.accumulate(&canonical, &mut ctx)?;

        assert_eq!(actual.partial_scalar()?, expected.partial_scalar()?);
        assert_eq!(actual.partial_scalar()?, result);
    }

    Ok(())
}

/// A constant sum fits at the precision limit and becomes null after one more row.
/// The fitting sum exceeds the child's sum precision, so it also requires the logical sum dtype.
#[rstest]
#[case::fits(0)]
#[case::overflows(1)]
fn test_decimal_sum_at_precision_bound(
    #[values(9, -9)] value: i8,
    #[case] extra_rows: usize,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let aggregate = Sum.bind(NumericalAggregateOpts { skip_nans: true });
    let child_decimal = DecimalDType::new(1, 0);
    let parent_decimal = DecimalDType::new(3, 0);
    let dtype = DType::Decimal(parent_decimal, NonNullable);
    let len = 1_111_111_111_111 + extra_rows;
    let child_scalar = Scalar::decimal(DecimalValue::I8(value), child_decimal, NonNullable);
    let child = ConstantArray::new(child_scalar, len).into_array();
    let widen = WidenArray::try_new(child, dtype.clone())?.into_array();

    // NB: A kernel decline would materialize the rows of the `Widen`, so assert the kernel first.
    assert!(kernel_partial(&aggregate, &widen, &mut ctx)?.is_some());

    let mut actual = aggregate.accumulator(&dtype)?;
    actual.accumulate(&widen, &mut ctx)?;

    let sum_dtype = DecimalDType::new(13, 0);
    let unit: i64 = value.signum().into();
    let bound = unit * (10i64.pow(13) - 1);
    let result = match extra_rows {
        0 => Scalar::decimal(DecimalValue::I64(bound), sum_dtype, Nullable),
        _ => Scalar::null(DType::Decimal(sum_dtype, Nullable)),
    };
    assert_eq!(actual.finish()?, result);

    Ok(())
}

/// A logical sum must ignore child sum statistics and leave them unchanged.
#[rstest]
#[case::cached(true)]
#[case::absent(false)]
fn test_decimal_sum_ignores_values_statistics(#[case] cached: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let aggregate = Sum.bind(NumericalAggregateOpts { skip_nans: true });
    let planted = Scalar::decimal(DecimalValue::I64(-1), DecimalDType::new(12, 0), Nullable)
        .into_value()
        .vortex_expect("the planted sum is not null");
    let (widen, _) = sum_batch(&[40, 2])?;
    let child = widen.as_::<Widen>().values().clone();
    if cached {
        child.statistics().set(Stat::Sum, Precision::Exact(planted));
    }
    let child_sum = child.statistics().get(Stat::Sum);

    assert!(kernel_partial(&aggregate, &widen, &mut ctx)?.is_some());
    let mut accumulator = aggregate.accumulator(widen.dtype())?;
    accumulator.accumulate(&widen, &mut ctx)?;

    let expected = Scalar::decimal(DecimalValue::I64(42), DecimalDType::new(14, 0), Nullable);
    assert_eq!(accumulator.finish()?, expected);
    assert_eq!(child.statistics().get(Stat::Sum), child_sum);

    Ok(())
}
