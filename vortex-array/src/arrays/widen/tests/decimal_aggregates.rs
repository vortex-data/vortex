// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal aggregate equivalence, including the precision-dependent sum fallback.
//!
//! Each batch uses fresh canonical arrays so cached statistics cannot hide the kernel path.

use rstest::rstest;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::aggregates::assert_aggregate_batches;
use super::aggregates::kernel_partial;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VTable;
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

/// Every aggregate that applies to decimals, and whether the `Widen` aggregate kernel serves it.
///
/// The kernel serves `Sum` only with `skip_nans`, which has the legacy `Stat::Sum` slot.
/// `decimal_sum_partial` in `compute/aggregate.rs` explains why that slot keeps the partial exact.
fn aggregates() -> Vec<(AggregateFnRef, bool)> {
    let mut aggregates = vec![
        (IsConstant.bind(EmptyOptions), true),
        (IsSorted.bind(IsSortedOptions { strict: false }), true),
        (IsSorted.bind(IsSortedOptions { strict: true }), true),
        (First.bind(EmptyOptions), false),
        (Last.bind(EmptyOptions), false),
    ];

    for skip_nans in [true, false] {
        let options = NumericalAggregateOpts { skip_nans };
        aggregates.extend([
            (MinMax.bind(options), true),
            (Min.bind(options), true),
            (Max.bind(options), true),
            (Sum.bind(options), skip_nans),
            (SumV2.bind(options), false),
        ]);
    }

    aggregates
}

fn canonical_batch(values: &[Option<i128>], dtype: &DType) -> DecimalArray {
    let decimal = *dtype
        .as_decimal_opt()
        .expect("decimal fixture has a decimal dtype");

    match_each_decimal_value_type!(decimal.canonical_values_type(), |T| {
        let values = values.iter().map(|value| {
            value.map(|value| {
                <T as BigCast>::from(value).expect("fixture values fit the parent canonical width")
            })
        });
        match dtype.nullability() {
            Nullable => DecimalArray::from_option_iter(values, decimal),
            NonNullable => DecimalArray::from_iter(
                values.map(|value| value.expect("non-nullable fixture contains only valid values")),
                decimal,
            ),
        }
    })
}

/// Every fixture must exercise the `Widen` path, including empty and all-null batches.
#[track_caller]
fn encode_batch(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let encoded = WidenArray::encode_decimal(array, ctx)?;
    assert!(encoded.is::<Widen>());

    Ok(encoded)
}

#[rstest]
#[case::extrema(
    vec![
        vec![Some(-99), None, Some(99)],                                             //
        vec![Some(-9999), Some(9999)],                                               //
        vec![Some(-999_999_999), Some(999_999_999)],                                 //
        vec![Some(-999_999_999_999_999_999), Some(999_999_999_999_999_999)],         //
        vec![Some(-(10i128.pow(38) - 1)), Some(10i128.pow(38) - 1)],                 //
    ],
    Nullable,
)]
#[case::ascending(
    vec![vec![Some(-99), Some(-1)], vec![Some(0), Some(99)]],
    NonNullable,
)]
#[case::descending_boundary(
    vec![vec![Some(0), Some(99)], vec![Some(-99), Some(-1)]],
    Nullable,
)]
#[case::unsorted_batch(vec![vec![Some(99), Some(-99)]], NonNullable)]
#[case::equal_boundary(vec![vec![Some(7)], vec![Some(7)]], Nullable)]
#[case::constant(vec![vec![Some(7), Some(7)], vec![Some(7)]], NonNullable)]
#[case::changed_constant(vec![vec![Some(7), Some(7)], vec![Some(8)]], Nullable)]
#[case::constant_with_null(vec![vec![Some(7), None, Some(7)]], Nullable)]
#[case::empty(vec![vec![], vec![]], Nullable)]
#[case::all_null(vec![vec![None], vec![None, None]], Nullable)]
#[case::single_null(vec![vec![None]], Nullable)]
#[case::single_null_then_sorted(vec![vec![None], vec![Some(1), Some(2)]], Nullable)]
#[case::empty_then_constant(vec![vec![], vec![Some(7), Some(7)]], Nullable)]
#[case::leading_and_trailing_nulls(
    vec![vec![None, Some(-7)], vec![Some(7), None]],
    Nullable,
)]
fn test_decimal_aggregate_partials(
    #[case] batches: Vec<Vec<Option<i128>>>,
    #[case] nullability: Nullability,
    #[values(-3, 0, 2)] scale: i8,
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

/// A child sum would overflow precision 12, the precision of `sum_decimal_dtype(Decimal(2, 0))`,
/// while the logical sum fits `Decimal(28, 0)`. The kernel serves only `Sum` with `skip_nans`, and
/// it sums in the logical sum dtype without materializing the huge constant child.
#[rstest]
#[case::sum_skip_nans(Sum.bind(NumericalAggregateOpts { skip_nans: true }), true)]
#[case::sum_include_nans(Sum.bind(NumericalAggregateOpts { skip_nans: false }), false)]
#[case::sum_v2_skip_nans(SumV2.bind(NumericalAggregateOpts { skip_nans: true }), false)]
#[case::sum_v2_include_nans(SumV2.bind(NumericalAggregateOpts { skip_nans: false }), false)]
fn test_decimal_sum_uses_logical_precision(
    #[case] aggregate: AggregateFnRef,
    #[case] served: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child_decimal = DecimalDType::new(2, 0);
    let parent_decimal = DecimalDType::new(18, 0);
    let parent_dtype = DType::Decimal(parent_decimal, NonNullable);
    let child_scalar = Scalar::decimal(DecimalValue::I8(99), child_decimal, NonNullable);
    let parent_scalar = Scalar::decimal(DecimalValue::I64(99), parent_decimal, NonNullable);
    let len = 1usize << 34;
    let huge_child = ConstantArray::new(child_scalar, len).into_array();
    let huge_widen = WidenArray::try_new(huge_child.clone(), parent_dtype.clone())?.into_array();
    let expected = Scalar::decimal(
        DecimalValue::I128(1_700_807_049_216),
        DecimalDType::new(28, 0),
        Nullable,
    );

    // Calling the registered kernel directly must not execute this huge array.
    let kernel = ctx
        .aggregate_kernels()
        .find_aggregate_kernel(Widen.id(), aggregate.id());
    if aggregate.is::<Sum>() {
        let kernel = kernel.expect("the session registers the Widen Sum kernel");
        let partial = kernel.aggregate(&aggregate, &huge_widen, &mut ctx)?;
        assert_eq!(partial, served.then(|| expected.clone()));
    } else {
        assert!(kernel.is_none(), "SumV2 has no Widen kernel");
    }

    let mut parent = aggregate.accumulator(&parent_dtype)?;
    parent.accumulate(
        &ConstantArray::new(parent_scalar, len).into_array(),
        &mut ctx,
    )?;
    assert_eq!(parent.finish()?, expected);

    let mut child = aggregate.accumulator(huge_child.dtype())?;
    child.accumulate(&huge_child, &mut ctx)?;
    assert_eq!(
        child.finish()?,
        Scalar::null(DType::Decimal(DecimalDType::new(12, 0), Nullable))
    );

    let batches = vec![
        vec![],                //
        vec![Some(99); 256],   //
        vec![None, Some(-99)], //
    ];
    assert_aggregate_batches(
        &aggregate,
        served,
        &batches,
        &parent_dtype.as_nullable(),
        |values, dtype, _ctx| Ok(canonical_batch(values, dtype)),
        encode_batch,
        &mut ctx,
    )?;

    let values = DecimalArray::from_iter([99i8], child_decimal);
    let widen = WidenArray::try_new(values.into_array(), parent_dtype.clone())?.into_array();
    assert_eq!(
        kernel_partial(&aggregate, &widen, &mut ctx)?.is_some(),
        served
    );
    let canonical = DecimalArray::from_iter([99i64], parent_decimal).into_array();
    let mut actual = aggregate.accumulator(&parent_dtype)?;
    let mut reference = aggregate.accumulator(&parent_dtype)?;
    actual.accumulate(&widen, &mut ctx)?;
    reference.accumulate(&canonical, &mut ctx)?;
    assert_eq!(actual.partial_scalar()?, reference.partial_scalar()?);

    // Doubling real batch partials reaches the same logical length without allocating its rows.
    for doubling in 1..=34 {
        let actual_partial = actual.partial_scalar()?;
        let reference_partial = reference.partial_scalar()?;
        actual.combine_partials(&actual_partial)?;
        reference.combine_partials(&reference_partial)?;
        assert_eq!(
            actual.partial_scalar()?,
            reference.partial_scalar()?,
            "{aggregate} after {doubling} doublings"
        );
    }

    assert_eq!(actual.finish()?, expected);
    assert_eq!(reference.finish()?, expected);

    Ok(())
}

/// Returns a non-nullable `Widen` of `Decimal(4, 0)` over `Decimal(2, 0)` values, and the
/// canonical array with the same values.
fn sum_batches(values: &[i8]) -> VortexResult<(ArrayRef, ArrayRef)> {
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

/// The sum dtype of `Decimal(4, 0)` is `Decimal(14, 0)`. From a running sum 100 below its bound,
/// a batch of 99 still fits, and one more unit overflows to null. Each batch partial folds into
/// the running sum with the same precision check as the canonical batch.
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
        let (widen, canonical) = sum_batches(&[value])?;
        assert!(kernel_partial(&aggregate, &widen, &mut ctx)?.is_some());
        actual.accumulate(&widen, &mut ctx)?;
        expected.accumulate(&canonical, &mut ctx)?;

        assert_eq!(actual.partial_scalar()?, expected.partial_scalar()?);
        assert_eq!(actual.partial_scalar()?, result);
    }

    Ok(())
}

/// A constant child sums with one multiplication in the logical sum dtype `Decimal(13, 0)`. The
/// sum `9 * 1_111_111_111_111` is its largest value, and one more row overflows to null.
///
/// The reference is a constant too, because its rows cannot be materialized, so it takes the same
/// multiplication. The doc of `decimal_sum_partial` shows that adding the rows gives the same sum.
#[rstest]
fn test_decimal_sum_at_precision_bound(
    #[values(9, -9)] value: i8,
    #[values(0, 1)] extra_rows: usize,
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
    let parent_scalar =
        Scalar::decimal(DecimalValue::I16(value.into()), parent_decimal, NonNullable);
    let canonical = ConstantArray::new(parent_scalar, len).into_array();

    // NB: A kernel decline would materialize the rows of the `Widen`, so assert the kernel first.
    assert!(kernel_partial(&aggregate, &widen, &mut ctx)?.is_some());

    let mut actual = aggregate.accumulator(&dtype)?;
    let mut expected = aggregate.accumulator(&dtype)?;
    actual.accumulate(&widen, &mut ctx)?;
    expected.accumulate(&canonical, &mut ctx)?;

    let sum_dtype = DecimalDType::new(13, 0);
    let unit: i64 = value.signum().into();
    let bound = unit * (10i64.pow(13) - 1);
    let result = match extra_rows {
        0 => Scalar::decimal(DecimalValue::I64(bound), sum_dtype, Nullable),
        _ => Scalar::null(DType::Decimal(sum_dtype, Nullable)),
    };
    assert_eq!(actual.finish()?, result);
    assert_eq!(expected.finish()?, result);

    Ok(())
}

/// A sum of the values has the sum dtype of the child, so the kernel neither reads nor caches it.
#[test]
fn test_decimal_sum_ignores_values_statistics() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let aggregate = Sum.bind(NumericalAggregateOpts { skip_nans: true });
    let child_sum_dtype = DecimalDType::new(12, 0);
    let planted = Scalar::decimal(DecimalValue::I64(-1), child_sum_dtype, Nullable);
    let planted = planted
        .into_value()
        .vortex_expect("the planted sum is not null");

    let (planted_widen, _) = sum_batches(&[40, 2])?;
    let planted_child = planted_widen.as_::<Widen>().values().clone();
    planted_child
        .statistics()
        .set(Stat::Sum, Precision::Exact(planted.clone()));
    let (fresh_widen, _) = sum_batches(&[40, 2])?;
    let fresh_child = fresh_widen.as_::<Widen>().values().clone();

    let sum_dtype = DecimalDType::new(14, 0);
    let expected = Scalar::decimal(DecimalValue::I64(42), sum_dtype, Nullable);
    for widen in [&planted_widen, &fresh_widen] {
        assert!(kernel_partial(&aggregate, widen, &mut ctx)?.is_some());
        let mut accumulator = aggregate.accumulator(widen.dtype())?;
        accumulator.accumulate(widen, &mut ctx)?;
        assert_eq!(accumulator.finish()?, expected);
    }

    let planted_stat = planted_child.statistics().get(Stat::Sum);
    assert_eq!(
        planted_stat.map(|sum| sum.into_value()),
        Precision::Exact(Some(planted))
    );
    assert!(fresh_child.statistics().get(Stat::Sum).is_absent());

    Ok(())
}
