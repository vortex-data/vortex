// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for aggregate results and child-statistics reuse.
//!
//! Shared batch assertions compare integer and decimal [`Widen`] arrays with fresh canonical inputs.

use std::fmt::Debug;
use std::num::NonZeroUsize;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::WidenArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SESSION;
use super::widen;
use crate::all_non_null::AllNonNull;
use crate::all_null::AllNull;
use crate::bounded_max::BoundedMax;
use crate::bounded_max::BoundedMaxOptions;
use crate::bounded_min::BoundedMin;
use crate::bounded_min::BoundedMinOptions;
use crate::count::Count;
use crate::first::First;
use crate::is_constant::IsConstant;
use crate::is_sorted::IsSorted;
use crate::is_sorted::IsSortedOptions;
use crate::last::Last;
use crate::max::Max;
use crate::min::Min;
use crate::min_max::MinMax;
use crate::min_max::make_minmax_dtype;
use crate::null_count::NullCount;
use crate::sum::Sum;
use crate::sum_v2::SumV2;
use crate::uncompressed_size_in_bytes::UncompressedSizeInBytes;

/// Every aggregate that applies to integers, and whether the `Widen` aggregate kernel serves it.
fn aggregates() -> Vec<(AggregateFnRef, bool)> {
    let skip_nans = NumericalAggregateOpts::default();
    let include_nans = NumericalAggregateOpts { skip_nans: false };
    let max_bytes = NonZeroUsize::MIN;

    vec![
        (MinMax.bind(skip_nans), true),
        (Min.bind(skip_nans), true),
        (Max.bind(skip_nans), true),
        (IsConstant.bind(EmptyOptions), true),
        (IsSorted.bind(IsSortedOptions { strict: false }), true),
        (IsSorted.bind(IsSortedOptions { strict: true }), true),
        (Sum.bind(skip_nans), true),
        (Sum.bind(include_nans), false),
        (SumV2.bind(skip_nans), false),
        (First.bind(EmptyOptions), false),
        (Last.bind(EmptyOptions), false),
        (Count.bind(skip_nans), false),
        (NullCount.bind(EmptyOptions), false),
        (AllNull.bind(EmptyOptions), false),
        (AllNonNull.bind(EmptyOptions), false),
        (BoundedMin.bind(BoundedMinOptions { max_bytes }), false),
        (BoundedMax.bind(BoundedMaxOptions { max_bytes }), false),
        (UncompressedSizeInBytes.bind(EmptyOptions), false),
    ]
}

/// Returns a canonical batch of `dtype` with `values`, with statistics of its own.
fn canonical_batch(
    values: &[Option<i64>],
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    PrimitiveArray::from_option_iter(values.iter().copied())
        .into_array()
        .cast(dtype.clone())?
        .execute::<PrimitiveArray>(ctx)
}

/// Returns the registered [`Widen`] kernel's partial, or `None` if it is absent or declines.
pub(super) fn kernel_partial(
    aggregate: &AggregateFnRef,
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    match SESSION
        .aggregate_fns()
        .find_aggregate_kernel(Widen.id(), aggregate.id())
    {
        Some(kernel) => kernel.aggregate(aggregate, array, ctx),
        None => Ok(None),
    }
}

/// Checks encoded aggregates against canonical inputs, including kernel selection and partials.
///
/// `served` says whether the [`Widen`] kernel must return a partial. `build_batch` **must** return a
/// fresh array with its own statistics on every call. Reusing arrays could let aggregation read a
/// cached answer instead of testing the kernel.
#[track_caller]
pub(super) fn assert_aggregate_batches<T: Debug, C: IntoArray>(
    aggregate: &AggregateFnRef,
    served: bool,
    batches: &[T],
    dtype: &DType,
    mut build_batch: impl FnMut(&T, &DType, &mut ExecutionCtx) -> VortexResult<C>,
    mut encode: impl FnMut(C, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mut actual = aggregate.accumulator(dtype)?;
    let mut expected = aggregate.accumulator(dtype)?;

    for values in batches {
        let encoded = encode(build_batch(values, dtype, ctx)?, ctx)?;

        if encoded.is::<Widen>() {
            let partial = kernel_partial(aggregate, &encoded, ctx)?;
            assert_eq!(partial.is_some(), served, "{aggregate} on {values:?}");

            if let Some(partial) = partial {
                let mut reference = aggregate.accumulator(dtype)?;
                reference.accumulate(&build_batch(values, dtype, ctx)?.into_array(), ctx)?;
                assert_eq!(partial, reference.flush()?, "{aggregate} on {values:?}");
            }
        }

        actual.accumulate(&encoded, ctx)?;
        expected.accumulate(&build_batch(values, dtype, ctx)?.into_array(), ctx)?;
        assert_eq!(
            actual.partial_scalar()?,
            expected.partial_scalar()?,
            "{aggregate} after {values:?}"
        );
    }

    assert_eq!(actual.finish()?, expected.finish()?, "{aggregate}");

    Ok(())
}

#[rstest]
#[case::signed_child_bounds(
    vec![
        vec![],                                                   //
        vec![None, None],                                         //
        vec![Some(i8::MIN.into()), Some(42), None],               //
        vec![Some(i8::MAX.into())],                               //
        vec![Some(i16::MIN.into()), Some(i16::MAX.into())],       //
        vec![Some(i32::MIN.into()), None, Some(i32::MAX.into())], //
        vec![],                                                   //
    ],
    PType::I64,
    Nullable,
)]
#[case::unsigned_child_bounds(
    vec![
        vec![],                              //
        vec![None],                          //
        vec![Some(0), Some(u8::MAX.into())], //
        vec![Some(u16::MAX.into()), None],   //
        vec![Some(u32::MAX.into())],         //
    ],
    PType::U64,
    Nullable,
)]
#[case::signed_non_nullable(
    vec![
        vec![],                                             //
        vec![Some(i8::MIN.into()), Some(i8::MAX.into())],   //
        vec![Some(i16::MIN.into()), Some(i16::MAX.into())], //
        vec![Some(i32::MIN.into()), Some(i32::MAX.into())], //
    ],
    PType::I32,
    NonNullable,
)]
#[case::unsigned_non_nullable(
    vec![
        vec![Some(0), Some(u8::MAX.into())], //
        vec![Some(7), Some(7)],              //
        vec![Some(u16::MAX.into())],         //
    ],
    PType::U16,
    NonNullable,
)]
#[case::descending_boundary(
    vec![
        vec![Some(1000), Some(2000)], //
        vec![Some(-128), Some(127)],  //
    ],
    PType::I64,
    Nullable,
)]
#[case::equal_boundary(vec![vec![Some(7)], vec![Some(7)]], PType::U64, Nullable)]
#[case::empty(vec![vec![], vec![]], PType::I64, Nullable)]
#[case::all_null(vec![vec![None], vec![None, None]], PType::U64, Nullable)]
#[case::single_null(vec![vec![None]], PType::I64, Nullable)]
#[case::single_null_then_sorted(vec![vec![None], vec![Some(1), Some(2)]], PType::I64, Nullable)]
#[case::empty_then_constant(vec![vec![], vec![Some(7), Some(7)]], PType::I64, Nullable)]
// No narrower type holds `i64::MAX - 5`, so the first batch stays canonical.
#[case::running_overflow(vec![vec![Some(i64::MAX - 5)], vec![Some(10)]], PType::I64, Nullable)]
// A sum without the `Stat::Sum` slot adds 10 to the running sum and overflows, but the total of the
// second batch is zero.
#[case::running_overflow_recovers(
    vec![
        vec![Some(i64::MAX - 5)],  //
        vec![Some(10), Some(-10)], //
    ],
    PType::I64,
    Nullable,
)]
fn test_aggregate_partials(
    #[case] batches: Vec<Vec<Option<i64>>>,
    #[case] logical: PType,
    #[case] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Primitive(logical, nullability);

    for (aggregate, served) in aggregates() {
        assert_aggregate_batches(
            &aggregate,
            served,
            &batches,
            &dtype,
            |values, dtype, ctx| canonical_batch(values, dtype, ctx),
            WidenArray::encode,
            &mut ctx,
        )?;
    }

    Ok(())
}

/// The kernel serves a `Widen` over values that are not primitive, and gives the canonical result.
///
/// The test compares results, not partials. The accumulator skips the chunks after one that decides
/// the result, so the partial of chunked values can lack fields that the result ignores, such as
/// the last value of an unsorted batch.
#[rstest]
#[case::constant(ConstantArray::new(5i8, 4).into_array())]
#[case::chunked(
    ChunkedArray::from_iter([
        PrimitiveArray::from_option_iter([Some(-3i8), None]).into_array(),
        PrimitiveArray::from_option_iter([Some(100i8)]).into_array(),
    ])
    .into_array()
)]
fn test_non_primitive_values(#[case] values: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(values, PType::I64).into_array();

    for (aggregate, served) in aggregates() {
        let partial = kernel_partial(&aggregate, &array, &mut ctx)?;
        assert_eq!(partial.is_some(), served, "{aggregate}");

        if let Some(partial) = partial {
            let mut actual = aggregate.accumulator(array.dtype())?;
            actual.combine_partials(&partial)?;
            let canonical = array.clone().execute::<PrimitiveArray>(&mut ctx)?;
            let mut expected = aggregate.accumulator(array.dtype())?;
            expected.accumulate(&canonical.into_array(), &mut ctx)?;

            assert_eq!(actual.finish()?, expected.finish()?, "{aggregate}");
        }
    }

    Ok(())
}

/// The chunked kernel accumulates each chunk, so a `Widen` chunk and a canonical chunk give the
/// canonical result.
#[test]
fn test_chunked_matches_canonical() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Primitive(PType::I64, Nullable);
    let widen_chunk = widen(
        PrimitiveArray::from_option_iter([Some(-3i8), None, Some(5)]),
        PType::I64,
    );
    let canonical_chunk = PrimitiveArray::from_option_iter([Some(1000i64), Some(-7)]);
    let chunked = ChunkedArray::try_new(
        vec![widen_chunk.into_array(), canonical_chunk.into_array()],
        dtype.clone(),
    )?
    .into_array();
    let canonical = chunked
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();

    for (aggregate, _) in aggregates() {
        let mut actual = aggregate.accumulator(&dtype)?;
        actual.accumulate(&chunked, &mut ctx)?;
        let mut expected = aggregate.accumulator(&dtype)?;
        expected.accumulate(&canonical, &mut ctx)?;

        assert_eq!(actual.finish()?, expected.finish()?, "{aggregate}");
    }

    Ok(())
}

/// Cached extrema deliberately differ from the data, so the result proves the cache was read.
#[rstest]
#[case::integer(
    PrimitiveArray::from_iter([1i8, 2, 3]).into_array(),
    DType::Primitive(PType::I64, NonNullable),
    Scalar::primitive(-100i8, NonNullable),
    Scalar::primitive(100i8, NonNullable),
)]
#[case::decimal(
    DecimalArray::from_option_iter([Some(1i16), None], DecimalDType::new(4, 2)).into_array(),
    DType::Decimal(DecimalDType::new(18, 2), Nullable),
    Scalar::decimal(DecimalValue::I16(-9_999), DecimalDType::new(4, 2), NonNullable),
    Scalar::decimal(DecimalValue::I16(9_999), DecimalDType::new(4, 2), NonNullable),
)]
fn test_min_max_reads_values_extrema(
    #[case] values: ArrayRef,
    #[case] logical_dtype: DType,
    #[case] min: Scalar,
    #[case] max: Scalar,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    for (stat, extremum) in [(Stat::Min, &min), (Stat::Max, &max)] {
        let value = extremum.value().cloned();
        let value = value.vortex_expect("the planted extremum is not null");
        values.statistics().set(stat, Precision::Exact(value));
    }

    let array = WidenArray::try_new(values, logical_dtype.clone())?.into_array();
    let aggregate = MinMax.bind(NumericalAggregateOpts::default());

    let partial = kernel_partial(&aggregate, &array, &mut ctx)?;

    let element_dtype = logical_dtype.as_nonnullable();
    let expected = Scalar::struct_(
        make_minmax_dtype(&logical_dtype),
        vec![min.cast(&element_dtype)?, max.cast(&element_dtype)?],
    );
    assert_eq!(partial, Some(expected));

    Ok(())
}
