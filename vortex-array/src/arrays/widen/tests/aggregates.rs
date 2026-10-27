// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;

use rstest::rstest;
use vortex_error::VortexResult;

use super::SESSION;
use super::partially_valid;
use super::widen;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use crate::aggregate_fn::fns::bounded_min::BoundedMin;
use crate::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::first::First;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::last::Last;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum::sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeOpts;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::sum::WidenSumKernel;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;

/// Every aggregate that applies to integers, and whether the `Widen` kernels serve it. The others
/// take the default path.
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
        (
            UncompressedSizeInBytes.bind(UncompressedSizeOpts::inexact()),
            false,
        ),
        (
            UncompressedSizeInBytes.bind(UncompressedSizeOpts::exact()),
            false,
        ),
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

/// Accumulates `batches` of `dtype` with `aggregate` once as [`WidenArray::encode`] encodes them
/// and once as canonical arrays, and asserts that the running partials and the results are equal.
///
/// On each batch that encodes to a `Widen`, the kernel that the session finds for `aggregate` on
/// `Widen` must return a partial when `served`, and the partial must equal the partial of a fresh
/// accumulator on the canonical batch. When not `served`, the lookup must miss or the kernel must
/// decline. Encoding and each accumulator get their own canonical batch, because both cache
/// statistics on the array they read, and an accumulator reads a cached statistic instead of the
/// data.
#[track_caller]
fn assert_aggregate_batches(
    aggregate: &AggregateFnRef,
    served: bool,
    batches: &[Vec<Option<i64>>],
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let kernel = ctx
        .aggregate_kernels()
        .find_aggregate_kernel(Widen.id(), aggregate.id());
    let mut actual = aggregate.accumulator(dtype)?;
    let mut expected = aggregate.accumulator(dtype)?;

    for values in batches {
        let encoded = WidenArray::encode(canonical_batch(values, dtype, ctx)?, ctx)?;

        if encoded.is::<Widen>() {
            let partial = match kernel {
                Some(kernel) => kernel.aggregate(aggregate, &encoded, ctx)?,
                None => None,
            };
            assert_eq!(partial.is_some(), served, "{aggregate} on {values:?}");

            if let Some(partial) = partial {
                let mut reference = aggregate.accumulator(dtype)?;
                reference.accumulate(&canonical_batch(values, dtype, ctx)?.into_array(), ctx)?;
                assert_eq!(partial, reference.flush()?, "{aggregate} on {values:?}");
            }
        }

        actual.accumulate(&encoded, ctx)?;
        expected.accumulate(&canonical_batch(values, dtype, ctx)?.into_array(), ctx)?;
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
        assert_aggregate_batches(&aggregate, served, &batches, &dtype, &mut ctx)?;
    }

    Ok(())
}

/// The sum kernel declines values that have a sum path of their own, and the default path gives the
/// canonical sum.
#[rstest]
#[case::constant(ConstantArray::new(5i8, 4).into_array())]
#[case::chunked(
    ChunkedArray::from_iter([
        PrimitiveArray::from_option_iter([Some(-3i8), None]).into_array(),
        PrimitiveArray::from_option_iter([Some(100i8)]).into_array(),
    ])
    .into_array()
)]
fn test_sum_declines_values_with_a_default_path(#[case] values: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen(values, PType::I64).into_array();
    let canonical = array
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
    let aggregate = Sum.bind(NumericalAggregateOpts::default());

    assert!(
        WidenSumKernel
            .aggregate(&aggregate, &array, &mut ctx)?
            .is_none()
    );
    assert_eq!(sum(&array, &mut ctx)?, sum(&canonical, &mut ctx)?);

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

/// The accumulator caches a statistic of the `Widen` on the `Widen`, in the logical dtype. A
/// kernel that asks the values caches their statistic on the values, in the values dtype.
#[test]
fn test_statistics_land_on_their_array() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = partially_valid();
    let values = array.values().clone();
    let array = array.into_array();

    sum(&array, &mut ctx)?;
    let mut min = Min
        .bind(NumericalAggregateOpts::default())
        .accumulator(array.dtype())?;
    min.accumulate(&array, &mut ctx)?;

    let logical_min = Scalar::primitive(-300i64, NonNullable);
    let values_min = Scalar::primitive(-300i16, NonNullable);
    assert_eq!(
        array.statistics().get(Stat::Sum),
        Precision::Exact(Scalar::primitive(0i64, Nullable))
    );
    assert_eq!(
        array.statistics().get(Stat::Min),
        Precision::Exact(logical_min)
    );
    assert_eq!(
        values.statistics().get(Stat::Min),
        Precision::Exact(values_min)
    );
    assert_eq!(values.statistics().get(Stat::Sum), Precision::Absent);

    Ok(())
}
