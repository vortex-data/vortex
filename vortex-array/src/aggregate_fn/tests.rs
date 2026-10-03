// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::combined::BinaryCombined;
use crate::aggregate_fn::combined::Combined;
use crate::aggregate_fn::combined::CombinedOptions;
use crate::aggregate_fn::combined::PairOptions;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::mean::Mean;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::array_session;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

fn final_result(aggregate: &AggregateFnRef, batches: &[ArrayRef]) -> VortexResult<Scalar> {
    let mut ctx = array_session().create_execution_ctx();
    let mut accumulator = aggregate.accumulator(batches[0].dtype())?;
    for batch in batches {
        accumulator.accumulate(batch, &mut ctx)?;
    }
    accumulator.finish()
}

#[rstest]
#[case::min(Min.bind(NumericalAggregateOpts::skip_nans()))]
#[case::max(Max.bind(NumericalAggregateOpts::skip_nans()))]
#[case::sum(Sum.bind(NumericalAggregateOpts::skip_nans()))]
#[case::count(Count.bind(NumericalAggregateOpts::skip_nans()))]
#[case::null_count(NullCount.bind(EmptyOptions))]
#[case::nan_count(NanCount.bind(EmptyOptions))]
#[case::physical_size(UncompressedSizeInBytes.bind(EmptyOptions))]
fn recovered_results_merge_like_accumulated_batches(
    #[case] aggregate: AggregateFnRef,
    #[values(&[], &[None, None], &[Some(2.0), None, Some(4.0)])] first_values: &[Option<f64>],
) -> VortexResult<()> {
    let first = PrimitiveArray::from_option_iter(first_values.iter().copied()).into_array();
    let second = PrimitiveArray::from_option_iter([Some(-3.0f64), Some(5.0)]).into_array();
    let batches = [first, second];
    let mut recovered = aggregate.accumulator(batches[0].dtype())?;
    for batch in &batches {
        let result = final_result(&aggregate, std::slice::from_ref(batch))?;
        let partial = aggregate
            .partial_from_result(batch.dtype(), &result)?
            .expect("the proven builtin conversions must recover their final results");
        recovered.combine_partials(partial)?;
    }
    assert_eq!(recovered.finish()?, final_result(&aggregate, &batches)?);
    Ok(())
}

#[rstest]
#[case::sorted(IsSorted.bind(IsSortedOptions { strict: false }), [3, 4], [1, 2])]
#[case::constant(IsConstant.bind(EmptyOptions), [1, 1], [2, 2])]
fn boolean_finals_cannot_recover_batch_boundaries(
    #[case] aggregate: AggregateFnRef,
    #[case] first: [i32; 2],
    #[case] second: [i32; 2],
) -> VortexResult<()> {
    let batches = [
        PrimitiveArray::from_iter(first).into_array(),
        PrimitiveArray::from_iter(second).into_array(),
    ];
    for batch in &batches {
        let result = final_result(&aggregate, std::slice::from_ref(batch))?;
        assert_eq!(result, true.into());
        assert!(
            aggregate
                .partial_from_result(batch.dtype(), &result)?
                .is_none()
        );
    }
    assert_eq!(final_result(&aggregate, &batches)?, false.into());
    Ok(())
}

#[test]
fn mean_final_cannot_recover_batch_weights() -> VortexResult<()> {
    let options = NumericalAggregateOpts::skip_nans();
    let mean = Mean::combined().bind(PairOptions(options, options));
    let batches = [
        PrimitiveArray::from_option_iter([Some(0.0f64), None]).into_array(),
        PrimitiveArray::from_option_iter([Some(9.0f64), Some(9.0), Some(9.0)]).into_array(),
    ];
    let first_result = final_result(&mean, &batches[..1])?;
    assert!(
        mean.partial_from_result(batches[0].dtype(), &first_result)?
            .is_none()
    );
    assert_eq!(
        final_result(&mean, &batches)?,
        Scalar::primitive(6.75f64, Nullability::Nullable)
    );
    Ok(())
}

#[rstest]
#[case::empty(vec![])]
#[case::all_null(vec![None, None])]
#[case::overflow(vec![Some(i64::MAX), Some(1)])]
fn sum_v2_final_cannot_recover_empty_or_overflow_state(
    #[case] values: Vec<Option<i64>>,
) -> VortexResult<()> {
    let sum = SumV2.bind(NumericalAggregateOpts::skip_nans());
    let batch = PrimitiveArray::from_option_iter(values).into_array();
    let result = final_result(&sum, std::slice::from_ref(&batch))?;
    assert!(result.is_null());
    assert!(sum.partial_from_result(batch.dtype(), &result)?.is_none());
    Ok(())
}

#[rstest]
#[case::overflow_first(true)]
#[case::overflow_second(false)]
fn original_sum_recovers_null_as_saturated_overflow(
    #[case] overflow_first: bool,
) -> VortexResult<()> {
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let batch = PrimitiveArray::from_iter([i64::MAX, 1]).into_array();
    let result = final_result(&sum, std::slice::from_ref(&batch))?;
    assert!(result.is_null());
    let overflow = sum.partial_from_result(batch.dtype(), &result)?.unwrap();
    let five = sum
        .partial_from_result(
            batch.dtype(),
            &Scalar::primitive(5i64, Nullability::Nullable),
        )?
        .unwrap();
    let mut accumulator = sum.accumulator(batch.dtype())?;
    for partial in if overflow_first {
        [overflow, five]
    } else {
        [five, overflow]
    } {
        accumulator.combine_partials(partial)?;
    }
    assert!(accumulator.is_saturated());
    assert!(accumulator.finish()?.is_null());
    Ok(())
}

#[rstest]
#[case::skip_nans(NumericalAggregateOpts::skip_nans())]
#[case::include_nans(NumericalAggregateOpts::include_nans())]
fn recovered_min_uses_bound_nan_options(
    #[case] options: NumericalAggregateOpts,
) -> VortexResult<()> {
    let min = Min.bind(options);
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let nan_batch = PrimitiveArray::from_iter([f64::NAN, 2.0]).into_array();
    let result = final_result(&min, &[nan_batch])?;
    let partial = min.partial_from_result(&dtype, &result)?.unwrap();
    let mut accumulator = min.accumulator(&dtype)?;
    accumulator.combine_partials(partial)?;
    let batch = PrimitiveArray::from_iter([3.0f64]).into_array();
    accumulator.accumulate(&batch, &mut array_session().create_execution_ctx())?;
    let result = accumulator.finish()?;
    if options.skip_nans {
        assert_eq!(result, Scalar::primitive(2.0f64, Nullability::Nullable));
    } else {
        assert!(result.as_primitive().is_nan());
    }
    Ok(())
}

#[test]
fn result_recovery_checks_types_and_unsupported_inputs() -> VortexResult<()> {
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    assert!(
        min.partial_from_result(&DType::from(PType::I32), &true.into())
            .is_err()
    );
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    assert!(
        sum.partial_from_result(&DType::Utf8(Nullability::NonNullable), &true.into())?
            .is_none()
    );
    assert!(
        WrongPartialDType
            .bind(EmptyOptions)
            .partial_from_result(&DType::from(PType::I32), &0u64.into())
            .is_err()
    );
    Ok(())
}

#[derive(Clone)]
struct WrongPartialDType;

impl AggregateFnVTable for WrongPartialDType {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        NullCount.id()
    }

    fn return_dtype(&self, options: &Self::Options, input: &DType) -> Option<DType> {
        NullCount.return_dtype(options, input)
    }

    fn partial_dtype(&self, options: &Self::Options, input: &DType) -> Option<DType> {
        NullCount.partial_dtype(options, input)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        NullCount.empty_partial(args)
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        NullCount.partial_from_scalar(args, scalar)
    }

    fn partial_from_result(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        result: Scalar,
    ) -> VortexResult<Option<Self::Partial>> {
        NullCount.partial_from_result(args, result)
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        NullCount.merge_partials(args, first, second)
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(true.into())
    }

    fn is_saturated(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        NullCount.is_saturated(args, partial)
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        NullCount.accumulate(args, partial, batch, ctx)
    }

    fn finalize(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        states: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        NullCount.finalize(args, states)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        NullCount.finalize_scalar(args, partial)
    }
}

#[rstest]
#[case::physical(UncompressedSizeInBytes.bind(EmptyOptions))]
#[case::custom(WrongPartialDType.bind(EmptyOptions))]
fn result_scope_defaults_to_the_physical_input(#[case] aggregate: AggregateFnRef) {
    assert!(!aggregate.is_representation_invariant());
}

#[rstest]
#[case::logical(Mean::combined().bind(PairOptions(NumericalAggregateOpts::skip_nans(), NumericalAggregateOpts::skip_nans())), true)]
#[case::physical_left(Combined::new(LeftResult(UncompressedSizeInBytes, NullCount)).bind(PairOptions(EmptyOptions, EmptyOptions)), false)]
#[case::physical_right(Combined::new(LeftResult(NullCount, UncompressedSizeInBytes)).bind(PairOptions(EmptyOptions, EmptyOptions)), false)]
#[case::custom(Combined::new(LeftResult(NullCount, WrongPartialDType)).bind(PairOptions(EmptyOptions, EmptyOptions)), false)]
fn combined_result_scope_requires_portable_children(
    #[case] aggregate: AggregateFnRef,
    #[case] expected: bool,
) {
    assert_eq!(aggregate.is_representation_invariant(), expected);
}

#[rstest]
#[case::skip(NumericalAggregateOpts::skip_nans())]
#[case::include(NumericalAggregateOpts::include_nans())]
fn result_scope_forwards_bound_options(#[case] options: NumericalAggregateOpts) {
    assert_eq!(
        OptionScopedCount
            .bind(options)
            .is_representation_invariant(),
        options.skip_nans
    );
    for aggregate in [
        Combined::new(LeftResult(OptionScopedCount, Count))
            .bind(PairOptions(options, NumericalAggregateOpts::skip_nans())),
        Combined::new(LeftResult(Count, OptionScopedCount))
            .bind(PairOptions(NumericalAggregateOpts::skip_nans(), options)),
    ] {
        assert_eq!(aggregate.is_representation_invariant(), options.skip_nans);
    }
}

#[derive(Clone)]
struct OptionScopedCount;

impl AggregateFnVTable for OptionScopedCount {
    type Options = NumericalAggregateOpts;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("test.option_scoped_count");
        *ID
    }

    fn is_representation_invariant(&self, options: &Self::Options) -> bool {
        options.skip_nans
    }

    fn return_dtype(&self, options: &Self::Options, input: &DType) -> Option<DType> {
        Count.return_dtype(options, input)
    }

    fn partial_dtype(&self, options: &Self::Options, input: &DType) -> Option<DType> {
        Count.partial_dtype(options, input)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        Count.empty_partial(args)
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        Count.partial_from_scalar(args, scalar)
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        Count.merge_partials(args, first, second)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Count.to_scalar(args, partial)
    }

    fn is_saturated(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        Count.is_saturated(args, partial)
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        Count.accumulate(args, partial, batch, ctx)
    }

    fn finalize(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        states: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Count.finalize(args, states)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Count.finalize_scalar(args, partial)
    }
}

#[derive(Clone)]
struct LeftResult<L, R>(L, R);

impl<L: AggregateFnVTable, R: AggregateFnVTable> BinaryCombined for LeftResult<L, R> {
    type Left = L;
    type Right = R;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("test.left_result");
        *ID
    }

    fn left(&self) -> Self::Left {
        self.0.clone()
    }

    fn right(&self) -> Self::Right {
        self.1.clone()
    }

    fn return_dtype(&self, options: &CombinedOptions<Self>, input: &DType) -> Option<DType> {
        self.0.return_dtype(&options.0, input)
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, CombinedOptions<Self>>,
        left: ArrayRef,
        _right: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(left)
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, CombinedOptions<Self>>,
        left: Scalar,
        _right: Scalar,
    ) -> VortexResult<Scalar> {
        Ok(left)
    }
}
