// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::columnar::AnyColumnar;
use crate::dtype::DType;
use crate::executor::max_iterations;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;

/// Reference-counted type-erased accumulator.
pub type AccumulatorRef = Box<dyn DynAccumulator>;

/// An accumulator used for computing aggregates over an entire stream of arrays.
pub struct Accumulator<V: AggregateFnVTable> {
    /// The vtable of the aggregate function.
    vtable: V,
    /// Type-erased aggregate function used for kernel dispatch.
    aggregate_fn: AggregateFnRef,
    /// The DType of the input.
    dtype: DType,
    /// The DType of the aggregate.
    return_dtype: DType,
    /// The DType of the accumulator state.
    partial_dtype: DType,
    /// The partial state of the accumulator, updated after each accumulate/merge call.
    partial: V::Partial,
}

impl<V: AggregateFnVTable> Accumulator<V> {
    pub fn try_new(vtable: V, options: V::Options, dtype: DType) -> VortexResult<Self> {
        let return_dtype = vtable.return_dtype(&options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;
        let partial_dtype = vtable.partial_dtype(&options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;
        let partial = vtable.empty_partial(&options, &dtype)?;
        let aggregate_fn = AggregateFn::new(vtable.clone(), options).erased();

        Ok(Self {
            vtable,
            aggregate_fn,
            dtype,
            return_dtype,
            partial_dtype,
            partial,
        })
    }
}

/// A trait object for type-erased accumulators, used for dynamic dispatch when the aggregate
/// function is not known at compile time.
pub trait DynAccumulator: 'static + Send {
    /// Accumulate a new array into the accumulator's state.
    fn accumulate(&mut self, batch: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()>;

    /// Fold an external partial-state scalar into this accumulator's state.
    ///
    /// The scalar must have the dtype reported by the vtable's `partial_dtype` for the
    /// options and input dtype used to construct this accumulator.
    fn combine_partials(&mut self, other: Scalar) -> VortexResult<()>;

    /// Whether the accumulator's result is fully determined.
    fn is_saturated(&self) -> bool;

    /// Reset the accumulator's state to the empty group.
    fn reset(&mut self);

    /// Read the current partial state as a scalar without resetting it.
    ///
    /// The returned scalar has the dtype reported by the vtable's `partial_dtype`.
    fn partial_scalar(&self) -> VortexResult<Scalar>;

    /// Compute the final aggregate result as a scalar without resetting state.
    fn final_scalar(&self) -> VortexResult<Scalar>;

    /// Flush the accumulation state and return the partial aggregate result as a scalar.
    ///
    /// Resets the accumulator state back to the initial state.
    fn flush(&mut self) -> VortexResult<Scalar>;

    /// Finish the accumulation and return the final aggregate result as a scalar.
    ///
    /// Resets the accumulator state back to the initial state.
    fn finish(&mut self) -> VortexResult<Scalar>;
}

impl<V: AggregateFnVTable> DynAccumulator for Accumulator<V> {
    fn accumulate(&mut self, batch: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        if self.is_saturated() {
            return Ok(());
        }

        vortex_ensure!(
            batch.dtype() == &self.dtype,
            "Input DType mismatch: expected {}, got {}",
            self.dtype,
            batch.dtype()
        );

        // 0. Legacy stats bridge: if this aggregate is still cached under a legacy Stat slot,
        //    consume that exact stat before kernel dispatch or decode.
        if let Some(stat) = Stat::from_aggregate_fn(&self.aggregate_fn)
            && let Precision::Exact(partial) = batch.statistics().get(stat)
        {
            let partial = if partial.dtype() == &self.partial_dtype {
                partial
            } else {
                vortex_ensure!(
                    partial.dtype().eq_ignore_nullability(&self.partial_dtype),
                    "Aggregate {} read legacy stat {} with dtype {}, expected {}",
                    self.aggregate_fn,
                    stat,
                    partial.dtype(),
                    self.partial_dtype,
                );
                partial.cast(&self.partial_dtype)?
            };
            self.vtable.combine_partials(&mut self.partial, partial)?;
            return Ok(());
        }

        let session = ctx.session().clone();

        // 1. Kernel registry first: a registered `(encoding, aggregate_fn)` kernel is strictly
        //    more specific than the vtable's `try_accumulate` short-circuit. Checking the
        //    registry first gives kernels for `Combined<V>` aggregates a chance to fire —
        //    `Combined::try_accumulate` always returns true, so a later kernel check would be
        //    unreachable.
        {
            let kernel = session
                .aggregate_fns()
                .find_aggregate_kernel(batch.encoding_id(), self.aggregate_fn.id());
            if let Some(kernel) = kernel
                && let Some(result) = kernel.aggregate(&self.aggregate_fn, batch, ctx)?
            {
                vortex_ensure!(
                    result.dtype() == &self.partial_dtype,
                    "Aggregate kernel returned {}, expected {}",
                    result.dtype(),
                    self.partial_dtype,
                );
                self.vtable.combine_partials(&mut self.partial, result)?;
                return Ok(());
            }
        }

        // 2. Allow the vtable to short-circuit on the raw array before decompression.
        if self.vtable.try_accumulate(&mut self.partial, batch, ctx)? {
            return Ok(());
        }

        // 3. Iteratively check the registry against each intermediate encoding, executing one
        //    step between checks. Mirrors the loop in `GroupedAccumulator::accumulate_list_view`.
        //    Iteration 0 re-checks the initial encoding — a redundant HashMap miss, the price of
        //    keeping the loop body uniform. Terminates on `AnyColumnar` (Canonical or Constant)
        //    since the vtable's `accumulate(&Columnar)` handles both cases directly.
        let mut batch = batch.clone();
        for _ in 0..max_iterations() {
            if batch.is::<AnyColumnar>() {
                break;
            }

            if let Some(kernel) = session
                .aggregate_fns()
                .find_aggregate_kernel(batch.encoding_id(), self.aggregate_fn.id())
                && let Some(result) = kernel.aggregate(&self.aggregate_fn, &batch, ctx)?
            {
                vortex_ensure!(
                    result.dtype() == &self.partial_dtype,
                    "Aggregate kernel returned {}, expected {}",
                    result.dtype(),
                    self.partial_dtype,
                );
                self.vtable.combine_partials(&mut self.partial, result)?;
                return Ok(());
            }

            batch = batch.execute(ctx)?;
        }

        // 4. Otherwise, execute the batch until it is columnar and accumulate it into the state.
        let columnar = batch.execute::<Columnar>(ctx)?;

        self.vtable.accumulate(&mut self.partial, &columnar, ctx)
    }

    fn combine_partials(&mut self, other: Scalar) -> VortexResult<()> {
        self.vtable.combine_partials(&mut self.partial, other)
    }

    fn is_saturated(&self) -> bool {
        self.vtable.is_saturated(&self.partial)
    }

    fn reset(&mut self) {
        self.vtable.reset(&mut self.partial);
    }

    fn partial_scalar(&self) -> VortexResult<Scalar> {
        let partial = self.vtable.to_scalar(&self.partial)?;

        #[cfg(debug_assertions)]
        {
            vortex_ensure!(
                partial.dtype() == &self.partial_dtype,
                "Aggregate returned incorrect DType on partial_scalar: expected {}, got {}",
                self.partial_dtype,
                partial.dtype(),
            );
        }

        Ok(partial)
    }

    fn final_scalar(&self) -> VortexResult<Scalar> {
        let result = self.vtable.finalize_scalar(&self.partial)?;

        vortex_ensure!(
            result.dtype() == &self.return_dtype,
            "Aggregate returned incorrect DType on final_scalar: expected {}, got {}",
            self.return_dtype,
            result.dtype(),
        );

        Ok(result)
    }

    fn flush(&mut self) -> VortexResult<Scalar> {
        let partial = self.partial_scalar()?;
        self.reset();
        Ok(partial)
    }

    fn finish(&mut self) -> VortexResult<Scalar> {
        let result = self.final_scalar()?;
        self.reset();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::combined::Combined;
    use crate::aggregate_fn::combined::PairOptions;
    use crate::aggregate_fn::fns::mean::Mean;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::aggregate_fn::session::AggregateFnSession;
    use crate::array::VTable;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;

    /// Mean partial sentinel `{sum: 42.0, count: 1}` — distinguishable from the
    /// natural fan-out result `{sum: 7.0, count: 1}` that `Combined::try_accumulate`
    /// would produce for `dict_of_seven()`.
    #[derive(Debug)]
    struct SentinelMeanPartialKernel;
    impl DynAggregateKernel for SentinelMeanPartialKernel {
        fn aggregate(
            &self,
            _aggregate_fn: &AggregateFnRef,
            _batch: &ArrayRef,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<Scalar>> {
            Ok(Some(sentinel_partial()))
        }
    }

    /// Returns `Ok(None)` => kernel declined, dispatch falls through.
    #[derive(Debug)]
    struct DeclineKernel;
    impl DynAggregateKernel for DeclineKernel {
        fn aggregate(
            &self,
            _aggregate_fn: &AggregateFnRef,
            _batch: &ArrayRef,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<Scalar>> {
            Ok(None)
        }
    }

    /// Sum partial sentinel `42.0` — distinguishable from the natural Sum of
    /// `dict_of_seven()` which is `7.0`.
    #[derive(Debug)]
    struct SentinelSumPartialKernel;
    impl DynAggregateKernel for SentinelSumPartialKernel {
        fn aggregate(
            &self,
            _aggregate_fn: &AggregateFnRef,
            _batch: &ArrayRef,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<Scalar>> {
            Ok(Some(Scalar::primitive(42.0f64, Nullability::Nullable)))
        }
    }

    fn fresh_session() -> VortexSession {
        crate::array_session()
    }

    fn dict_of_seven() -> ArrayRef {
        DictArray::try_new(buffer![0u32].into_array(), buffer![7.0f64].into_array())
            .expect("valid dictionary")
            .into_array()
    }

    fn mean_f64_accumulator() -> VortexResult<Accumulator<Combined<Mean>>> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        Accumulator::try_new(
            Mean::combined(),
            PairOptions(
                NumericalAggregateOpts::default(),
                NumericalAggregateOpts::default(),
            ),
            dtype,
        )
    }

    fn sentinel_partial() -> Scalar {
        let acc = mean_f64_accumulator().expect("build accumulator");
        let sum = Scalar::primitive(42.0f64, Nullability::Nullable);
        let count = Scalar::primitive(1u64, Nullability::NonNullable);
        Scalar::struct_(acc.partial_dtype, vec![sum, count])
    }

    /// Kernel registered for `(Dict, Combined<Mean>)` fires in preference to
    /// `Combined::try_accumulate`'s fan-out path — proves the dispatch reorder.
    #[test]
    fn combined_kernel_fires() -> VortexResult<()> {
        static KERNEL: SentinelMeanPartialKernel = SentinelMeanPartialKernel;
        let session = fresh_session();
        session
            .get::<AggregateFnSession>()
            .register_aggregate_kernel(Dict.id(), Some(Mean::combined().id()), &KERNEL);
        let mut ctx = session.create_execution_ctx();

        let mut acc = mean_f64_accumulator()?;
        acc.accumulate(&dict_of_seven(), &mut ctx)?;
        let partial = acc.flush()?;

        let s = partial.as_struct();
        assert_eq!(
            s.field("sum").unwrap().as_primitive().as_::<f64>(),
            Some(42.0)
        );
        assert_eq!(
            s.field("count").unwrap().as_primitive().as_::<u64>(),
            Some(1)
        );
        Ok(())
    }

    /// Kernel returns `Ok(None)` => dispatch falls through to `Combined::try_accumulate`'s
    /// natural fan-out. The natural partial is `{sum: 7.0, count: 1}`.
    #[test]
    fn fallback_when_kernel_declines() -> VortexResult<()> {
        static KERNEL: DeclineKernel = DeclineKernel;
        let session = fresh_session();
        session
            .get::<AggregateFnSession>()
            .register_aggregate_kernel(Dict.id(), Some(Mean::combined().id()), &KERNEL);
        let mut ctx = session.create_execution_ctx();

        let mut acc = mean_f64_accumulator()?;
        acc.accumulate(&dict_of_seven(), &mut ctx)?;
        let partial = acc.flush()?;

        let s = partial.as_struct();
        assert_eq!(
            s.field("sum").unwrap().as_primitive().as_::<f64>(),
            Some(7.0)
        );
        assert_eq!(
            s.field("count").unwrap().as_primitive().as_::<u64>(),
            Some(1)
        );
        Ok(())
    }

    /// A kernel registered for the inner `(Dict, Sum)` child fires when accumulating a
    /// Dict batch through `Combined<Mean>`. This is the reusable-primitive case the
    /// refactor enables: no `(Dict, Combined<Mean>)` kernel is needed.
    #[test]
    fn child_kernel_fires_through_combined() -> VortexResult<()> {
        static KERNEL: SentinelSumPartialKernel = SentinelSumPartialKernel;
        let session = fresh_session();
        session
            .get::<AggregateFnSession>()
            .register_aggregate_kernel(Dict.id(), Some(Sum.id()), &KERNEL);
        let mut ctx = session.create_execution_ctx();

        let mut acc = mean_f64_accumulator()?;
        acc.accumulate(&dict_of_seven(), &mut ctx)?;
        let partial = acc.flush()?;

        let s = partial.as_struct();
        // `Sum` child returned the sentinel 42.0 — proves the (Dict, Sum) kernel fired
        // via `Combined<Mean>`'s fan-out. `Count`'s native `try_accumulate` reads the
        // batch's valid_count, so count is the real 1.
        assert_eq!(
            s.field("sum").unwrap().as_primitive().as_::<f64>(),
            Some(42.0)
        );
        assert_eq!(
            s.field("count").unwrap().as_primitive().as_::<u64>(),
            Some(1)
        );
        Ok(())
    }
}
