// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateDTypes;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::demand::valid_demand;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::builtins::ArrayBuiltins;
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
    /// The options of the aggregate function.
    options: V::Options,
    /// Type-erased aggregate function used for kernel dispatch.
    aggregate_fn: AggregateFnRef,
    /// The input, partial, and result dtypes lent to every vtable call.
    dtypes: AggregateDTypes,
    /// The partial state of the accumulator, with `None` as the empty group.
    partial: Option<V::Partial>,
    /// The dtypes of the nullable input, lent to the vtable for a batch whose undemanded rows were
    /// masked to null. Resolved on the first such batch.
    masked_dtypes: Option<AggregateDTypes>,
}

impl<V: AggregateFnVTable> Accumulator<V> {
    pub fn try_new(vtable: V, options: V::Options, dtype: DType) -> VortexResult<Self> {
        let dtypes = AggregateDTypes::try_new(&vtable, &options, dtype)?;

        Ok(Self::from_dtypes(vtable, options, dtypes))
    }

    /// Build an accumulator over dtypes that are already resolved.
    ///
    /// A nested accumulator takes the dtypes its parent already derived, instead of resolving
    /// them again for every batch.
    pub fn from_dtypes(vtable: V, options: V::Options, dtypes: AggregateDTypes) -> Self {
        let aggregate_fn = AggregateFn::new(vtable.clone(), options.clone()).erased();

        Self {
            vtable,
            options,
            aggregate_fn,
            dtypes,
            partial: None,
            masked_dtypes: None,
        }
    }

    /// Resolve the dtypes for a batch masked to the nullable input dtype.
    ///
    /// Masking only widens the input's nullability, which must not change the partial state.
    fn resolve_masked_dtypes(&mut self) -> VortexResult<()> {
        if self.masked_dtypes.is_some() {
            return Ok(());
        }
        let nullable = self.dtypes.dtype.as_nullable();
        let dtypes = if nullable == self.dtypes.dtype {
            self.dtypes.clone()
        } else {
            AggregateDTypes::try_new(&self.vtable, &self.options, nullable)?
        };
        vortex_ensure_eq!(
            dtypes.partial_dtype,
            self.dtypes.partial_dtype,
            "Aggregate {} has a different partial dtype for a nullable input",
            self.aggregate_fn,
        );
        self.masked_dtypes = Some(dtypes);
        Ok(())
    }

    /// The state of a group with no accumulated values.
    pub(crate) fn empty_partial(&self) -> VortexResult<V::Partial> {
        self.vtable.empty_partial(self.dtypes.args(&self.options))
    }

    fn ensure_partial(&mut self) -> VortexResult<()> {
        if self.partial.is_none() {
            self.partial = Some(self.empty_partial()?);
        }
        Ok(())
    }

    pub(crate) fn fold_partial(&mut self, other: V::Partial) -> VortexResult<()> {
        self.partial = Some(match self.partial.take() {
            // The empty state is the identity of the merge.
            None => other,
            Some(current) => {
                self.vtable
                    .merge_partials(self.dtypes.args(&self.options), current, other)?
            }
        });
        Ok(())
    }

    fn fold_partial_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        let other = self
            .vtable
            .partial_from_scalar(self.dtypes.args(&self.options), scalar)?;
        self.fold_partial(other)
    }

    /// Accumulate `batch` into the partial state through kernel dispatch, the vtable's
    /// short-circuit, or execution to a columnar array, without consulting cached statistics.
    ///
    /// With `masked`, `batch` was masked to the nullable input dtype, and the vtable is lent the
    /// resolved [`masked_dtypes`](Self::resolve_masked_dtypes).
    fn accumulate_batch(
        &mut self,
        batch: &ArrayRef,
        masked: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
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
                vortex_ensure_eq!(
                    result.dtype(),
                    &self.dtypes.partial_dtype,
                    "Aggregate kernel returned the wrong partial dtype",
                );
                self.fold_partial_scalar(&result)?;
                return Ok(());
            }
        }

        // 2. Allow the vtable to short-circuit on the raw array before decompression.
        self.ensure_partial()?;
        let dtypes = batch_dtypes(&self.dtypes, self.masked_dtypes.as_ref(), masked);
        let partial = self.partial.as_mut().vortex_expect("partial materialized");
        if self
            .vtable
            .try_accumulate(dtypes.args(&self.options), partial, batch, ctx)?
        {
            return Ok(());
        }

        // 3. An aggregate that skips nulls only needs the valid rows. Like a row function's
        //    valid-only policy, a partially valid batch is filtered to its valid rows once, here,
        //    after the direct paths above had their chance on the original batch, so the steps
        //    below see an all-valid batch and the kernels take their dense paths. A batch that has
        //    definitely no nulls never resolves its validity.
        let Some(mut batch) = self.valid_rows(batch, ctx)? else {
            return Ok(());
        };

        // 4. Iteratively check the registry against each intermediate encoding, executing one
        //    step between checks. Mirrors the loop in `GroupedAccumulator::accumulate_list_view`.
        //    Iteration 0 re-checks the initial encoding — a redundant HashMap miss, the price of
        //    keeping the loop body uniform. Terminates on `AnyColumnar` (Canonical or Constant)
        //    since the vtable's `accumulate(&Columnar)` handles both cases directly.
        for _ in 0..max_iterations() {
            if batch.is::<AnyColumnar>() {
                break;
            }

            if let Some(kernel) = session
                .aggregate_fns()
                .find_aggregate_kernel(batch.encoding_id(), self.aggregate_fn.id())
                && let Some(result) = kernel.aggregate(&self.aggregate_fn, &batch, ctx)?
            {
                vortex_ensure_eq!(
                    result.dtype(),
                    &self.dtypes.partial_dtype,
                    "Aggregate kernel returned the wrong partial dtype",
                );
                self.fold_partial_scalar(&result)?;
                return Ok(());
            }

            batch = batch.execute(ctx)?;
        }

        // 5. Otherwise, execute the batch until it is columnar and accumulate it into the state.
        let columnar = batch.execute::<Columnar>(ctx)?;

        self.ensure_partial()?;
        let dtypes = batch_dtypes(&self.dtypes, self.masked_dtypes.as_ref(), masked);
        let partial = self.partial.as_mut().vortex_expect("partial materialized");
        self.vtable
            .accumulate(dtypes.args(&self.options), partial, &columnar, ctx)
    }
}

/// The dtypes lent to the vtable for a batch, which are the nullable input's for a `masked` batch.
fn batch_dtypes<'a>(
    dtypes: &'a AggregateDTypes,
    masked_dtypes: Option<&'a AggregateDTypes>,
    masked: bool,
) -> &'a AggregateDTypes {
    match masked_dtypes {
        Some(masked_dtypes) if masked => masked_dtypes,
        _ => dtypes,
    }
}

impl<V: AggregateFnVTable> Accumulator<V> {
    /// The rows of `batch` the aggregate needs: its valid rows if the aggregate
    /// [skips nulls](AggregateFnVTable::SKIPS_NULLS), otherwise all of them. Returns `None` if no
    /// row is needed.
    fn valid_rows(
        &self,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !V::SKIPS_NULLS {
            return Ok(Some(batch.clone()));
        }
        let validity = batch.validity()?;
        if validity.definitely_no_nulls() {
            return Ok(Some(batch.clone()));
        }
        Ok(match validity.execute_mask(batch.len(), ctx)? {
            Mask::AllTrue(_) => Some(batch.clone()),
            Mask::AllFalse(_) => None,
            valid @ Mask::Values(_) => Some(batch.filter(valid)?),
        })
    }

    /// Accumulate every row of `batch`, consuming or caching its result as a legacy statistic.
    fn accumulate_all(&mut self, batch: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        // Legacy stats bridge: an aggregate still cached under a legacy `Stat` slot consumes that
        // exact stat before kernel dispatch or decode, and otherwise caches its result there.
        let Some(stat) = Stat::from_aggregate_fn(&self.aggregate_fn) else {
            return self.accumulate_batch(batch, false, ctx);
        };

        if let Precision::Exact(partial) = batch.statistics().get(stat) {
            let partial = if partial.dtype() == &self.dtypes.partial_dtype {
                partial
            } else {
                vortex_ensure!(
                    partial
                        .dtype()
                        .eq_ignore_nullability(&self.dtypes.partial_dtype),
                    "Aggregate {} read legacy stat {} with dtype {}, expected {}",
                    self.aggregate_fn,
                    stat,
                    partial.dtype(),
                    self.dtypes.partial_dtype,
                );
                partial.cast(&self.dtypes.partial_dtype)?
            };
            self.fold_partial_scalar(&partial)?;
            return Ok(());
        }

        // Accumulate the batch into an empty state of its own, so that its result is cached on
        // the requested batch rather than on whatever array the batch executes into.
        let running = self.partial.take();
        let accumulated = self.accumulate_batch(batch, false, ctx);
        let batch_partial = std::mem::replace(&mut self.partial, running);
        accumulated?;

        let Some(batch_partial) = batch_partial else {
            return Ok(());
        };
        let batch_result = self
            .vtable
            .to_scalar(self.dtypes.args(&self.options), &batch_partial)?;
        // A null partial, e.g. an overflowed sum or the minimum of an all-null batch, has no exact
        // stat value.
        if let Some(value) = batch_result.into_value() {
            batch.statistics().set(stat, Precision::Exact(value));
        }
        self.fold_partial(batch_partial)
    }
}

/// A trait object for type-erased accumulators, used for dynamic dispatch when the aggregate
/// function is not known at compile time.
pub trait DynAccumulator: 'static + Send {
    /// Accumulate every row of `batch` into the accumulator's state.
    fn accumulate(&mut self, batch: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        self.accumulate_demanded(batch, &Mask::new_true(batch.len()), ctx)
    }

    /// Accumulate `batch` into the accumulator's state, as null wherever `demand` is false.
    ///
    /// `demand` must have the length of `batch`. Undemanded rows are null whatever they hold, e.g.
    /// the unmasked field values under the null rows of a parent struct, so the result equals
    /// accumulating `batch.mask(demand)`: the null count counts them, while aggregates that skip
    /// nulls ignore them.
    ///
    /// An aggregate that [skips nulls](AggregateFnVTable::SKIPS_NULLS) receives the batch lazily
    /// filtered to its valid demanded rows, so its kernels take their dense paths and filters that
    /// encodings push down keep their kernels reachable, e.g. a dictionary filters only its codes.
    /// Every other aggregate receives the masked batch, whose validity is the batch's validity
    /// conjoined with the demand.
    fn accumulate_demanded(
        &mut self,
        batch: &ArrayRef,
        demand: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Drain another accumulator's state into this one, resetting `other`.
    ///
    /// The other accumulator must have been constructed for the same aggregate function,
    /// options, and input dtype as this one.
    fn merge_from(&mut self, other: &mut dyn DynAccumulator) -> VortexResult<()>;

    /// Parse a partial scalar and merge it into this accumulator's state.
    ///
    /// The scalar must have the dtype reported by the vtable's `partial_dtype`, and represents
    /// input following the input already accumulated.
    fn combine_partials(&mut self, partial: &Scalar) -> VortexResult<()>;

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

    /// Access the accumulator as [`Any`], so it can be downcast to a typed [`Accumulator`].
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl dyn DynAccumulator {
    /// Downcast to the typed [`Accumulator`] of the aggregate vtable `V`.
    pub fn downcast_mut<V: AggregateFnVTable>(&mut self) -> Option<&mut Accumulator<V>> {
        self.as_any_mut().downcast_mut()
    }
}

impl<V: AggregateFnVTable> DynAccumulator for Accumulator<V> {
    fn merge_from(&mut self, other: &mut dyn DynAccumulator) -> VortexResult<()> {
        let Some(other) = other.downcast_mut::<V>() else {
            vortex_bail!(
                "Cannot merge into a {} accumulator from an accumulator of a different aggregate",
                self.aggregate_fn,
            );
        };
        vortex_ensure!(
            other.options == self.options && other.dtypes.dtype == self.dtypes.dtype,
            "Cannot merge {} accumulators with different options or input dtypes",
            self.aggregate_fn,
        );
        match other.partial.take() {
            Some(partial) => self.fold_partial(partial),
            None => Ok(()),
        }
    }

    fn accumulate_demanded(
        &mut self,
        batch: &ArrayRef,
        demand: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if self.is_saturated() {
            return Ok(());
        }
        vortex_ensure_eq!(batch.dtype(), &self.dtypes.dtype, "Input DType mismatch");
        vortex_ensure_eq!(
            demand.len(),
            batch.len(),
            "Demand mask length must match the batch length",
        );
        if demand.all_true() {
            return self.accumulate_all(batch, ctx);
        }

        // Undemanded rows are null, so the demand merges into the batch's validity. The masked or
        // filtered batch is a temporary array, so no statistics are read from or cached on it.
        if V::SKIPS_NULLS {
            // Only the valid rows matter, so drop the undemanded and the null rows in one filter,
            // leaving the kernel an all-valid batch.
            let rows = valid_demand(&batch.validity()?, demand, ctx)?;
            if rows.all_false() {
                return Ok(());
            }
            return self.accumulate_batch(&batch.filter(rows)?, false, ctx);
        }
        // The aggregate observes nulls, e.g. the null count, so it counts the undemanded rows as
        // nulls of the masked batch.
        self.resolve_masked_dtypes()?;
        let masked = batch.clone().mask(demand.clone().into_array())?;
        self.accumulate_batch(&masked, true, ctx)
    }

    fn combine_partials(&mut self, partial: &Scalar) -> VortexResult<()> {
        vortex_ensure_eq!(
            partial.dtype(),
            &self.dtypes.partial_dtype,
            "Partial DType mismatch for {}",
            self.aggregate_fn,
        );
        self.fold_partial_scalar(partial)
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_saturated(&self) -> bool {
        self.partial.as_ref().is_some_and(|partial| {
            self.vtable
                .is_saturated(self.dtypes.args(&self.options), partial)
        })
    }

    fn reset(&mut self) {
        self.partial = None;
    }

    fn partial_scalar(&self) -> VortexResult<Scalar> {
        let args = self.dtypes.args(&self.options);
        let partial = match &self.partial {
            Some(partial) => self.vtable.to_scalar(args, partial)?,
            None => self.vtable.to_scalar(args, &self.empty_partial()?)?,
        };

        #[cfg(debug_assertions)]
        {
            vortex_ensure_eq!(
                partial.dtype(),
                args.partial_dtype,
                "Aggregate returned incorrect DType on partial_scalar",
            );
        }

        Ok(partial)
    }

    fn final_scalar(&self) -> VortexResult<Scalar> {
        let args = self.dtypes.args(&self.options);
        let result = match &self.partial {
            Some(partial) => self.vtable.finalize_scalar(args, partial)?,
            None => self.vtable.finalize_scalar(args, &self.empty_partial()?)?,
        };

        vortex_ensure_eq!(
            result.dtype(),
            args.return_dtype,
            "Aggregate returned incorrect DType on final_scalar",
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
    use vortex_session::SessionExt;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AccumulatorRef;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::combined::Combined;
    use crate::aggregate_fn::combined::PairOptions;
    use crate::aggregate_fn::fns::mean::Mean;
    use crate::aggregate_fn::fns::min::Min;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::aggregate_fn::session::AggregateFnSession;
    use crate::array::VTable;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::expr::stats::StatsProviderExt;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

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

    /// A sentinel sum of `42.0`, distinguishable from `dict_of_seven()`'s natural sum of `7.0`.
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
        Scalar::struct_(acc.dtypes.partial_dtype, vec![sum, count])
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

    #[test]
    fn cached_sum_precedes_encoding_kernel() -> VortexResult<()> {
        static KERNEL: SentinelSumPartialKernel = SentinelSumPartialKernel;
        let session = fresh_session();
        session
            .get::<AggregateFnSession>()
            .register_aggregate_kernel(Dict.id(), Some(Sum.id()), &KERNEL);
        let mut ctx = session.create_execution_ctx();

        let batch = dict_of_seven();
        batch
            .statistics()
            .set(Stat::Sum, Precision::Exact(ScalarValue::from(11.0f64)));

        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;
        acc.accumulate(&batch, &mut ctx)?;

        assert_eq!(acc.finish()?.as_primitive().as_::<f64>(), Some(11.0));
        Ok(())
    }

    /// Each batch caches its own result, not the running result and not on the canonical array
    /// it executes into.
    #[test]
    fn caches_result_on_requested_batch() -> VortexResult<()> {
        let mut ctx = fresh_session().create_execution_ctx();
        let dict = |codes: ArrayRef| -> VortexResult<ArrayRef> {
            Ok(DictArray::try_new(codes, buffer![5i32, 1, 9].into_array())?.into_array())
        };
        let batch1 = dict(buffer![0u32, 1, 2].into_array())?;
        let batch2 = dict(buffer![2u32, 0].into_array())?;

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Min, NumericalAggregateOpts::default(), dtype)?;
        acc.accumulate(&batch1, &mut ctx)?;
        acc.accumulate(&batch2, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(1i32, Nullability::Nullable)
        );
        assert_eq!(
            batch1.statistics().get_as::<i32>(Stat::Min),
            Precision::exact(1i32)
        );
        assert_eq!(
            batch2.statistics().get_as::<i32>(Stat::Min),
            Precision::exact(5i32)
        );
        Ok(())
    }

    #[test]
    fn overflowed_sum_is_not_cached() -> VortexResult<()> {
        let mut ctx = fresh_session().create_execution_ctx();
        let batch = buffer![i64::MAX, i64::MAX].into_array();

        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;
        acc.accumulate(&batch, &mut ctx)?;

        assert!(acc.finish()?.is_null());
        assert!(batch.statistics().get(Stat::Sum).as_exact().is_none());
        Ok(())
    }

    fn sum_i32_accumulator(options: NumericalAggregateOpts) -> VortexResult<AccumulatorRef> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        Ok(Box::new(Accumulator::try_new(Sum, options, dtype)?))
    }

    #[test]
    fn merge_from_drains_other_accumulator() -> VortexResult<()> {
        let mut ctx = fresh_session().create_execution_ctx();
        let mut global = sum_i32_accumulator(NumericalAggregateOpts::default())?;
        let mut local = sum_i32_accumulator(NumericalAggregateOpts::default())?;

        global.accumulate(&buffer![10i32, 20].into_array(), &mut ctx)?;
        local.accumulate(&buffer![5i32].into_array(), &mut ctx)?;
        global.merge_from(local.as_mut())?;

        assert_eq!(
            global.finish()?,
            Scalar::primitive(35i64, Nullability::Nullable)
        );
        // The merged-from accumulator is reset back to the empty state.
        assert_eq!(
            local.finish()?,
            Scalar::primitive(0i64, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn merge_from_rejects_a_different_aggregate() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut sum = sum_i32_accumulator(NumericalAggregateOpts::default())?;
        let mut min: AccumulatorRef = Box::new(Accumulator::try_new(
            Min,
            NumericalAggregateOpts::default(),
            dtype,
        )?);

        assert!(sum.merge_from(min.as_mut()).is_err());
        Ok(())
    }

    #[test]
    fn merge_from_rejects_mismatched_options() -> VortexResult<()> {
        let mut skipping = sum_i32_accumulator(NumericalAggregateOpts::skip_nans())?;
        let mut including = sum_i32_accumulator(NumericalAggregateOpts::include_nans())?;

        assert!(skipping.merge_from(including.as_mut()).is_err());
        Ok(())
    }

    #[test]
    fn merge_from_combines_child_accumulators() -> VortexResult<()> {
        let mut ctx = fresh_session().create_execution_ctx();
        let mut global: AccumulatorRef = Box::new(mean_f64_accumulator()?);
        let mut local: AccumulatorRef = Box::new(mean_f64_accumulator()?);

        global.accumulate(&buffer![1.0f64, 2.0].into_array(), &mut ctx)?;
        local.accumulate(&buffer![6.0f64].into_array(), &mut ctx)?;
        global.merge_from(local.as_mut())?;

        assert_eq!(global.finish()?.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }
}
