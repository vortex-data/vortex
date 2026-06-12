// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::kernels::GroupedAggregateKernelResult;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::builders::builder_with_capacity;
use crate::columnar::AnyColumnar;
use crate::dtype::DType;
use crate::executor::max_iterations;
use crate::scalar::Scalar;

/// Reference-counted type-erased grouped accumulator.
pub type GroupedAccumulatorRef = Box<dyn DynGroupedAccumulator>;

/// An accumulator used for computing aggregates over dense group ids.
///
/// Group ids are caller-assigned `u32` ordinals in the dense range `0..num_groups`. Input batches
/// may repeat, omit, and reorder those ids, but every id must identify a state slot rather than a
/// raw group key. The accumulator keeps one partial state per slot, so ordered and unordered
/// grouping only differ in how the caller assigns ids.
pub struct GroupedAccumulator<V: AggregateFnVTable> {
    /// The vtable of the aggregate function.
    vtable: V,
    /// The options of the aggregate function.
    options: V::Options,
    /// Type-erased aggregate function used for kernel dispatch.
    aggregate_fn: AggregateFnRef,
    /// The DType of the input.
    dtype: DType,
    /// The DType of the aggregate.
    return_dtype: DType,
    /// The DType of the partial accumulator state.
    partial_dtype: DType,
    /// Dense per-group partial state.
    partials: Vec<V::Partial>,
}

impl<V: AggregateFnVTable> GroupedAccumulator<V> {
    pub fn try_new(vtable: V, options: V::Options, dtype: DType) -> VortexResult<Self> {
        let aggregate_fn = AggregateFn::new(vtable.clone(), options.clone()).erased();
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

        Ok(Self {
            vtable,
            options,
            aggregate_fn,
            dtype,
            return_dtype,
            partial_dtype,
            partials: Vec::new(),
        })
    }

    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()> {
        validate_num_groups(num_groups)?;

        while self.partials.len() < num_groups {
            self.partials
                .push(self.vtable.empty_partial(&self.options, &self.dtype)?);
        }
        Ok(())
    }

    fn validate_group_ids(&self, group_ids: &[u32], num_groups: usize) -> VortexResult<()> {
        validate_num_groups(num_groups)?;
        for &group_id in group_ids {
            vortex_ensure!(
                (group_id as usize) < num_groups,
                "Group id {} out of range for {} groups",
                group_id,
                num_groups
            );
        }
        Ok(())
    }

    fn accumulate_kernel_result(
        &mut self,
        result: GroupedAggregateKernelResult,
        num_groups: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.accumulate_partials(result.partials(), result.group_ids(), num_groups, ctx)
    }

    fn try_accumulate_kernel(
        &mut self,
        batch: &ArrayRef,
        group_ids: &[u32],
        num_groups: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let session = ctx.session().clone();

        if let Some(kernel) = session
            .aggregate_fns()
            .find_grouped_encoding_kernel(batch.encoding_id(), self.aggregate_fn.id())
            && let Some(result) =
                kernel.grouped_aggregate(&self.aggregate_fn, batch, group_ids, num_groups, ctx)?
        {
            self.accumulate_kernel_result(result, num_groups, ctx)?;
            return Ok(true);
        }

        if let Some(kernel) = session
            .aggregate_fns()
            .find_grouped_kernel(self.aggregate_fn.id())
            && let Some(result) =
                kernel.grouped_aggregate(&self.aggregate_fn, batch, group_ids, num_groups, ctx)?
        {
            self.accumulate_kernel_result(result, num_groups, ctx)?;
            return Ok(true);
        }

        Ok(false)
    }

    fn accumulate_fallback(
        &mut self,
        batch: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some((&first, rest)) = group_ids.split_first() else {
            return Ok(());
        };
        let mut first = first;
        let mut last = first;
        for &group_id in rest {
            first = first.min(group_id);
            last = last.max(group_id);
        }

        let first = first as usize;
        let mut buckets = vec![Vec::new(); last as usize - first + 1];
        for (row_idx, &group_id) in group_ids.iter().enumerate() {
            buckets[group_id as usize - first].push(row_idx as u64);
        }

        for (offset, rows) in buckets.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }

            let group = first + offset;
            if self.vtable.is_saturated(&self.partials[group]) {
                continue;
            }

            let taken = batch.clone().take(Buffer::from_iter(rows).into_array())?;
            let mut accumulator = Accumulator::try_new(
                self.vtable.clone(),
                self.options.clone(),
                self.dtype.clone(),
            )?;
            accumulator.accumulate(&taken, ctx)?;
            let partial = accumulator.flush()?;
            self.vtable
                .combine_partials(&mut self.partials[group], partial)?;
        }
        Ok(())
    }
}

fn validate_num_groups(num_groups: usize) -> VortexResult<()> {
    vortex_ensure!(
        num_groups == 0 || u32::try_from(num_groups - 1).is_ok(),
        "num_groups {} exceeds dense u32 group id capacity",
        num_groups
    );
    Ok(())
}

/// A trait object for type-erased grouped accumulators, used for dynamic dispatch when the
/// aggregate function is not known at compile time.
pub trait DynGroupedAccumulator: 'static + Send {
    /// Accumulate a values batch into dense group state.
    ///
    /// `group_ids` is parallel to `batch`. Each id must be a caller-assigned group ordinal in
    /// `0..num_groups`; ids may repeat, appear out of order, or be absent from a given batch.
    fn accumulate(
        &mut self,
        batch: &ArrayRef,
        group_ids: &[u32],
        num_groups: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Fold columnar partial states into dense group state.
    ///
    /// `group_ids` is parallel to `partials` and follows the same dense ordinal contract as
    /// [`Self::accumulate`].
    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        num_groups: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Merge one group from another grouped accumulator into this accumulator.
    fn merge_group(
        &mut self,
        into: u32,
        other: &dyn DynGroupedAccumulator,
        from: u32,
    ) -> VortexResult<()>;

    /// Return this accumulator's partial dtype.
    fn partial_dtype(&self) -> &DType;

    /// Read one group's current partial state.
    fn partial_scalar(&self, group_id: u32) -> VortexResult<Scalar>;

    /// Finish the accumulation and return partial aggregate results for all groups.
    ///
    /// Resets the accumulator state for the next round of accumulation.
    fn flush_partials(&mut self, num_groups: usize) -> VortexResult<ArrayRef>;

    /// Finish the accumulation and return final aggregate results for all groups.
    ///
    /// Resets the accumulator state for the next round of accumulation.
    fn finish(&mut self, num_groups: usize) -> VortexResult<ArrayRef>;
}

impl<V: AggregateFnVTable> DynGroupedAccumulator for GroupedAccumulator<V> {
    fn accumulate(
        &mut self,
        batch: &ArrayRef,
        group_ids: &[u32],
        num_groups: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            batch.dtype() == &self.dtype,
            "Input DType mismatch: expected {}, got {}",
            self.dtype,
            batch.dtype()
        );
        vortex_ensure!(
            batch.len() == group_ids.len(),
            "Grouped aggregate input length mismatch: {} values, {} group ids",
            batch.len(),
            group_ids.len()
        );

        self.validate_group_ids(group_ids, num_groups)?;
        self.ensure_groups(num_groups)?;

        if self.try_accumulate_kernel(batch, group_ids, num_groups, ctx)? {
            return Ok(());
        }

        if self.vtable.try_accumulate_grouped(
            &mut self.partials[..num_groups],
            batch,
            group_ids,
            ctx,
        )? {
            return Ok(());
        }

        let input = batch.clone();
        let mut batch = batch.clone();
        for _ in 0..max_iterations() {
            if batch.is::<AnyColumnar>() {
                break;
            }

            if self.try_accumulate_kernel(&batch, group_ids, num_groups, ctx)? {
                return Ok(());
            }

            batch = batch.execute(ctx)?;
        }

        let columnar = batch.clone().execute::<Columnar>(ctx)?;
        if self.vtable.accumulate_grouped(
            &mut self.partials[..num_groups],
            &columnar,
            group_ids,
            ctx,
        )? {
            return Ok(());
        }

        self.accumulate_fallback(&input, group_ids, ctx)
    }

    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        num_groups: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            partials.dtype() == &self.partial_dtype,
            "Partial DType mismatch: expected {}, got {}",
            self.partial_dtype,
            partials.dtype()
        );
        vortex_ensure!(
            partials.len() == group_ids.len(),
            "Grouped aggregate partial length mismatch: {} partials, {} group ids",
            partials.len(),
            group_ids.len()
        );

        self.validate_group_ids(group_ids, num_groups)?;
        self.ensure_groups(num_groups)?;

        for (row_idx, &group_id) in group_ids.iter().enumerate() {
            let partial = partials.execute_scalar(row_idx, ctx)?;
            self.vtable
                .combine_partials(&mut self.partials[group_id as usize], partial)?;
        }
        Ok(())
    }

    fn merge_group(
        &mut self,
        into: u32,
        other: &dyn DynGroupedAccumulator,
        from: u32,
    ) -> VortexResult<()> {
        vortex_ensure!(
            other.partial_dtype() == &self.partial_dtype,
            "Partial DType mismatch: expected {}, got {}",
            self.partial_dtype,
            other.partial_dtype()
        );
        self.ensure_groups((into as usize) + 1)?;
        let partial = other.partial_scalar(from)?;
        self.vtable
            .combine_partials(&mut self.partials[into as usize], partial)
    }

    fn partial_dtype(&self) -> &DType {
        &self.partial_dtype
    }

    fn partial_scalar(&self, group_id: u32) -> VortexResult<Scalar> {
        if let Some(partial) = self.partials.get(group_id as usize) {
            self.vtable.to_scalar(partial)
        } else {
            let partial = self.vtable.empty_partial(&self.options, &self.dtype)?;
            self.vtable.to_scalar(&partial)
        }
    }

    fn flush_partials(&mut self, num_groups: usize) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            num_groups >= self.partials.len(),
            "Cannot flush {} groups after accumulating {} groups",
            num_groups,
            self.partials.len()
        );
        self.ensure_groups(num_groups)?;

        let mut states = builder_with_capacity(&self.partial_dtype, num_groups);
        for partial in &self.partials {
            states.append_scalar(&self.vtable.to_scalar(partial)?)?;
        }
        self.partials.clear();

        Ok(states.finish())
    }

    fn finish(&mut self, num_groups: usize) -> VortexResult<ArrayRef> {
        let states = self.flush_partials(num_groups)?;
        let results = self.vtable.finalize(states)?;

        vortex_ensure!(
            results.dtype() == &self.return_dtype,
            "Return DType mismatch: expected {}, got {}",
            self.return_dtype,
            results.dtype()
        );

        Ok(results)
    }
}
