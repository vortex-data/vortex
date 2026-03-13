// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Columnar;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::dtype::DType;
use crate::executor::MAX_ITERATIONS;
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

    /// Whether the accumulator's result is fully determined.
    fn is_saturated(&self) -> bool;

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

        let session = ctx.session().clone();
        let kernels = &session.aggregate_fns().kernels;

        let mut batch = batch.clone();
        for _ in 0..*MAX_ITERATIONS {
            if batch.is::<AnyCanonical>() {
                break;
            }

            let kernels_r = kernels.read();
            let batch_id = batch.encoding_id();
            if let Some(result) = kernels_r
                .get(&(batch_id.clone(), Some(self.aggregate_fn.id())))
                .or_else(|| kernels_r.get(&(batch_id, None)))
                .and_then(|kernel| {
                    kernel
                        .aggregate(&self.aggregate_fn, &batch, ctx)
                        .transpose()
                })
                .transpose()?
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

            // Execute one step and try again
            batch = batch.execute(ctx)?;
        }

        // Otherwise, execute the batch until it is columnar and accumulate it into the state.
        let columnar = batch.execute::<Columnar>(ctx)?;

        self.vtable.accumulate(&mut self.partial, &columnar, ctx)
    }

    fn is_saturated(&self) -> bool {
        self.vtable.is_saturated(&self.partial)
    }

    fn flush(&mut self) -> VortexResult<Scalar> {
        let partial = self.vtable.flush(&mut self.partial)?;

        #[cfg(debug_assertions)]
        {
            vortex_ensure!(
                partial.dtype() == &self.partial_dtype,
                "Aggregate kernel returned incorrect DType on flush: expected {}, got {}",
                self.partial_dtype,
                partial.dtype(),
            );
        }

        Ok(partial)
    }

    fn finish(&mut self) -> VortexResult<Scalar> {
        let partial = self.flush()?;
        let result = self.vtable.finalize_scalar(partial)?;

        vortex_ensure!(
            result.dtype() == &self.return_dtype,
            "Aggregate kernel returned incorrect DType on finalize: expected {}, got {}",
            self.return_dtype,
            result.dtype(),
        );

        Ok(result)
    }
}
