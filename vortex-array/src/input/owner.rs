// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::OnceLock;

use vortex_error::VortexResult;

use super::AggregateCacheMode;
use super::ArrayInput;
use super::ArrayInputInner;
use super::VerifiedIntegerBounds;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::builtins::ArrayBuiltins;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::stats::AggregateResults;
use crate::stats::Aggregations;
use crate::stats::AggregationsRef;

impl ArrayInput {
    /// Own an input without computing any aggregates.
    pub fn new(array: ArrayRef) -> Self {
        Self {
            inner: Arc::new(ArrayInputInner {
                array,
                aggregations: Aggregations::default(),
                verified_bounds: OnceLock::new(),
            }),
            cache_mode: AggregateCacheMode::Input,
        }
    }

    /// Select the result store used by this handle and its scoped execution.
    ///
    /// Other clones keep their mode and share the same stores. Selecting a mode performs no work.
    pub fn with_cache_mode(mut self, mode: AggregateCacheMode) -> Self {
        self.cache_mode = mode;
        self
    }

    /// Return the exact array whose results this owner retains.
    pub fn array(&self) -> &ArrayRef {
        &self.inner.array
    }

    /// Release this owner and return a shared array handle.
    pub fn into_array(self) -> ArrayRef {
        self.inner.array.clone()
    }

    /// Return a retained finalized result or bound without reading values or computing a miss.
    pub fn get_result(&self, aggregate: &AggregateFnRef) -> Precision<Scalar> {
        match self.cache_mode {
            AggregateCacheMode::Array => self.array().aggregations().get_result(aggregate),
            AggregateCacheMode::Input => self.aggregations().get_result(aggregate),
            AggregateCacheMode::Disabled => Precision::Absent,
        }
    }

    /// Compute an exact finalized result, retaining it in the selected store.
    ///
    /// A bound cannot answer an exact request. Errors are not retained. Generic registered kernels
    /// keep their normal dispatch behavior and cannot establish an unchecked cast's safety proof.
    pub fn compute_result(
        &self,
        aggregate: &AggregateFnRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let mut scoped = ctx.with_aggregate_input(self);
        match self.cache_mode {
            AggregateCacheMode::Array => {
                if let Precision::Exact(result) = self.array().aggregations().get_result(aggregate)
                {
                    return Ok(result);
                }
                if let Some(result) = self.aggregations().finalize_retained_result(aggregate)? {
                    self.array()
                        .aggregations()
                        .insert_result(aggregate.clone(), Precision::Exact(result.clone()))?;
                    return Ok(result);
                }
                self.array()
                    .aggregations()
                    .compute_result(aggregate, &mut scoped)
            }
            AggregateCacheMode::Input => self.aggregations().compute_result(aggregate, &mut scoped),
            AggregateCacheMode::Disabled => {
                let mut accumulator = aggregate.accumulator(self.array().dtype())?;
                accumulator.accumulate_uncached(self.array(), &mut scoped)?;
                accumulator.finish()
            }
        }
    }

    /// Copy finalized results from the selected store into an immutable summary.
    ///
    /// The summary contains no typed partial states or cast proof tokens.
    pub fn snapshot_results(&self) -> AggregateResults {
        match self.cache_mode {
            AggregateCacheMode::Array => self.array().aggregations().snapshot_results(),
            AggregateCacheMode::Input => self.aggregations().snapshot_results(),
            AggregateCacheMode::Disabled => AggregateResults::default(),
        }
    }

    /// Execute a cast to completion while this input's facts remain available.
    ///
    /// Lazy operations returned by ordinary array APIs retain no input owner. Physical rewrites
    /// conservatively miss identity-bound facts rather than attaching them to unrelated handles.
    pub fn execute_cast(
        &self,
        dtype: crate::dtype::DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Canonical> {
        self.array()
            .cast(dtype)?
            .execute(&mut ctx.with_aggregate_input(self))
    }

    /// Take from this input using an explicitly retained indices owner, and finish execution.
    ///
    /// Both owners remain available to kernels and retain their selected cache modes. If source
    /// and indices share the exact array handle, the indices binding takes precedence.
    pub fn execute_take(&self, indices: &Self, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        self.array()
            .take(indices.array().clone())?
            .execute(&mut ctx.with_aggregate_input(self).with_aggregate_input(indices))
    }

    pub(crate) fn aggregations(&self) -> AggregationsRef<'_> {
        self.inner.aggregations.to_ref(self.array())
    }

    pub(super) fn cache_mode(&self) -> AggregateCacheMode {
        self.cache_mode
    }

    pub(super) fn verified_bounds(&self) -> Option<&VerifiedIntegerBounds> {
        if self.cache_mode == AggregateCacheMode::Disabled {
            return None;
        }
        self.inner.verified_bounds.get()
    }
}
