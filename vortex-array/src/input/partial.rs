// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Explicit retention of typed accumulator state.
//!
//! A retained state keeps information that finalization may discard. It uses the same store as
//! finalized results and has no serialization or subset propagation path.

use std::sync::Arc;

use vortex_error::VortexResult;

use super::AggregateCacheMode;
use super::ArrayInput;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;

impl ArrayInput {
    /// Compute and retain a typed partial state for this input.
    ///
    /// Enabled modes share the returned state across clones; disabled mode computes a fresh state.
    /// A prior finalized result does not replace accumulation because finalization can discard
    /// information needed by the partial. A later finalized request can finalize this retained
    /// state without reading values again. The state is never serialized
    /// or propagated to a subset. `V::Partial` must support shared access and should own its values
    /// independently of this owner. Typed states always stay on this owner, including in array
    /// cache mode, so a state may safely retain the underlying array.
    pub fn compute_partial<V>(
        &self,
        aggregate: &AggregateFn<V>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Arc<V::Partial>>
    where
        V: AggregateFnVTable,
        V::Partial: Sync,
    {
        let mut scoped = ctx.with_aggregate_input(self);
        match self.cache_mode() {
            AggregateCacheMode::Array | AggregateCacheMode::Input => {
                self.aggregations().compute_partial(aggregate, &mut scoped)
            }
            AggregateCacheMode::Disabled => {
                let mut accumulator = Accumulator::try_new(
                    aggregate.vtable().clone(),
                    aggregate.options().clone(),
                    self.array().dtype().clone(),
                )?;
                accumulator.accumulate_uncached(self.array(), &mut scoped)?;
                Ok(Arc::new(accumulator.into_partial()?))
            }
        }
    }
}
