// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pluggable aggregate function kernels used to provide encoding-specific implementations of
//! aggregate functions.

use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::GroupIds;
use crate::scalar::Scalar;

/// A pluggable kernel for an aggregate function.
///
/// The provided array should be aggregated into a single scalar representing the partial state
/// of a single group.
pub trait DynAggregateKernel: 'static + Send + Sync + Debug {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>>;
}

/// A typed grouped aggregate kernel.
///
/// Implementations receive the concrete aggregate options and typed partial state. Return
/// `Ok(false)` when the kernel cannot handle the current values or group-id encodings.
pub trait GroupedAggregateKernel<V: AggregateFnVTable>: 'static + Send + Sync + Debug {
    /// Accumulate `batch` into `states` according to `group_ids`.
    fn grouped_accumulate(
        &self,
        options: &V::Options,
        states: &mut [V::Partial],
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool>;
}

/// Bridges a typed [`GroupedAggregateKernel`] to type-erased grouped kernel dispatch.
pub struct GroupedAggregateKernelAdapter<V, K> {
    kernel: K,
    _phantom: PhantomData<fn() -> V>,
}

impl<V, K> GroupedAggregateKernelAdapter<V, K> {
    /// Create a new adapter around `kernel`.
    pub const fn new(kernel: K) -> Self {
        Self {
            kernel,
            _phantom: PhantomData,
        }
    }
}

impl<V, K> Debug for GroupedAggregateKernelAdapter<V, K>
where
    V: AggregateFnVTable,
    K: GroupedAggregateKernel<V>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedAggregateKernelAdapter")
            .field("kernel", &self.kernel)
            .finish()
    }
}

/// A pluggable kernel for batch aggregation of many groups.
///
/// A grouped kernel can be registered for an aggregate function regardless of input encodings, or
/// for a specific aggregate function plus values and/or group-id encoding.
///
/// Kernels receive the same dense group ordinals that the caller passed to the grouped accumulator
/// and may aggregate directly in the encoded domain.
///
/// Return `Ok(false)` if the kernel cannot be applied to the given aggregate function or input
/// encodings.
pub trait DynGroupedAggregateKernel: 'static + Send + Sync + Debug {
    /// Accumulate values into type-erased partial state.
    fn grouped_accumulate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        states: &mut dyn Any,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool>;
}

impl<V, K> DynGroupedAggregateKernel for GroupedAggregateKernelAdapter<V, K>
where
    V: AggregateFnVTable,
    K: GroupedAggregateKernel<V>,
{
    fn grouped_accumulate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        states: &mut dyn Any,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(options) = aggregate_fn.as_opt::<V>() else {
            return Ok(false);
        };

        let Some(states) = states.downcast_mut::<Vec<V::Partial>>() else {
            vortex_bail!(
                "Grouped aggregate kernel for {} received incompatible partial state",
                aggregate_fn.id()
            );
        };

        vortex_ensure!(
            states.len() >= group_ids.num_groups(),
            "Grouped aggregate kernel for {} received {} partial states for {} groups",
            aggregate_fn.id(),
            states.len(),
            group_ids.num_groups()
        );

        self.kernel.grouped_accumulate(
            options,
            &mut states[..group_ids.num_groups()],
            batch,
            group_ids,
            ctx,
        )
    }
}
