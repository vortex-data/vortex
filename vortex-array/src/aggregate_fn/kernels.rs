// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pluggable aggregate function kernels used to provide encoding-specific implementations of
//! aggregate functions.

use std::fmt::Debug;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
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

/// Partial grouped aggregate output produced by an encoding-specific grouped kernel.
///
/// `group_ids` is parallel to `partials`: each row in `partials` is a partial state for the
/// corresponding dense group ordinal. The ids may repeat, omit, and reorder groups, but must be
/// valid slots in the accumulator's `0..num_groups` range. The grouped accumulator merges this
/// batch through `accumulate_partials`.
#[derive(Clone, Debug)]
pub struct GroupedAggregateKernelResult {
    group_ids: Buffer<u32>,
    partials: ArrayRef,
}

impl GroupedAggregateKernelResult {
    pub fn new(group_ids: Buffer<u32>, partials: ArrayRef) -> Self {
        Self {
            group_ids,
            partials,
        }
    }

    pub fn group_ids(&self) -> &[u32] {
        self.group_ids.as_ref()
    }

    pub fn partials(&self) -> &ArrayRef {
        &self.partials
    }
}

/// A pluggable kernel for batch aggregation of many groups.
///
/// A grouped kernel can be registered for an aggregate function regardless of input encoding, or
/// for a specific aggregate function and array encoding. Encoding-specific kernels are matched on
/// the values array, not on a pre-grouped list wrapper.
///
/// Kernels receive the same dense group ordinals that the caller passed to the grouped accumulator
/// and may aggregate directly in the encoded domain.
///
/// Return `Ok(None)` if the kernel cannot be applied to the given aggregate function.
pub trait DynGroupedAggregateKernel: 'static + Send + Sync + Debug {
    /// Aggregate values into a partial-state batch keyed by dense group ordinal.
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        group_ids: &[u32],
        num_groups: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<GroupedAggregateKernelResult>>;
}
