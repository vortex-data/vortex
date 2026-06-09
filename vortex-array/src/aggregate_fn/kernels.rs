// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pluggable aggregate function kernels used to provide encoding-specific implementations of
//! aggregate functions.

use std::fmt::Debug;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupedArray;
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

/// A pluggable kernel for batch aggregation of many groups.
///
/// A kernel can be registered either for an aggregate function regardless of the element encoding,
/// or for a specific aggregate function and element encoding. Element-encoding kernels are matched
/// on the inner array of the provided grouped array, not on the outer list encoding. This is more
/// pragmatic than having every kernel match on the outer list encoding and having to deal with the
/// possibility of multiple list encodings.
///
/// Each value in the grouped array represents a group and the result of the grouped aggregate
/// should be an array of the same length, where each element is the aggregate state of the
/// corresponding group.
///
/// Return `Ok(None)` if the kernel cannot be applied to the given aggregate function.
pub trait DynGroupedAggregateKernel: 'static + Send + Sync + Debug {
    /// Aggregate each group in the provided grouped array and return an array of the aggregate
    /// states.
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}
