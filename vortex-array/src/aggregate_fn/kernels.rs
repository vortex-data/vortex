// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pluggable aggregate function kernels used to provide encoding-specific implementations of
//! aggregate functions.

use std::fmt::Debug;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::scalar::Scalar;

/// A pluggable kernel for an aggregate function.
///
/// The provided array should be aggregated into a single scalar representing the state of a single
/// group.
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
/// The kernel is matched on the encoding of the _elements_ array, which is the inner array of the
/// provided `ListViewArray`. This is more pragmatic than having every kernel match on the outer
/// list encoding and having to deal with the possibility of multiple list encodings.
///
/// Each element of the list array represents a group and the result of the grouped aggregate
/// should be an array of the same length, where each element is the aggregate state of the
/// corresponding group.
///
/// Return `Ok(None)` if the kernel cannot be applied to the given aggregate function.
pub trait DynGroupedAggregateKernel: 'static + Send + Sync + Debug {
    /// Aggregate each group in the provided `ListViewArray` and return an array of the
    /// aggregate states.
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &ListViewArray,
    ) -> VortexResult<Option<ArrayRef>>;

    /// Aggregate each group in the provided `FixedSizeListArray` and return an array of the
    /// aggregate states.
    fn grouped_aggregate_fixed_size(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &FixedSizeListArray,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(ngates): we could automatically delegate to `grouped_aggregate` if SequenceArray
        //  was in the vortex-array crate
        let _ = (aggregate_fn, groups);
        Ok(None)
    }
}
