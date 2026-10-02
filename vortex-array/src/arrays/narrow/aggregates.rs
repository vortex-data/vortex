// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate Narrow values without widening the input buffer.
//!
//! Supported aggregates preserve integer ordering or use wide sum states. Partial states are cast
//! to the logical dtype so later batches observe the same boundaries as canonical integers.

use std::sync::LazyLock;

use vortex_error::VortexResult;

use super::Narrow;
use super::NarrowArraySlotsExt;
use crate::ArrayRef;
use crate::ArrayVTable;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::first::First;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::last::Last;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::session::AggregateFnSession;
use crate::scalar::Scalar;

pub(crate) fn register_aggregate_kernels(session: &AggregateFnSession) {
    // One fallback avoids cloning the registry for each supported aggregate.
    session.register_aggregate_kernel(Narrow.id(), None::<AggregateFnId>, &NarrowAggregateKernel);
}

static SUPPORTED_AGGREGATES: LazyLock<[AggregateFnId; 13]> = LazyLock::new(|| {
    [
        MinMax.id(),
        Min.id(),
        Max.id(),
        Sum.id(),
        SumV2.id(),
        IsConstant.id(),
        IsSorted.id(),
        First.id(),
        Last.id(),
        Count.id(),
        NullCount.id(),
        AllNull.id(),
        AllNonNull.id(),
    ]
});

#[derive(Debug)]
struct NarrowAggregateKernel;

impl DynAggregateKernel for NarrowAggregateKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !SUPPORTED_AGGREGATES.contains(&aggregate_fn.id()) {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<Narrow>() else {
            return Ok(None);
        };

        // These aggregates preserve integer ordering and accumulate sums at i64/u64 width.
        // Cast the partial state too, since it can contain extrema or boundary values.
        let mut accumulator = aggregate_fn.accumulator(array.values().dtype())?;
        accumulator.accumulate(array.values(), ctx)?;
        let partial = accumulator.partial_scalar()?;
        let Some(dtype) = aggregate_fn.state_dtype(batch.dtype()) else {
            return Ok(None);
        };

        Ok(Some(partial.cast(&dtype)?))
    }
}
