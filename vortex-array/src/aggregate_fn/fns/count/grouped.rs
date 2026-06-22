// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Count;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelResult;
use crate::arrays::PrimitiveArray;

#[derive(Debug)]
pub(crate) struct CountGroupedKernel;

impl DynGroupedAggregateKernel for CountGroupedKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<GroupedAggregateKernelResult>> {
        if aggregate_fn.as_opt::<Count>().is_none() {
            return Ok(None);
        }

        let partials = accumulate_grouped(batch, group_ids, ctx)?;
        Ok(Some(GroupedAggregateKernelResult::dense(
            PrimitiveArray::from_iter(partials).into_array(),
            group_ids.num_groups(),
        )?))
    }
}

fn accumulate_grouped(
    batch: &ArrayRef,
    group_ids: &GroupIds,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<u64>> {
    let ids = group_ids.validated_ids(ctx)?;
    let mut partials = vec![0u64; group_ids.num_groups()];
    let validity = batch.validity()?.execute_mask(batch.len(), ctx)?;
    for (&group_id, valid) in ids.iter().zip(validity.iter()) {
        if valid {
            partials[group_id as usize] += 1;
        }
    }
    Ok(partials)
}
