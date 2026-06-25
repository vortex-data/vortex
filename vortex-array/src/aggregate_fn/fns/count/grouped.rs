// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Count;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::kernels::GroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelAdapter;

pub(crate) static COUNT_GROUPED_KERNEL: GroupedAggregateKernelAdapter<Count, CountGroupedKernel> =
    GroupedAggregateKernelAdapter::new(CountGroupedKernel);

#[derive(Debug)]
pub(crate) struct CountGroupedKernel;

impl GroupedAggregateKernel<Count> for CountGroupedKernel {
    fn grouped_accumulate(
        &self,
        _options: &EmptyOptions,
        states: &mut [u64],
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let group_ids = group_ids.validated_ids(ctx)?;
        let validity = batch.validity()?.execute_mask(batch.len(), ctx)?;
        for (&group_id, valid) in group_ids.iter().zip(validity.iter()) {
            if valid {
                states[group_id as usize] += 1;
            }
        }
        Ok(true)
    }
}
