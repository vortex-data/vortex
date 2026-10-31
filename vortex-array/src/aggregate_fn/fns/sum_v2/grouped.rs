// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::SumV2;
use super::grouped_state::SumV2GroupedState;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::GroupRuns;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::sum::SumBatch;
use crate::aggregate_fn::kernels::GroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelAdapter;
use crate::aggregate_fn::kernels::for_each_group_run;
use crate::aggregate_fn::kernels::has_long_group_runs;

pub(crate) static SUM_V2_GROUPED_KERNEL: GroupedAggregateKernelAdapter<SumV2, SumV2GroupedKernel> =
    GroupedAggregateKernelAdapter::new(SumV2GroupedKernel);

pub(crate) static SUM_V2_RUN_GROUPED_KERNEL: GroupedAggregateKernelAdapter<
    SumV2,
    SumV2RunGroupedKernel,
> = GroupedAggregateKernelAdapter::new(SumV2RunGroupedKernel);

/// Grouped [`SumV2`] kernel for canonical values arrays.
///
/// Sums values with the same dense machinery as the [`Sum`](crate::aggregate_fn::fns::sum::Sum)
/// kernel, then clears the empty flag of every group that saw a valid value. NaNs still count as
/// values under `skip_nans`, matching scalar `sum_v2`.
#[derive(Debug)]
pub(crate) struct SumV2GroupedKernel;

impl GroupedAggregateKernel<SumV2> for SumV2GroupedKernel {
    type State = SumV2GroupedState;

    fn grouped_accumulate(
        &self,
        options: &NumericalAggregateOpts,
        state: &mut Self::State,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(batch) = SumBatch::try_new(batch) else {
            return Ok(false);
        };
        let group_ids = group_ids.validated_ids(ctx)?;
        let group_ids = group_ids.as_ref();
        let validity = batch.validity(ctx)?;

        let (sums, empty) = state.parts_mut();
        batch.accumulate(sums, group_ids, &validity, options.skip_nans);
        clear_empty(empty, group_ids, &validity);
        Ok(true)
    }
}

/// Clear the empty flag of every group holding at least one valid value in this batch.
fn clear_empty(empty: &mut [bool], group_ids: &[u32], validity: &Mask) {
    match validity.bit_buffer() {
        AllOr::All if has_long_group_runs(group_ids) => {
            // Clustered ids: one write per run instead of one write per row.
            for_each_group_run(group_ids, |group_id, _, _| {
                empty[group_id as usize] = false;
            });
        }
        AllOr::All => {
            for &group_id in group_ids {
                empty[group_id as usize] = false;
            }
        }
        AllOr::None => {}
        AllOr::Some(valid) => valid.for_each_set_index(|idx| {
            empty[group_ids[idx] as usize] = false;
        }),
    }
}

/// Grouped [`SumV2`] kernel for run-encoded group ids.
///
/// The run counterpart of [`SumV2GroupedKernel`]: it reduces one run of values per group and
/// clears the empty flag of every run that holds a valid value.
#[derive(Debug)]
pub(crate) struct SumV2RunGroupedKernel;

impl GroupedAggregateKernel<SumV2> for SumV2RunGroupedKernel {
    type State = SumV2GroupedState;

    fn grouped_accumulate(
        &self,
        options: &NumericalAggregateOpts,
        state: &mut Self::State,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(batch) = SumBatch::try_new(batch) else {
            return Ok(false);
        };
        let validity = batch.validity(ctx)?;
        let Some(runs) = group_ids.runs(ctx)? else {
            // Registered for the encoding, but these ids are not runs after all.
            let group_ids = group_ids.validated_ids(ctx)?;
            let group_ids = group_ids.as_ref();
            let (sums, empty) = state.parts_mut();
            batch.accumulate(sums, group_ids, &validity, options.skip_nans);
            clear_empty(empty, group_ids, &validity);
            return Ok(true);
        };

        let (sums, empty) = state.parts_mut();
        batch.accumulate_runs(sums, &runs, &validity, options.skip_nans);
        clear_empty_runs(empty, &runs, &validity);
        Ok(true)
    }
}

/// Clear the empty flag of every run holding at least one valid value.
fn clear_empty_runs(empty: &mut [bool], runs: &GroupRuns, validity: &Mask) {
    match validity.bit_buffer() {
        AllOr::All => runs.for_each(|group, _, _| empty[group] = false),
        AllOr::None => {}
        AllOr::Some(valid) => runs.for_each(|group, start, end| {
            if valid.count_range(start, end) != 0 {
                empty[group] = false;
            }
        }),
    }
}
