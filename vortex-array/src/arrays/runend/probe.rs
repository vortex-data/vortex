// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! RunEnd probes retain their ends and values child probes across lookups.

use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::RunEnd;
use crate::RunEndArrayExt;
use crate::RunEndSlots;

pub(crate) fn scalar_at(
    state: &mut ProbeState<'_, RunEnd>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let array = state.array();
    let logical_index = array
        .offset()
        .checked_add(index)
        .ok_or_else(|| vortex_err!("RunEnd logical index overflow"))?;
    // Search for the first end strictly greater than the logical index. Every comparison
    // uses the same ends probe, so a retained read keeps the child's preparation within and
    // between searches.
    let mut ends = state
        .slot(RunEndSlots::ENDS)?
        .ok_or_else(|| vortex_err!("RunEnd ends slot is missing"))?;
    let mut left = 0;
    let mut right = ends.array().len();
    while left < right {
        let mid = left + (right - left) / 2;
        let end = usize::try_from(&ends.execute_scalar(mid, ctx)?)?;
        if end <= logical_index {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    state
        .slot(RunEndSlots::VALUES)?
        .ok_or_else(|| vortex_err!("RunEnd values slot is missing"))?
        .execute_scalar(left, ctx)
}

#[cfg(test)]
mod tests;
