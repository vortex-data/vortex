// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;

pub(super) fn try_accumulate_grouped(
    states: &mut [u64],
    batch: &ArrayRef,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let validity = batch.validity()?.execute_mask(batch.len(), ctx)?;
    for (&group_id, valid) in group_ids.iter().zip(validity.iter()) {
        if valid {
            states[group_id as usize] += 1;
        }
    }
    Ok(true)
}
