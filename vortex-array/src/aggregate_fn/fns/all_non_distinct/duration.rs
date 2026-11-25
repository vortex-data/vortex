// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::all_non_distinct;
use crate::ExecutionCtx;
use crate::arrays::DurationArray;
use crate::arrays::duration::DurationArraySlotsExt;

pub(super) fn check_duration_identical(
    lhs: &DurationArray,
    rhs: &DurationArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    all_non_distinct(lhs.values(), rhs.values(), ctx)
}
