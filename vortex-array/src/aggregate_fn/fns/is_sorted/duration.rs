// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::primitive::check_primitive_sorted;
use crate::ExecutionCtx;
use crate::arrays::DurationArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::duration::DurationArraySlotsExt;

pub(super) fn check_duration_sorted(
    array: &DurationArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    check_primitive_sorted(&values, strict, ctx)
}
