// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::is_constant;
use crate::ExecutionCtx;
use crate::arrays::StructArray;

/// Check if a struct array is constant by checking each field independently.
pub(super) fn check_struct_constant(s: &StructArray, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    for field in s.unmasked_fields().iter() {
        if !is_constant(field, ctx)? {
            return Ok(false);
        }
    }
    Ok(true)
}
