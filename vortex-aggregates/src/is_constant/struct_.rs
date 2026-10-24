// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_error::VortexResult;

use super::is_constant;

/// Check if a struct array is constant by checking each field independently.
pub(super) fn check_struct_constant(s: &StructArray, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    for field in s.iter_unmasked_fields() {
        if !is_constant(field, ctx)? {
            return Ok(false);
        }
    }
    Ok(true)
}
