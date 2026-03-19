// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::is_constant;
use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;

/// Check if an extension array is constant by delegating to its storage array.
pub(super) fn check_extension_constant(
    e: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    is_constant(e.storage_array(), ctx)
}
