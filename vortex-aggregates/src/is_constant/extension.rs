// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_error::VortexResult;

use super::is_constant;

/// Check if an extension array is constant by delegating to its storage array.
pub(super) fn check_extension_constant(
    e: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    is_constant(e.storage_array(), ctx)
}
