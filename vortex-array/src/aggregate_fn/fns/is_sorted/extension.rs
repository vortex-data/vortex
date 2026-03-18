// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;

pub(super) fn check_extension_sorted(
    array: &ExtensionArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if strict {
        super::is_strict_sorted(array.storage_array(), ctx)
    } else {
        super::is_sorted(array.storage_array(), ctx)
    }
}
