// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::all_non_distinct;
use crate::ExecutionCtx;
use crate::arrays::extension::ExtensionArrayExt;

pub(super) fn check_extension_identical<L, R>(
    lhs: &L,
    rhs: &R,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool>
where
    L: ExtensionArrayExt,
    R: ExtensionArrayExt,
{
    all_non_distinct(lhs.storage_array(), rhs.storage_array(), ctx)
}
