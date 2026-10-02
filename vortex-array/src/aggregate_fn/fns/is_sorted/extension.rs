// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::extension::integer::WideInteger;

pub(super) fn check_extension_sorted(
    array: &ExtensionArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if WideInteger::width(array.dtype()).is_some() {
        return super::integer::check_signed_integer_sorted(array.as_ref(), strict, ctx);
    }
    if strict {
        is_strict_sorted(array.storage_array(), ctx)
    } else {
        is_sorted(array.storage_array(), ctx)
    }
}
