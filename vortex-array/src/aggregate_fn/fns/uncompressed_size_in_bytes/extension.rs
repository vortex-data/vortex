// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::uncompressed_size_in_bytes_u64;
use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;

pub(super) fn extension_uncompressed_size_in_bytes(
    array: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    uncompressed_size_in_bytes_u64(array.storage_array(), ctx)
}
