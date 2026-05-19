// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::uncompressed_size_in_bytes_u64;
use crate::ExecutionCtx;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::validity::validity_uncompressed_size_in_bytes;

pub(super) fn fixed_size_list_uncompressed_size_in_bytes(
    array: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let elements_size = uncompressed_size_in_bytes_u64(array.elements(), ctx)?;
    let validity_size = validity_uncompressed_size_in_bytes(
        array
            .as_ref()
            .validity()?
            .execute_mask(array.as_ref().len(), ctx)?,
    )?;

    elements_size
        .checked_add(validity_size)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}
