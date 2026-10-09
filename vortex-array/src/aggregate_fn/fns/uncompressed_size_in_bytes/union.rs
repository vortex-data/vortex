// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::UncompressedSizeOpts;
use super::uncompressed_size_in_bytes_u64;
use crate::ExecutionCtx;
use crate::arrays::UnionArray;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionArraySlotsExt;

pub(super) fn union_uncompressed_size_in_bytes(
    array: &UnionArray,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let mut size = uncompressed_size_in_bytes_u64(array.type_ids(), opts, ctx)?;

    for child in array.iter_children() {
        size = size
            .checked_add(uncompressed_size_in_bytes_u64(child, opts, ctx)?)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    }

    Ok(size)
}
