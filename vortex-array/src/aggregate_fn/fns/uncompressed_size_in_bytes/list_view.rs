// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::uncompressed_size_in_bytes_u64;
use super::validity_uncompressed_size_in_bytes;
use crate::ExecutionCtx;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;

pub(super) fn list_view_uncompressed_size_in_bytes(
    array: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let mut size = if array.is_empty() {
        0
    } else {
        let rebuilt = array.rebuild(ListViewRebuildMode::MakeExact, ctx)?;
        uncompressed_size_in_bytes_u64(rebuilt.elements(), ctx)?
    };

    let view_buffer_size = u64::try_from(array.len())
        .map_err(|e| vortex_err!("Failed to convert list array length to u64: {e}"))?
        .checked_mul(8)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

    // ListView stores both offsets and sizes as u64 view buffers.
    size = size
        .checked_add(view_buffer_size)
        .and_then(|size| size.checked_add(view_buffer_size))
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    size = size
        .checked_add(validity_uncompressed_size_in_bytes(
            array
                .as_ref()
                .validity()?
                .execute_mask(array.as_ref().len(), ctx)?,
        )?)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

    Ok(size)
}
