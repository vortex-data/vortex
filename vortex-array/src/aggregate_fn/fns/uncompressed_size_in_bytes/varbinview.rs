// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::validity_uncompressed_size_in_bytes;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;

pub(super) fn varbinview_uncompressed_size_in_bytes(
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let mut size = u64::try_from(array.len())
        .map_err(|e| vortex_err!("Failed to convert varbinview array length to u64: {e}"))?
        .checked_mul(
            u64::try_from(size_of::<BinaryView>())
                .map_err(|e| vortex_err!("Failed to convert binary view width to u64: {e}"))?,
        )
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

    for buffer in array.data_buffers().iter() {
        size = size
            .checked_add(
                u64::try_from(buffer.len())
                    .map_err(|e| vortex_err!("Failed to convert data buffer length to u64: {e}"))?,
            )
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    }

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
