// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;

pub(super) fn primitive_uncompressed_size_in_bytes(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let value_size = u64::try_from(array.len())
        .map_err(|e| vortex_err!("Failed to convert primitive array length to u64: {e}"))?
        .checked_mul(
            u64::try_from(array.ptype().byte_width())
                .map_err(|e| vortex_err!("Failed to convert primitive byte width to u64: {e}"))?,
        )
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    let validity_size = super::validity_uncompressed_size_in_bytes(
        array
            .as_ref()
            .validity()?
            .execute_mask(array.as_ref().len(), ctx)?,
    )?;

    value_size
        .checked_add(validity_size)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}
