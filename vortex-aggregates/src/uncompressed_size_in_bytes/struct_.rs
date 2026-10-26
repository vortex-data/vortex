// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::UncompressedSizeOpts;
use super::uncompressed_size_in_bytes_u64;
use super::validity_uncompressed_size_in_bytes;

pub(super) fn struct_uncompressed_size_in_bytes(
    array: &StructArray,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let mut size = 0u64;

    for field in array.iter_unmasked_fields() {
        size = size
            .checked_add(uncompressed_size_in_bytes_u64(field, opts, ctx)?)
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
