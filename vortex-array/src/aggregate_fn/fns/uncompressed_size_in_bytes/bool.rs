// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::packed_bit_buffer_size_in_bytes;
use crate::ExecutionCtx;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::validity_uncompressed_size_in_bytes;
use crate::arrays::BoolArray;

pub(super) fn bool_uncompressed_size_in_bytes(
    array: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let value_size = packed_bit_buffer_size_in_bytes(array.len())?;
    let validity_size = validity_uncompressed_size_in_bytes(
        array
            .as_ref()
            .validity()?
            .execute_mask(array.as_ref().len(), ctx)?,
    )?;

    value_size
        .checked_add(validity_size)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}
