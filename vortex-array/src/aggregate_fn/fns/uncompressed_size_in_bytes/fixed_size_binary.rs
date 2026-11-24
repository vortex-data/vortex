// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::checked_len_mul;
use super::validity_uncompressed_size_in_bytes;
use crate::ExecutionCtx;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;

/// The size of the values, `byte_width` bytes per row, plus the size of the validity bitmap.
pub(super) fn fixed_size_binary_uncompressed_size_in_bytes(
    array: &FixedSizeBinaryArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let value_size = checked_len_mul(
        array.len(),
        array.byte_width() as usize,
        "fixed-size binary",
    )?;
    let validity_size =
        validity_uncompressed_size_in_bytes(array.validity()?.execute_mask(array.len(), ctx)?)?;

    value_size
        .checked_add(validity_size)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}
