// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::validity_uncompressed_size_in_bytes;
use crate::ExecutionCtx;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::dtype::DecimalType;

pub(super) fn decimal_uncompressed_size_in_bytes(
    array: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let value_size = u64::try_from(array.len())
        .map_err(|e| vortex_err!("Failed to convert decimal array length to u64: {e}"))?
        .checked_mul(
            u64::try_from(
                DecimalType::smallest_decimal_value_type(&array.decimal_dtype()).byte_width(),
            )
            .map_err(|e| vortex_err!("Failed to convert decimal byte width to u64: {e}"))?,
        )
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
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
