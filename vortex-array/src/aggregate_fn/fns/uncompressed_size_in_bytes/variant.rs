// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::arrays::VariantArray;
use crate::arrays::variant::VariantArrayExt;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;

pub(super) fn variant_uncompressed_size_in_bytes(
    array: &VariantArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let mut size = if let Some(Precision::Exact(size_scalar)) = array
        .core_storage()
        .statistics()
        .get(Stat::UncompressedSizeInBytes)
    {
        u64::try_from(&size_scalar)
            .map_err(|e| vortex_err!("Failed to convert uncompressed size stat to u64: {e}"))?
    } else {
        array.core_storage().nbytes()
    };

    if !array.shredded_is_derived()
        && let Some(shredded) = array.shredded()
    {
        size = size
            .checked_add(super::uncompressed_size_in_bytes_u64(&shredded, ctx)?)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    }

    Ok(size)
}
