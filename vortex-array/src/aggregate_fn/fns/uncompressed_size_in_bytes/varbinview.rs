// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::UncompressedSizeOpts;
use super::validity_uncompressed_size_in_bytes;
use crate::ExecutionCtx;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;

pub(super) fn varbinview_uncompressed_size_in_bytes(
    array: &VarBinViewArray,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let views_size = u64::try_from(array.len())
        .map_err(|e| vortex_err!("Failed to convert varbinview array length to u64: {e}"))?
        .checked_mul(
            u64::try_from(size_of::<BinaryView>())
                .map_err(|e| vortex_err!("Failed to convert binary view width to u64: {e}"))?,
        )
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;

    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;

    let data_size = if opts.is_exact() {
        referenced_data_size(array.views(), &validity)
    } else {
        held_data_size(array)?
    };

    views_size
        .checked_add(data_size)
        .and_then(|size| size.checked_add(validity_uncompressed_size_in_bytes(validity).ok()?))
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}

/// Every byte of every data buffer, referenced or not.
fn held_data_size(array: &VarBinViewArray) -> VortexResult<u64> {
    let mut size = 0u64;
    for buffer in array.data_buffers().iter() {
        size = size
            .checked_add(
                u64::try_from(buffer.len())
                    .map_err(|e| vortex_err!("Failed to convert data buffer length to u64: {e}"))?,
            )
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
    }
    Ok(size)
}

/// The bytes the valid, non-inlined views point at: what the data buffers would hold once the
/// array is rebuilt from its rows.
///
/// Views of null rows are skipped, as a builder appends nulls without data. Overlapping views
/// (for example after a `take` that repeats rows) are each counted, matching a rebuild.
fn referenced_data_size(views: &[BinaryView], validity: &Mask) -> u64 {
    let out_of_line = |view: &BinaryView| (!view.is_inlined()).then(|| u64::from(view.len()));
    match validity {
        Mask::AllTrue(_) => views.iter().filter_map(out_of_line).sum(),
        Mask::AllFalse(_) => 0,
        Mask::Values(values) => values
            .indices()
            .iter()
            .filter_map(|&row| out_of_line(&views[row]))
            .sum(),
    }
}
