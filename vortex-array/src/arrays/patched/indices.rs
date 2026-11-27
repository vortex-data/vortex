// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ToPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::Patched;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::search_sorted::decode_sorted_positions;

#[expect(
    clippy::cast_possible_truncation,
    reason = "a position within a 1024-row chunk fits u16"
)]
pub(super) fn local_position(row: u64) -> u16 {
    (row % PATCH_CHUNK_SIZE as u64) as u16
}

pub(super) fn chunk_start(
    offsets: &ArrayRef,
    chunk: usize,
    count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    if chunk == offsets.len() {
        return Ok(count);
    }
    if let Some(p) = offsets.as_opt::<Primitive>()
        && p.buffer_handle().is_on_host()
    {
        return match_each_unsigned_integer_ptype!(p.ptype(), |I| {
            p.as_slice::<I>()[chunk]
                .to_usize()
                .ok_or_else(|| vortex_err!("Chunk offset exceeds usize"))
        });
    }
    usize::try_from(&offsets.execute_scalar(chunk, ctx)?)
}

pub(super) fn lower_bound(
    array: ArrayView<'_, Patched>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    let indices = array.patch_indices();
    if index == array.len() {
        return Ok(indices.len());
    }
    let row = array.offset + index;
    let chunk = row / PATCH_CHUNK_SIZE - array.offset / PATCH_CHUNK_SIZE;
    let offsets = array.patch_chunk_offsets();
    let mut lo = chunk_start(offsets, chunk, indices.len(), ctx)?;
    let mut hi = chunk_start(offsets, chunk + 1, indices.len(), ctx)?;
    let needle = local_position(row as u64);
    if let Some(p) = indices.as_opt::<Primitive>()
        && p.buffer_handle().is_on_host()
    {
        return Ok(lo + p.as_slice::<u16>()[lo..hi].partition_point(|&x| x < needle));
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if u16::try_from(&indices.execute_scalar(mid, ctx)?)? < needle {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// Visit each chunk with its physical row origin, first patch entry and local positions.
pub(super) fn visit_chunks<const VALIDATE: bool>(
    indices: &ArrayRef,
    offsets: &ArrayRef,
    offset: usize,
    len: usize,
    ctx: &mut ExecutionCtx,
    mut visit: impl FnMut(usize, usize, &[u16]),
) -> VortexResult<()> {
    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let local = indices.as_slice::<u16>();
    let starts = decode_sorted_positions(offsets, ctx)?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| vortex_err!("Patch range overflow"))?;
    let first_chunk = offset / PATCH_CHUNK_SIZE;
    vortex_ensure_eq!(starts.len(), end.div_ceil(PATCH_CHUNK_SIZE) - first_chunk);
    vortex_ensure!(starts.first() == Some(&0));
    for (chunk, &start) in starts.iter().enumerate() {
        let stop = starts.get(chunk + 1).copied().unwrap_or(local.len() as u64);
        let entries = local
            .get(usize::try_from(start)?..usize::try_from(stop)?)
            .ok_or_else(|| vortex_err!("Invalid patch chunk range"))?;
        let base = (first_chunk + chunk) * PATCH_CHUNK_SIZE;
        if VALIDATE {
            vortex_ensure!(entries.is_sorted_by(|a, b| a < b));
            if let (Some(&first), Some(&last)) = (entries.first(), entries.last()) {
                vortex_ensure!(usize::from(last) < PATCH_CHUNK_SIZE);
                vortex_ensure!(
                    base + usize::from(first) >= offset && base + usize::from(last) < end
                );
            }
        }
        visit(base, usize::try_from(start)?, entries);
    }
    Ok(())
}
