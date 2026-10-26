// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Verification of sparse primitive reads before any data bytes are omitted.

use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::flatbuffers::array as fba;

use crate::plan::physical::pipeline::PrimitiveRead;

/// Returns the value-buffer offset only for a verified, single-buffer primitive array.
pub(super) fn value_offset(
    tail: &[u8],
    segment_len: usize,
    demand: &PrimitiveRead,
) -> Option<usize> {
    let length_offset = tail.len().checked_sub(4)?;
    let metadata_len = u32::from_le_bytes(tail[length_offset..].try_into().ok()?) as usize;
    let metadata_start = length_offset.checked_sub(metadata_len)?;
    let array = fba::root_as_array(&tail[metadata_start..length_offset]).ok()?;
    let root = array.root()?;
    if demand.array_ctx.resolve(root.encoding())? != Primitive.id()
        || root.children().is_some_and(|children| !children.is_empty())
        || root.metadata().is_some_and(|metadata| !metadata.is_empty())
    {
        return None;
    }
    let references = root.buffers()?;
    let buffers = array.buffers()?;
    if references.len() != 1 || references.get(0) != 0 || buffers.len() != 1 {
        return None;
    }
    let buffer = buffers.get(0);
    if buffer.length() as usize != demand.buffer_len
        || buffer.compression() != fba::Compression::None
    {
        return None;
    }
    let start = buffer.padding() as usize;
    let end = start.checked_add(demand.buffer_len)?;
    let metadata_offset = segment_len.checked_sub(metadata_len.checked_add(4)?)?;
    if end > metadata_offset
        || demand
            .ranges
            .iter()
            .any(|range| range.start > range.end || range.end > demand.buffer_len)
    {
        return None;
    }
    Some(start)
}
