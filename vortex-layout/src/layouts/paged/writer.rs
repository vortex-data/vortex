// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_flatbuffers::WriteFlatBufferExt;
use vortex_session::registry::ReadContext;

use crate::LayoutContext;
use crate::LayoutRef;
use crate::layouts::paged::ChunkBoundaries;
use crate::layouts::paged::PagedLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;

/// Serialize `layout` into its own segment, returning the page that stands in for it.
///
/// The subtree is written as a nested `Layout` flatbuffer, interned into the same encoding
/// dictionary the footer serializes through, so the page indexes into the file's one dictionary
/// rather than carrying a copy. The parent spends one table on the whole subtree, and the page is
/// verified against the table and depth limits on its own.
///
/// `row_offsets` are the subtree's exclusive chunk boundaries, relative to its first row. They are
/// promoted to the page so scan planning, which is synchronous, can proceed without reading it,
/// and are kept symbolic when uniform.
pub async fn write_page(
    layout: &LayoutRef,
    row_offsets: &[u64],
    array_ctx: ReadContext,
    layout_ctx: &LayoutContext,
    segment_sink: &SegmentSinkRef,
    sequence_id: SequenceId,
) -> VortexResult<LayoutRef> {
    let page = layout
        .flatbuffer_writer(layout_ctx)
        .write_flatbuffer_bytes()?;
    let segment_id = segment_sink
        .write(sequence_id, vec![page.into_inner()])
        .await?;

    Ok(PagedLayout::new(
        layout.row_count(),
        layout.dtype().clone(),
        segment_id,
        ReadContext::new(layout_ctx.to_ids()),
        array_ctx,
        ChunkBoundaries::from_offsets(row_offsets, layout.row_count()),
    )
    .into_layout())
}
