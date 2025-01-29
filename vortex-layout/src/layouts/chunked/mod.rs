mod eval_expr;
mod eval_stats;
mod reader;
pub mod stats_table;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::data::LayoutData;
use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::{LayoutReader, LayoutReaderExt};
use crate::segments::AsyncSegmentReader;
use crate::vtable::{LayoutId, LayoutVTable};
use crate::CHUNKED_LAYOUT_ID;

#[derive(Default, Debug)]
pub struct ChunkedLayout;

/// In-memory representation of Chunked layout.
///
/// First child in the list is the metadata table
/// Subsequent children are consecutive chunks of this layout
impl LayoutVTable for ChunkedLayout {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: LayoutData,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(ChunkedReader::try_new(layout, ctx, segments)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &LayoutData,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let nchunks = layout.nchildren() - (if layout.metadata().is_some() { 1 } else { 0 });
        let mut offset = row_offset;
        for i in 0..nchunks {
            let child = layout.child(i, layout.dtype().clone())?;
            child.register_splits(field_mask, offset, splits)?;
            offset += child.row_count();
            splits.insert(offset);
        }
        Ok(())
    }
}
