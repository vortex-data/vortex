mod eval_expr;
mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::data::Layout;
use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::{LayoutReader, LayoutReaderExt};
use crate::segments::SegmentSource;
use crate::vtable::LayoutVTable;
use crate::{CHUNKED_LAYOUT_ID, LayoutId};

#[derive(Default, Debug)]
pub struct ChunkedLayout;

/// In-memory representation of Chunked layout.
impl LayoutVTable for ChunkedLayout {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: Layout,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(ChunkedReader::try_new(layout, segment_source.clone(), ctx.clone())?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let mut offset = row_offset;
        for i in 0..layout.nchildren() {
            let child = layout.child(i, layout.dtype().clone(), format!("[{}]", i))?;
            child.register_splits(field_mask, offset, splits)?;
            offset += child.row_count();
            splits.insert(offset);
        }
        Ok(())
    }
}
