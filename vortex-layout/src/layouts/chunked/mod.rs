mod eval_expr;
mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, FieldMask};
use vortex_error::VortexResult;

use crate::data::LayoutData;
use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::LayoutReader;
use crate::segments::{SegmentId, SegmentSource};
use crate::visitor::ReaderVisitor;
use crate::{CHUNKED_LAYOUT_ID, LayoutId, LayoutRef, ReaderChildren, VTable, vtable};

vtable!(Chunked);

impl VTable for ChunkedVTable {
    type Reader = ChunkedReader;
    type Layout = ChunkedLayout;
    type Metadata = EmptyMetadata;

    fn id(_layout: &Self::Layout) -> LayoutId {
        ArcRef::new_ref("vortex.chunked")
    }

    fn layout(reader: &Self::Reader) -> LayoutRef {}

    fn row_count(reader: &Self::Reader) -> u64 {
        todo!()
    }

    fn dtype(reader: &Self::Reader) -> DType {
        todo!()
    }

    fn visit_children(
        reader: &Self::Reader,
        field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn ReaderVisitor,
    ) {
        todo!()
    }

    fn reader_from_parts(
        layout: &Self::Layout,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn ReaderChildren,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Self::Reader> {
        todo!()
    }
}

#[derive(Debug)]
pub struct ChunkedLayout;

/// In-memory representation of Chunked layout.
impl LayoutVTable for ChunkedLayout {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: LayoutData,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(ChunkedReader::try_new(layout, segment_source.clone(), ctx.clone())?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &LayoutData,
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
