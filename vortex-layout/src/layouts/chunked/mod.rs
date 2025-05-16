mod reader;
pub mod writer;

use std::sync::Arc;

use arcref::ArcRef;
use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, FieldMask, FieldPath};
use vortex_error::VortexResult;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, LayoutVisitor, VTable,
    vtable,
};

vtable!(Chunked);

impl VTable for ChunkedVTable {
    type Layout = ChunkedLayout;
    type Encoding = ChunkedLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        ArcRef::new_ref("vortex.chunked")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        ArcRef::new_ref(ChunkedLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        layout.children.len()
    }

    fn visit_children(
        layout: &Self::Layout,
        _field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn LayoutVisitor,
    ) {
        let mut row_offset = 0;
        for i in 0..layout.children.len() {
            let child = layout.children.child(i, &layout.dtype);
            visitor.visit_child(&format!("[{}]", i), row_offset, &FieldPath::root(), &child);
            row_offset += child.row_count();
        }
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn new_reader(
        layout: &Arc<Self::Layout>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ChunkedReader::new(
            layout.clone(),
            segment_source,
            ctx,
        )))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout> {
        Ok(ChunkedLayout {
            row_count,
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }
}

#[derive(Debug)]
pub struct ChunkedLayoutEncoding;

#[derive(Debug)]
pub struct ChunkedLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl ChunkedLayout {
    pub fn new(row_count: u64, dtype: DType, children: Arc<[LayoutRef]>) -> VortexResult<Self> {
        Ok(Self {
            row_count,
            dtype,
            children,
        })
    }
}
