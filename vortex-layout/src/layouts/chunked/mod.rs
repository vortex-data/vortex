mod reader;
pub mod writer;

use std::sync::Arc;

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
        LayoutId::new_ref("vortex.chunked")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ChunkedLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        layout.children.nchildren()
    }

    fn visit_children(
        layout: &Self::Layout,
        _field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn LayoutVisitor,
    ) {
        let mut row_offset = 0;
        for i in 0..layout.children.nchildren() {
            let child = layout.children.child(i, &layout.dtype);
            visitor.visit_child(
                &format!("[{}]", i),
                row_offset,
                Some(&FieldPath::root()),
                &child,
            );
            row_offset += child.row_count();
        }
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn new_reader(
        layout: &Self::Layout,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ChunkedReader::new(
            layout.clone(),
            name.clone(),
            segment_source.clone(),
            ctx.clone(),
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

#[derive(Clone)]
pub struct ChunkedLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl ChunkedLayout {
    pub fn new(row_count: u64, dtype: DType, children: Arc<[LayoutRef]>) -> Self {
        Self {
            row_count,
            dtype,
            children: children.to_arc(),
        }
    }
}
