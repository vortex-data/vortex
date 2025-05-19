mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, FieldMask};
use vortex_error::VortexResult;

use crate::children::LayoutChildren;
use crate::layouts::chunked::reader::ChunkedReader;
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, VTable, vtable,
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

    fn metadata(_layout: &Self::Layout) -> Self::Metadata {
        EmptyMetadata
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        layout.children.nchildren()
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        layout.children.child(idx, &layout.dtype)
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        LayoutChildType::Chunk((idx, layout.chunk_offsets[idx]))
    }

    fn register_splits(
        layout: &Self::Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let mut offset = row_offset;
        for i in 0..layout.nchildren() {
            let child = layout.child(i)?;
            child.register_splits(field_mask, offset, splits)?;
            offset += child.row_count();
            splits.insert(offset);
        }
        Ok(())
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
        Ok(ChunkedLayout::new(
            row_count,
            dtype.clone(),
            children.to_arc(),
        ))
    }
}

#[derive(Debug)]
pub struct ChunkedLayoutEncoding;

#[derive(Clone, Debug)]
pub struct ChunkedLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
    chunk_offsets: Vec<u64>,
}

impl ChunkedLayout {
    pub fn new(row_count: u64, dtype: DType, children: Arc<dyn LayoutChildren>) -> Self {
        let mut chunk_offsets = Vec::with_capacity(children.nchildren() + 1);

        chunk_offsets.push(0);
        for i in 0..children.nchildren() {
            chunk_offsets.push(chunk_offsets[i] + children.child_row_count(i));
        }
        assert_eq!(
            chunk_offsets[children.nchildren()],
            row_count,
            "Row count mismatch"
        );
        Self {
            row_count,
            dtype,
            children,
            chunk_offsets,
        }
    }
}
