// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::reader::ChunkedReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(Chunked);

impl VTable for Chunked {
    type Layout = ChunkedLayout;
    type Encoding = ChunkedLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.chunked")
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
        layout.children.child(idx, Self::dtype(layout))
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        LayoutChildType::Chunk((idx, layout.chunk_offsets[idx]))
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ChunkedReader::new(
            layout.clone(),
            name,
            segment_source,
            session,
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
        _ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        Ok(ChunkedLayout::new(
            row_count,
            dtype.clone(),
            children.to_arc(),
        ))
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        let new_children = OwnedLayoutChildren::layout_children(children);

        // Recalculate chunk offsets based on new children
        let mut chunk_offsets = vec![0; new_children.nchildren() + 1];
        for i in 0..new_children.nchildren() {
            chunk_offsets[i + 1] = chunk_offsets[i] + new_children.child_row_count(i);
        }

        layout.children = new_children;
        layout.chunk_offsets = chunk_offsets;
        Ok(())
    }
}

#[derive(Debug)]
pub struct ChunkedLayoutEncoding;

/// Partitions a column into row-based chunks so that each chunk can be read independently.
///
/// Used to break large columns into smaller pieces for parallel I/O and to limit memory
/// usage when scanning.
#[derive(Clone, Debug)]
pub struct ChunkedLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
    chunk_offsets: Vec<u64>,
}

impl ChunkedLayout {
    pub fn new(row_count: u64, dtype: DType, children: Arc<dyn LayoutChildren>) -> Self {
        let mut chunk_offsets = vec![0; children.nchildren() + 1];
        for i in 0..children.nchildren() {
            chunk_offsets[i + 1] = chunk_offsets[i] + children.child_row_count(i);
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

    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.children
    }
}
