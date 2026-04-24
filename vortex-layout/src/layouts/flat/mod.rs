// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::layouts::flat::reader::FlatReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

/// Check if inline array node is enabled.
pub(super) fn flat_layout_inline_array_node() -> bool {
    static FLAT_LAYOUT_INLINE_ARRAY_NODE: LazyLock<bool> =
        LazyLock::new(|| env::var("FLAT_LAYOUT_INLINE_ARRAY_NODE").is_ok_and(|v| v == "1"));
    *FLAT_LAYOUT_INLINE_ARRAY_NODE
}

vtable!(Flat);

impl VTable for Flat {
    type Layout = FlatLayout;
    type Encoding = FlatLayoutEncoding;
    type Metadata = ProstMetadata<FlatLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.flat")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(FlatLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ProstMetadata(FlatLayoutMetadata {
            array_encoding_tree: layout.array_tree.as_ref().map(|bytes| bytes.to_vec()),
        })
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        vec![layout.segment_id]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        0
    }

    fn child(_layout: &Self::Layout, _idx: usize) -> VortexResult<LayoutRef> {
        vortex_bail!("Flat layout has no children");
    }

    fn child_type(_layout: &Self::Layout, _idx: usize) -> LayoutChildType {
        vortex_panic!("Flat layout has no children");
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(FlatReader::new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
        )))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
        ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        if segment_ids.len() != 1 {
            vortex_bail!("Flat layout must have exactly one segment ID");
        }
        Ok(FlatLayout::new_with_metadata(
            row_count,
            dtype.clone(),
            segment_ids[0],
            ctx.clone(),
            metadata
                .array_encoding_tree
                .as_ref()
                .map(|v| ByteBuffer::from(v.clone())),
        ))
    }

    fn with_children(_layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        if !children.is_empty() {
            vortex_bail!("Flat layout has no children, got {}", children.len());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FlatLayoutEncoding;

/// The terminal node of a layout tree. Stores a single chunk of array data as one serialized
/// segment on disk.
#[derive(Clone, Debug)]
pub struct FlatLayout {
    row_count: u64,
    dtype: DType,
    segment_id: SegmentId,
    ctx: ReadContext,
    array_tree: Option<ByteBuffer>,
}

impl FlatLayout {
    pub fn new(row_count: u64, dtype: DType, segment_id: SegmentId, ctx: ReadContext) -> Self {
        Self {
            row_count,
            dtype,
            segment_id,
            ctx,
            array_tree: None,
        }
    }

    pub fn new_with_metadata(
        row_count: u64,
        dtype: DType,
        segment_id: SegmentId,
        ctx: ReadContext,
        metadata: Option<ByteBuffer>,
    ) -> Self {
        Self {
            row_count,
            dtype,
            segment_id,
            ctx,
            array_tree: metadata,
        }
    }

    #[inline]
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    #[inline]
    pub fn array_ctx(&self) -> &ReadContext {
        &self.ctx
    }

    #[inline]
    pub fn array_tree(&self) -> Option<&ByteBuffer> {
        self.array_tree.as_ref()
    }
}

#[derive(prost::Message)]
pub struct FlatLayoutMetadata {
    // We can optionally store the array encoding tree here to avoid needing to fetch the segment
    // to plan array deserialization.
    // This will be a `ArrayNode`.
    #[prost(optional, bytes, tag = "1")]
    pub array_encoding_tree: Option<Vec<u8>>,
}
