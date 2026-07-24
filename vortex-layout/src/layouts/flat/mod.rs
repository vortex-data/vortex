// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::layouts::flat::reader::FlatReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Check if inline array node is enabled.
pub(super) fn flat_layout_inline_array_node() -> bool {
    static FLAT_LAYOUT_INLINE_ARRAY_NODE: LazyLock<bool> =
        LazyLock::new(|| env::var("FLAT_LAYOUT_INLINE_ARRAY_NODE").is_ok_and(|v| v == "1"));
    *FLAT_LAYOUT_INLINE_ARRAY_NODE
}

/// Flat layout vtable.
#[derive(Clone, Debug)]
pub struct Flat;

/// Backwards-compatible name for the flat layout plugin.
pub use Flat as FlatLayoutEncoding;

/// Flat-layout-specific data.
#[derive(Clone, Debug)]
pub struct FlatData {
    segment_id: SegmentId,
    ctx: ReadContext,
    array_tree: Option<ByteBuffer>,
}

/// A terminal layout storing one serialized array segment.
pub type FlatLayout = Layout<Flat>;

impl VTable for Flat {
    type LayoutData = FlatData;
    type Metadata = ProstMetadata<FlatLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.flat");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(FlatLayoutMetadata {
            array_encoding_tree: layout.array_tree.as_ref().map(|bytes| bytes.to_vec()),
        })
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &FlatLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        if args.segment_ids.len() != 1 {
            vortex_bail!("Flat layout must have exactly one segment ID");
        }
        if args.children.nchildren() != 0 {
            vortex_bail!("Flat layout must not have children");
        }
        Ok(FlatData {
            segment_id: args.segment_ids[0],
            ctx: args.array_read_ctx.clone(),
            array_tree: metadata
                .array_encoding_tree
                .as_ref()
                .map(|bytes| ByteBuffer::from(bytes.clone())),
        })
    }

    fn child_dtype(_layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        vortex_bail!("Flat layout has no child {idx}")
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        vortex_panic!("Flat layout has no child {idx}")
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        _ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(FlatReader::new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
        )))
    }
}

impl Layout<Flat> {
    /// Construct a flat layout without an inline array encoding tree.
    pub fn new(row_count: u64, dtype: DType, segment_id: SegmentId, ctx: ReadContext) -> Self {
        Self::new_with_metadata(row_count, dtype, segment_id, ctx, None)
    }

    /// Construct a flat layout with optional inline array metadata.
    pub fn new_with_metadata(
        row_count: u64,
        dtype: DType,
        segment_id: SegmentId,
        ctx: ReadContext,
        array_tree: Option<ByteBuffer>,
    ) -> Self {
        LayoutParts::new(
            Flat,
            dtype,
            row_count,
            vec![segment_id],
            OwnedLayoutChildren::layout_children(Vec::new()),
            FlatData {
                segment_id,
                ctx,
                array_tree,
            },
        )
        .into_typed()
    }

    /// Returns the serialized array segment ID.
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    /// Returns the array read context.
    pub fn array_ctx(&self) -> &ReadContext {
        &self.ctx
    }

    /// Returns the optional inline array encoding tree.
    pub fn array_tree(&self) -> Option<&ByteBuffer> {
        self.array_tree.as_ref()
    }
}

#[derive(prost::Message)]
pub struct FlatLayoutMetadata {
    #[prost(optional, bytes, tag = "1")]
    pub array_encoding_tree: Option<Vec<u8>>,
}
