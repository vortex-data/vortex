// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ArrayContext;
use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

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

static FLAT_LAYOUT_INLINE_ARRAY_NODE: LazyLock<bool> =
    LazyLock::new(|| env::var("FLAT_LAYOUT_INLINE_ARRAY_NODE").is_ok());

vtable!(Flat);

impl VTable for FlatVTable {
    type Layout = FlatLayout;
    type Encoding = FlatLayoutEncoding;
    type Metadata = ProstMetadata<FlatLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.flat")
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
        _session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(FlatReader::new(
            layout.clone(),
            name,
            segment_source,
        )))
    }

    #[cfg(gpu_unstable)]
    fn new_gpu_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        ctx: Arc<cudarc::driver::CudaContext>,
    ) -> VortexResult<crate::gpu::GpuLayoutReaderRef> {
        Ok(Arc::new(crate::gpu::layouts::flat::GpuFlatReader::new(
            layout.clone(),
            name,
            segment_source,
            ctx,
        )))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
        ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        if segment_ids.len() != 1 {
            vortex_bail!("Flat layout must have exactly one segment ID");
        }
        Ok(FlatLayout::new_with_metadata(
            row_count,
            dtype.clone(),
            segment_ids[0],
            ctx,
            metadata
                .array_encoding_tree
                .as_ref()
                .map(|v| ByteBuffer::from(v.clone())),
        ))
    }
}

#[derive(Debug)]
pub struct FlatLayoutEncoding;

#[derive(Clone, Debug)]
pub struct FlatLayout {
    row_count: u64,
    dtype: DType,
    segment_id: SegmentId,
    ctx: ArrayContext,
    array_tree: Option<ByteBuffer>,
}

impl FlatLayout {
    pub fn new(row_count: u64, dtype: DType, segment_id: SegmentId, ctx: ArrayContext) -> Self {
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
        ctx: ArrayContext,
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
    pub fn array_ctx(&self) -> &ArrayContext {
        &self.ctx
    }

    #[inline]
    pub fn array_tree(&self) -> Option<&ByteBuffer> {
        self.array_tree.as_ref()
    }
}

/// Display wrapper for the array tree metadata stored in a FlatLayout.
pub struct ArrayTreeDisplay {
    array_tree: Option<ByteBuffer>,
    segment_id: SegmentId,
    ctx: ArrayContext,
}

impl std::fmt::Display for ArrayTreeDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use flatbuffers::root;
        use vortex_flatbuffers::FlatBuffer;
        use vortex_flatbuffers::array as fba;

        let Some(array_tree) = &self.array_tree else {
            return write!(f, "segment {}", *self.segment_id);
        };

        let fb_buffer = FlatBuffer::align_from(array_tree.clone());
        let Ok(fb_array) = root::<fba::Array>(fb_buffer.as_ref()) else {
            return write!(f, "<invalid flatbuffer>");
        };

        let Some(fb_root) = fb_array.root() else {
            return write!(f, "<missing root node>");
        };

        // Get the encoding name from context
        let encoding_id = fb_root.encoding();
        let encoding_name = self
            .ctx
            .lookup_encoding(encoding_id)
            .map(|v| v.id().to_string())
            .unwrap_or_else(|| format!("encoding#{}", encoding_id));

        write!(f, "ArrayTree({}", encoding_name)?;

        // Show children count
        let nchildren = fb_root.children().map_or(0, |c| c.len());
        if nchildren > 0 {
            write!(f, ", children={}", nchildren)?;
        }

        // Show buffer info
        if let Some(buffers) = fb_array.buffers() {
            write!(f, ", buffers=[")?;
            for (i, buf) in buffers.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}B", buf.length())?;
            }
            write!(f, "]")?;
        }

        write!(f, ")")?;
        Ok(())
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
