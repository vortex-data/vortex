// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::OnceLock;

use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::TryFromBytes;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
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
use crate::layouts::array_tree::ArrayTreesSource;
use crate::layouts::array_tree::reader::ArrayTreeFlatReader;
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(ArrayTreeFlat);

/// Encoding marker for [`ArrayTreeFlatLayout`].
#[derive(Debug)]
pub struct ArrayTreeFlatLayoutEncoding;

/// A flat layout variant that stores its compact encoding tree separately from the data segment.
///
/// At write time, the compact flatbuffer (encoding tree + buffer descriptors, no stats) is
/// stored in this layout and later collected by [`super::ArrayTreeLayout`] into a shared VarBin
/// array.
///
/// At read time, the compact flatbuffer is retrieved from the shared [`ArrayTreesSource`]
/// (injected by the parent [`super::ArrayTreeLayout`]'s reader-construction walk) rather than
/// being parsed from the data segment. This avoids fetching the segment for decode planning
/// and prevents device-to-host copies for device-resident buffers.
#[derive(Clone, Debug)]
pub struct ArrayTreeFlatLayout {
    inner: FlatLayout,
    chunk_idx: usize,
    /// The compact flatbuffer produced at write time. Not persisted — only used to communicate
    /// between the leaf strategy and the collector strategy via the layout tree.
    compact_tree: Option<ByteBuffer>,
    /// Shared source for compact flatbuffers, injected by the parent [`super::ArrayTreeLayout`]
    /// during reader construction.
    source: OnceLock<Arc<ArrayTreesSource>>,
}

impl ArrayTreeFlatLayout {
    /// Creates a new layout at write time with a compact flatbuffer.
    pub fn new(inner: FlatLayout, chunk_idx: usize, compact_tree: ByteBuffer) -> Self {
        Self {
            inner,
            chunk_idx,
            compact_tree: Some(compact_tree),
            source: OnceLock::new(),
        }
    }

    /// Returns the chunk index of this layout in the array trees VarBin.
    pub fn chunk_idx(&self) -> usize {
        self.chunk_idx
    }

    /// Returns the compact flatbuffer, if available (write-time only).
    pub fn compact_tree(&self) -> Option<&ByteBuffer> {
        self.compact_tree.as_ref()
    }

    /// Returns the inner flat layout.
    pub fn inner(&self) -> &FlatLayout {
        &self.inner
    }

    /// Sets the shared array trees source. Called by the parent [`super::ArrayTreeLayout`]
    /// during the reader-construction injection walk.
    pub fn set_source(&self, source: Arc<ArrayTreesSource>) {
        // Ignore if already set (e.g., in tests or double-init scenarios).
        drop(self.source.set(source));
    }

    /// Returns the shared array trees source, if set.
    pub fn source(&self) -> Option<&Arc<ArrayTreesSource>> {
        self.source.get()
    }
}

/// Metadata for [`ArrayTreeFlatLayout`]: stores the chunk index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayTreeFlatMetadata {
    pub chunk_idx: u32,
}

impl SerializeMetadata for ArrayTreeFlatMetadata {
    fn serialize(self) -> Vec<u8> {
        self.chunk_idx.to_le_bytes().to_vec()
    }
}

impl DeserializeMetadata for ArrayTreeFlatMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let chunk_idx = u32::try_from_le_bytes(&metadata[0..4])?;
        Ok(Self { chunk_idx })
    }
}

impl VTable for ArrayTreeFlat {
    type Layout = ArrayTreeFlatLayout;
    type Encoding = ArrayTreeFlatLayoutEncoding;
    type Metadata = ArrayTreeFlatMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_static("vortex.array_tree_flat")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ArrayTreeFlatLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.inner.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout.inner.dtype()
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ArrayTreeFlatMetadata {
            chunk_idx: u32::try_from(layout.chunk_idx).vortex_expect("chunk_idx must fit in u32"),
        }
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        vec![layout.inner.segment_id()]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        0
    }

    fn child(_layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        vortex_bail!("ArrayTreeFlatLayout has no children, got index {}", idx)
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        vortex_panic!("ArrayTreeFlatLayout has no children, got index {}", idx)
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ArrayTreeFlatReader::new(
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
        metadata: &ArrayTreeFlatMetadata,
        segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
        ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        if segment_ids.len() != 1 {
            vortex_bail!("ArrayTreeFlatLayout must have exactly one segment ID");
        }
        Ok(ArrayTreeFlatLayout {
            inner: FlatLayout::new(row_count, dtype.clone(), segment_ids[0], ctx.clone()),
            chunk_idx: metadata.chunk_idx as usize,
            compact_tree: None,
            source: OnceLock::new(),
        })
    }
}
