// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::serde::ColumnarChunkData;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
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
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(ArrayTreeFlat);

/// Encoding marker for [`ArrayTreeFlatLayout`].
#[derive(Debug)]
pub struct ArrayTreeFlatLayoutEncoding;

/// A flat layout variant that retrieves its compact encoding tree from a sibling layout's
/// payload rather than from the data segment trailer.
///
/// At write time, the compact flatbuffer is produced by the leaf strategy and pushed into a
/// side channel shared with the collector strategy — the layout itself just carries the same
/// state as a vanilla [`FlatLayout`].
///
/// At read time, this layout's reader looks up its compact tree in a shared
/// [`super::ArrayTreesSource`] using its own [`SegmentId`]. Construction requires that an
/// ancestor [`super::ArrayTreeLayout`] has registered a reader-builder override against
/// this encoding's ID — this layout has no useful default reader. Tools that need to
/// construct readers at arbitrary points in the layout tree (explorers, debuggers) should
/// use [`super::ArrayTreeLayout::derive_reader_ctx`] to build a context that registers the
/// override before descending to the leaf.
#[derive(Clone, Debug)]
pub struct ArrayTreeFlatLayout {
    inner: FlatLayout,
    /// Transient write-time state: the leaf strategy attaches its [`ColumnarChunkData`] for
    /// the collector to pluck via [`Self::take_chunk`]. Wrapped in `Mutex<Option<_>>` so the
    /// collector can take ownership cheaply during its post-write walk. Read-path
    /// construction (via the layout's `build` method) leaves this `None`; the field is never
    /// serialized to disk.
    chunk: Arc<Mutex<Option<ColumnarChunkData>>>,
}

impl ArrayTreeFlatLayout {
    /// Creates a new layout from the inner flat layout without any attached chunk data.
    pub fn new(inner: FlatLayout) -> Self {
        Self {
            inner,
            chunk: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates a new layout from the inner flat layout with attached transient
    /// [`ColumnarChunkData`]. Used only by the array-tree writer; the chunk is consumed by
    /// the collector and never serialized.
    pub fn with_chunk(inner: FlatLayout, chunk: ColumnarChunkData) -> Self {
        Self {
            inner,
            chunk: Arc::new(Mutex::new(Some(chunk))),
        }
    }

    /// Returns the inner flat layout.
    pub fn inner(&self) -> &FlatLayout {
        &self.inner
    }

    /// Take ownership of any attached transient chunk data, leaving `None` behind.
    pub fn take_chunk(&self) -> Option<ColumnarChunkData> {
        self.chunk.lock().take()
    }
}

impl VTable for ArrayTreeFlat {
    type Layout = ArrayTreeFlatLayout;
    type Encoding = ArrayTreeFlatLayoutEncoding;
    type Metadata = EmptyMetadata;

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

    fn metadata(_layout: &Self::Layout) -> Self::Metadata {
        EmptyMetadata
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
        _layout: &Self::Layout,
        _name: Arc<str>,
        _segment_source: Arc<dyn SegmentSource>,
        _session: &VortexSession,
        _ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        // ArrayTreeFlatLayout has no useful default reader. It exists to be intercepted by an
        // ancestor ArrayTreeLayout that registers a reader-builder override carrying the
        // shared ArrayTreesSource. If the dispatcher reached this method, no such ancestor
        // was present in the layout tree — see `ArrayTreeLayout::derive_reader_ctx` for the
        // helper tools should call when starting reader construction below the root.
        vortex_bail!(
            "ArrayTreeFlatLayout requires an ancestor ArrayTreeLayout to register a reader \
             builder override; call ArrayTreeLayout::derive_reader_ctx on each ArrayTreeLayout \
             ancestor before constructing a reader for this layout"
        )
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &EmptyMetadata,
        segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
        ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        if segment_ids.len() != 1 {
            vortex_bail!("ArrayTreeFlatLayout must have exactly one segment ID");
        }
        Ok(ArrayTreeFlatLayout::new(FlatLayout::new(
            row_count,
            dtype.clone(),
            segment_ids[0],
            ctx.clone(),
        )))
    }
}
