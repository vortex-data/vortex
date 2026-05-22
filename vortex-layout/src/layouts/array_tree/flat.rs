// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::serde::ColumnarArrayTree;
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
use crate::layouts::array_tree::ARRAY_TREES_SOURCE_ID;
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

/// Flat-layout variant that retrieves its encoding tree from a sibling consolidated array
/// rather than from a per-segment trailing flatbuffer.
///
/// At write time, the leaf strategy attaches the chunk's [`ColumnarArrayTree`] as transient
/// state (consumed by the collector strategy in its post-write subtree walk).
///
/// At read time, the reader pulls a shared [`ArrayTreesSource`] from the
/// [`LayoutReaderContext`] and resolves its tree by segment id. The source must be
/// published by an ancestor [`super::ArrayTreeLayout`]; constructing a reader without one
/// fails with a clear error.
#[derive(Clone, Debug)]
pub struct ArrayTreeFlatLayout {
    inner: FlatLayout,
    /// Transient write-time state: the leaf strategy attaches its [`ColumnarArrayTree`] for
    /// the collector to pluck via [`Self::take_tree`]. `Mutex<Option<_>>` so the collector
    /// can take ownership cheaply during its post-write walk. Read-path construction (via
    /// `build`) leaves this `None`; the field is never serialized to disk.
    tree: Arc<Mutex<Option<ColumnarArrayTree>>>,
}

impl ArrayTreeFlatLayout {
    /// Creates a new layout from the inner flat layout without any attached tree.
    pub fn new(inner: FlatLayout) -> Self {
        Self {
            inner,
            tree: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates a new layout with an attached transient [`ColumnarArrayTree`]. Used only by
    /// the array-tree writer; the tree is consumed by the collector and never serialized.
    pub fn with_tree(inner: FlatLayout, tree: ColumnarArrayTree) -> Self {
        Self {
            inner,
            tree: Arc::new(Mutex::new(Some(tree))),
        }
    }

    /// Returns the inner flat layout.
    pub fn inner(&self) -> &FlatLayout {
        &self.inner
    }

    /// Take ownership of any attached transient tree, leaving `None` behind.
    pub fn take_tree(&self) -> Option<ColumnarArrayTree> {
        self.tree.lock().take()
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
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let source = ctx.get::<ArrayTreesSource>(*ARRAY_TREES_SOURCE_ID).ok_or_else(|| {
            vortex_error::vortex_err!(
                "ArrayTreeFlatLayout requires an ancestor ArrayTreeLayout to publish an \
                 ArrayTreesSource into the reader context; call \
                 ArrayTreeLayout::derive_reader_ctx on each ArrayTreeLayout ancestor before \
                 constructing a reader for this layout"
            )
        })?;
        Ok(Arc::new(ArrayTreeFlatReader::new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            source,
        )))
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
