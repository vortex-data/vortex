// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array tree layout: stores compact encoding tree flatbuffers (without stats) as a VarBin
//! vortex array alongside the data layout, enabling decode planning and sub-segment random
//! access without fetching data segments.

mod flat;
mod reader;
pub mod writer;

use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use vortex_array::EmptyMetadata;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::root;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

pub use self::flat::ArrayTreeFlatLayout;
pub use self::flat::ArrayTreeFlatLayoutEncoding;
use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::layouts::array_tree::flat::ArrayTreeFlat;
use crate::layouts::array_tree::reader::ArrayTreeReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(ArrayTree);

/// Encoding marker for [`ArrayTreeLayout`].
#[derive(Debug)]
pub struct ArrayTreeLayoutEncoding;

/// Collects compact encoding tree flatbuffers from [`ArrayTreeFlatLayout`] descendants and stores
/// them as a VarBin array in an auxiliary child layout.
///
/// # Children
///
/// - Child 0 (`Transparent "data"`): The actual data layout tree (may contain any intermediate
///   layouts like `ChunkedLayout`, `DictLayout`, etc., with [`ArrayTreeFlatLayout`] at the leaves).
/// - Child 1 (`Auxiliary "array_trees"`): A VarBin array of compact `Array` flatbuffers, one per
///   [`ArrayTreeFlatLayout`] leaf in depth-first order.
#[derive(Clone, Debug)]
pub struct ArrayTreeLayout {
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl ArrayTreeLayout {
    /// Creates a new `ArrayTreeLayout` from the data and array_trees children.
    pub fn new(data: LayoutRef, array_trees: LayoutRef) -> Self {
        Self {
            dtype: data.dtype().clone(),
            children: OwnedLayoutChildren::layout_children(vec![data, array_trees]),
        }
    }
}

impl VTable for ArrayTree {
    type Layout = ArrayTreeLayout;
    type Encoding = ArrayTreeLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_static("vortex.array_tree")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ArrayTreeLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.children.child_row_count(0)
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

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match idx {
            0 => layout.children.child(0, &layout.dtype),
            1 => layout
                .children
                .child(1, &DType::Binary(Nullability::NonNullable)),
            _ => vortex_bail!("ArrayTreeLayout has 2 children, got index {}", idx),
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent("data".into()),
            1 => LayoutChildType::Auxiliary("array_trees".into()),
            _ => vortex_panic!("ArrayTreeLayout has 2 children, got index {}", idx),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        // Walk the data child to find all ArrayTreeFlatLayouts and inject the array_trees source.
        let data_child = Self::child(layout, 0)?;
        let array_trees_child = Self::child(layout, 1)?;

        // Create a reader for the array_trees VarBin child so the source can lazily read it.
        let trees_reader = array_trees_child.new_reader(
            Arc::from(format!("{name}/array_trees")),
            Arc::clone(&segment_source),
            session,
        )?;
        let source = Arc::new(ArrayTreesSource::new(trees_reader));

        // Inject the shared source into all ArrayTreeFlatLayout descendants.
        for layout_ref in data_child.depth_first_traversal() {
            let layout_ref = layout_ref?;
            if let Some(atf) = layout_ref.as_opt::<ArrayTreeFlat>() {
                atf.set_source(Arc::clone(&source));
            }
        }

        // Create a transparent reader that delegates to the data child.
        let data_reader = data_child.new_reader(Arc::clone(&name), segment_source, session)?;
        Ok(Arc::new(ArrayTreeReader::new(name, data_reader)))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        _metadata: &EmptyMetadata,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        Ok(ArrayTreeLayout {
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        if children.len() != 2 {
            vortex_bail!(
                "ArrayTreeLayout expects 2 children (data, array_trees), got {}",
                children.len()
            );
        }
        layout.children = OwnedLayoutChildren::layout_children(children);
        Ok(())
    }
}

/// Shared source for compact array tree flatbuffers.
///
/// Holds a reader for the array_trees child layout and provides lazy shared access
/// to the decoded VarBin array. The first reader to need it triggers the read; all
/// subsequent readers reuse the shared result.
pub struct ArrayTreesSource {
    reader: LayoutReaderRef,
    /// Lazily initialized shared future for the full VarBin array.
    array: OnceLock<SharedArrayFuture>,
}

impl std::fmt::Debug for ArrayTreesSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayTreesSource").finish_non_exhaustive()
    }
}

impl ArrayTreesSource {
    /// Creates a new source backed by the given array_trees reader.
    pub fn new(reader: LayoutReaderRef) -> Self {
        Self {
            reader,
            array: OnceLock::new(),
        }
    }

    /// Returns a shared future that resolves to the full VarBin array of compact trees.
    pub fn array_future(&self) -> SharedArrayFuture {
        self.array
            .get_or_init(|| {
                let row_count = self.reader.row_count();
                let reader = Arc::clone(&self.reader);
                async move {
                    reader
                        .projection_evaluation(
                            &(0..row_count),
                            &root(),
                            MaskFuture::new_true(
                                usize::try_from(row_count)
                                    .vortex_expect("row count must fit in usize"),
                            ),
                        )?
                        .await
                        .map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone()
    }
}

use crate::layouts::SharedArrayFuture;
