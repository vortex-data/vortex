// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array tree layout: stores per-chunk encoding trees as one consolidated columnar struct
//! array (one row per data segment), alongside the data layout itself. The leaves write only
//! data buffers — their encoding-tree metadata (plus per-node stats and buffer descriptors)
//! lives in the auxiliary `array_trees` child, which is BtrBlocks-compressed end-to-end. At
//! read time, the source published by [`ArrayTreeLayout::new_reader`] resolves a segment id
//! to its [`ColumnarArrayTree`] in one lookup, then [`ArrayTreeFlatReader`] pairs it with
//! the fetched data segment for decode.

mod flat;
mod reader;
pub mod writer;

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;

use futures::FutureExt;
use vortex_array::EmptyMetadata;
use vortex_array::Executable;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::root;
use vortex_array::serde::BUFFER_COLUMNS_DTYPE;
use vortex_array::serde::ColumnarArrayTree;
use vortex_array::serde::DEFAULT_STATS;
use vortex_array::serde::StatsColumns;
use vortex_array::serde::nodes_columns_dtype;
use vortex_error::SharedVortexResult;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_map::HashMap;

pub use self::flat::ArrayTreeFlat;
pub use self::flat::ArrayTreeFlatLayout;
pub use self::flat::ArrayTreeFlatLayoutEncoding;
use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::layouts::array_tree::reader::ArrayTreeReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(ArrayTree);

/// Well-known [`LayoutReaderContext`] key under which [`ArrayTreeLayout::derive_reader_ctx`]
/// publishes its [`ArrayTreesSource`].
///
/// Both the publisher (parent [`ArrayTreeLayout`]) and the consumer
/// ([`ArrayTreeFlatLayout`]'s `new_reader`) hardcode this id, so no metadata persistence is
/// needed to bind them. Two stacked `ArrayTreeLayouts` both publish under this id; the
/// inner one overrides the outer in the descendant's view — exactly the "nearest ancestor
/// wins" semantic each `ArrayTreeFlat` leaf wants.
pub static ARRAY_TREES_SOURCE_ID: LazyLock<Id> =
    LazyLock::new(|| Id::new_static("vortex.array_tree.source"));

/// Encoding marker for [`ArrayTreeLayout`].
#[derive(Debug)]
pub struct ArrayTreeLayoutEncoding;

/// Stores per-chunk [`ColumnarArrayTree`]s as a consolidated columnar struct array, sharing
/// schema (and compression) across all chunks.
///
/// # Children
///
/// - Child 0 (`Transparent "data"`): The actual data layout tree, with [`ArrayTreeFlatLayout`]
///   at the leaves.
/// - Child 1 (`Auxiliary "array_trees"`): A struct array (`{segment_id, nodes, buffers}`) with
///   one row per data leaf; see [`Self::array_trees_dtype`] for the schema.
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

    /// Returns the dtype of the auxiliary `array_trees` child.
    ///
    /// The consolidated form is a struct array with one row per data-segment chunk:
    ///
    /// ```text
    /// Struct {
    ///   segment_id: u32,
    ///   nodes: List<NODES_COLUMNS_DTYPE>,    // one element per ArrayNode in the chunk's tree
    ///   buffers: List<BUFFER_COLUMNS_DTYPE>, // one element per data buffer for the chunk
    /// }
    /// ```
    ///
    /// The List<> element types are produced by
    /// [`vortex_array::serde::nodes_columns_dtype`] / [`vortex_array::serde::BUFFER_COLUMNS_DTYPE`].
    /// Slicing one row's nodes (resp. buffers) yields a [`StructArray`] matching the inner
    /// shape of a [`ColumnarArrayTree`].
    ///
    /// The `nodes` schema is parameterized by the stat menu the layout tracks; today this
    /// defaults to [`vortex_array::serde::DEFAULT_STATS`] (the historical 11-stat set), but
    /// future writers can pick a different menu and the schema field names carry that menu
    /// across the wire.
    pub fn array_trees_dtype() -> DType {
        let nn = Nullability::NonNullable;
        DType::Struct(
            StructFields::new(
                vec![
                    FieldName::from("segment_id"),
                    FieldName::from("nodes"),
                    FieldName::from("buffers"),
                ]
                .into(),
                vec![
                    DType::Primitive(PType::U32, nn),
                    DType::List(Arc::new(nodes_columns_dtype(DEFAULT_STATS)), nn),
                    DType::List(Arc::new(BUFFER_COLUMNS_DTYPE.clone()), nn),
                ],
            ),
            nn,
        )
    }

    /// Derive a [`LayoutReaderContext`] that publishes an [`ArrayTreesSource`] backed by this
    /// layout's auxiliary `array_trees` child under [`ARRAY_TREES_SOURCE_ID`]. Descendant
    /// [`ArrayTreeFlatLayout`] readers pull the source by the same id to resolve their
    /// compact trees.
    ///
    /// Used by:
    /// - The normal [`crate::VTable::new_reader`] dispatch on `ArrayTreeLayout` (production path).
    /// - Tools that construct readers at arbitrary points in the layout tree (explorers,
    ///   debuggers): walk from the root to the target node, calling this method for each
    ///   `ArrayTreeLayout` ancestor on the path so the accumulated ctx carries the right
    ///   source when the leaf is finally constructed.
    pub fn derive_reader_ctx(
        &self,
        name: &str,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderContext> {
        let array_trees_child = self.children.child(1, &Self::array_trees_dtype())?;
        let trees_reader = array_trees_child.new_reader(
            Arc::from(format!("{name}/array_trees")),
            segment_source,
            session,
            ctx,
        )?;
        let source = Arc::new(ArrayTreesSource::new(trees_reader, session.clone()));
        Ok(ctx.with(*ARRAY_TREES_SOURCE_ID, source))
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
            1 => layout.children.child(1, &Self::Layout::array_trees_dtype()),
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
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let derived_ctx =
            layout.derive_reader_ctx(&name, Arc::clone(&segment_source), session, ctx)?;
        let data_child = Self::child(layout, 0)?;
        let data_reader =
            data_child.new_reader(Arc::clone(&name), segment_source, session, &derived_ctx)?;
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

/// Shared source for per-segment [`ColumnarArrayTree`]s. Holds a reader for the auxiliary
/// `array_trees` child; on first lookup materializes the consolidated array and builds a
/// `HashMap<SegmentId, Arc<ColumnarArrayTree>>`, then serves all subsequent lookups from
/// the cached map.
///
/// Published by [`ArrayTreeLayout::derive_reader_ctx`] into the [`LayoutReaderContext`]
/// passed to descendants under [`ARRAY_TREES_SOURCE_ID`]; pulled by
/// [`ArrayTreeFlatLayout`]'s reader by the same id.
pub struct ArrayTreesSource {
    reader: LayoutReaderRef,
    /// Session used to create execution contexts when canonicalizing the consolidated array
    /// (its fields may be compressed, depending on the writer's `array_trees_strategy`).
    session: VortexSession,
    /// Lazily initialized shared future for the segment-keyed lookup map.
    map: OnceLock<SharedSegmentMapFuture>,
}

type SharedSegmentMapFuture = futures::future::Shared<
    futures::future::BoxFuture<
        'static,
        SharedVortexResult<Arc<HashMap<SegmentId, Arc<ColumnarArrayTree>>>>,
    >,
>;

/// Future returned by [`ArrayTreesSource::get_for_segment`].
pub type SharedSegmentTreeFuture = futures::future::Shared<
    futures::future::BoxFuture<'static, SharedVortexResult<Arc<ColumnarArrayTree>>>,
>;

impl std::fmt::Debug for ArrayTreesSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayTreesSource").finish_non_exhaustive()
    }
}

impl ArrayTreesSource {
    /// Creates a new source backed by the given array_trees reader and session.
    pub fn new(reader: LayoutReaderRef, session: VortexSession) -> Self {
        Self {
            reader,
            session,
            map: OnceLock::new(),
        }
    }

    /// Future resolving to the per-chunk [`ColumnarArrayTree`] for the given data-leaf segment.
    ///
    /// First call triggers materialization of the entire consolidated struct + lookup map;
    /// subsequent calls reuse the cached map.
    pub fn get_for_segment(&self, segment_id: SegmentId) -> SharedSegmentTreeFuture {
        let map_fut = self.map_future();
        async move {
            let map = map_fut.await?;
            map.get(&segment_id).cloned().ok_or_else(|| {
                Arc::new(vortex_err!(
                    "no columnar array tree found for segment id {}",
                    *segment_id
                ))
            })
        }
        .boxed()
        .shared()
    }

    fn map_future(&self) -> SharedSegmentMapFuture {
        self.map
            .get_or_init(|| {
                let row_count = self.reader.row_count();
                let reader = Arc::clone(&self.reader);
                let session = self.session.clone();
                async move {
                    let array = reader
                        .projection_evaluation(
                            &(0..row_count),
                            &root(),
                            MaskFuture::new_true(
                                usize::try_from(row_count)
                                    .vortex_expect("row count must fit in usize"),
                            ),
                        )
                        .map_err(Arc::new)?
                        .await
                        .map_err(Arc::new)?;
                    let mut ctx = session.create_execution_ctx();
                    build_segment_map(array, &mut ctx)
                        .map(Arc::new)
                        .map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone()
    }
}

/// Decode the consolidated `array_trees` struct array into a per-segment lookup of
/// [`ColumnarArrayTree`].
///
/// Canonicalizes each field once (it may be in a compressed encoding), then slices the
/// per-row List<> ranges into typed columns and feeds them to
/// [`ColumnarArrayTree::try_new`].
fn build_segment_map(
    array: vortex_array::ArrayRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<HashMap<SegmentId, Arc<ColumnarArrayTree>>> {
    let outer = StructArray::execute(array, ctx)?;

    let segment_ids = PrimitiveArray::execute(field_clone(&outer, "segment_id")?, ctx)?;
    let segment_ids = segment_ids.as_slice::<u32>();

    let nodes_view = field_clone(&outer, "nodes")?.execute::<ListViewArray>(ctx)?;
    let nodes_list = list_from_list_view(nodes_view, ctx)?;
    let nodes_inner = StructArray::execute(nodes_list.elements().clone(), ctx)?;
    let nodes_cols = NodesColumns::extract(&nodes_inner, ctx)?;

    let buffers_view = field_clone(&outer, "buffers")?.execute::<ListViewArray>(ctx)?;
    let buffers_list = list_from_list_view(buffers_view, ctx)?;
    let buffers_inner = StructArray::execute(buffers_list.elements().clone(), ctx)?;
    let buffers_cols = BuffersColumns::extract(&buffers_inner, ctx)?;

    let mut map = HashMap::with_capacity(segment_ids.len());
    for (row, &seg) in segment_ids.iter().enumerate() {
        let n_start = nodes_list.offset_at(row)?;
        let n_end = nodes_list.offset_at(row + 1)?;
        let b_start = buffers_list.offset_at(row)?;
        let b_end = buffers_list.offset_at(row + 1)?;

        let tree = nodes_cols.tree_at(&buffers_cols, n_start..n_end, b_start..b_end)?;
        map.insert(SegmentId::from(seg), Arc::new(tree));
    }
    Ok(map)
}

fn field_clone(s: &StructArray, name: &str) -> VortexResult<vortex_array::ArrayRef> {
    Ok(s.unmasked_field_by_name_opt(name)
        .ok_or_else(|| vortex_err!("array_trees struct missing field '{}'", name))?
        .clone())
}

/// Canonical typed handles into the nodes inner struct, plus a `tree_at` row-slice helper.
struct NodesColumns {
    encoding_ids: PrimitiveArray,
    child_counts: PrimitiveArray,
    metadata: VarBinViewArray,
    buffers_per_node: PrimitiveArray,
    subtree_sizes: PrimitiveArray,
    buffer_offsets: PrimitiveArray,
    stats: StructArray,
}

impl NodesColumns {
    fn extract(inner: &StructArray, ctx: &mut vortex_array::ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            encoding_ids: PrimitiveArray::execute(field_clone(inner, "encoding_id")?, ctx)?,
            child_counts: PrimitiveArray::execute(field_clone(inner, "child_count")?, ctx)?,
            metadata: VarBinViewArray::execute(field_clone(inner, "metadata")?, ctx)?,
            buffers_per_node: PrimitiveArray::execute(
                field_clone(inner, "buffers_per_node")?,
                ctx,
            )?,
            subtree_sizes: PrimitiveArray::execute(field_clone(inner, "subtree_size")?, ctx)?,
            buffer_offsets: PrimitiveArray::execute(field_clone(inner, "buffer_offset")?, ctx)?,
            stats: StructArray::execute(field_clone(inner, "stats")?, ctx)?,
        })
    }

    fn tree_at(
        &self,
        buffers: &BuffersColumns,
        node_range: std::ops::Range<usize>,
        buffer_range: std::ops::Range<usize>,
    ) -> VortexResult<ColumnarArrayTree> {
        let n_len = node_range.end - node_range.start;
        let b_len = buffer_range.end - buffer_range.start;
        ColumnarArrayTree::try_new(
            slice_primitive(&self.encoding_ids, node_range.clone(), n_len)?,
            slice_primitive(&self.child_counts, node_range.clone(), n_len)?,
            slice_varbinview(&self.metadata, node_range.clone(), n_len)?,
            slice_primitive(&self.buffers_per_node, node_range.clone(), n_len)?,
            slice_primitive(&self.subtree_sizes, node_range.clone(), n_len)?,
            slice_primitive(&self.buffer_offsets, node_range.clone(), n_len)?,
            StatsColumns::new(slice_struct(&self.stats, node_range, n_len)?)?,
            slice_primitive(&buffers.padding, buffer_range.clone(), b_len)?,
            slice_primitive(&buffers.alignment_exponent, buffer_range.clone(), b_len)?,
            slice_primitive(&buffers.length, buffer_range, b_len)?,
        )
    }
}

struct BuffersColumns {
    padding: PrimitiveArray,
    alignment_exponent: PrimitiveArray,
    length: PrimitiveArray,
}

impl BuffersColumns {
    fn extract(inner: &StructArray, ctx: &mut vortex_array::ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            padding: PrimitiveArray::execute(field_clone(inner, "padding")?, ctx)?,
            alignment_exponent: PrimitiveArray::execute(
                field_clone(inner, "alignment_exponent")?,
                ctx,
            )?,
            length: PrimitiveArray::execute(field_clone(inner, "length")?, ctx)?,
        })
    }
}

fn slice_primitive(
    arr: &PrimitiveArray,
    range: std::ops::Range<usize>,
    expected_len: usize,
) -> VortexResult<PrimitiveArray> {
    let sliced = arr.as_ref().slice(range)?;
    debug_assert_eq!(sliced.len(), expected_len);
    sliced
        .try_downcast::<vortex_array::arrays::Primitive>()
        .map_err(|_| vortex_err!("sliced array_trees field is not a canonical PrimitiveArray"))
}

fn slice_varbinview(
    arr: &VarBinViewArray,
    range: std::ops::Range<usize>,
    expected_len: usize,
) -> VortexResult<VarBinViewArray> {
    let sliced = arr.as_ref().slice(range)?;
    debug_assert_eq!(sliced.len(), expected_len);
    sliced
        .try_downcast::<VarBinView>()
        .map_err(|_| vortex_err!("sliced array_trees field is not a canonical VarBinViewArray"))
}

fn slice_struct(
    arr: &StructArray,
    range: std::ops::Range<usize>,
    expected_len: usize,
) -> VortexResult<StructArray> {
    let sliced = arr.as_ref().slice(range)?;
    debug_assert_eq!(sliced.len(), expected_len);
    sliced
        .try_downcast::<Struct>()
        .map_err(|_| vortex_err!("sliced array_trees stats field is not a canonical StructArray"))
}
