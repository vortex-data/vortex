// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array tree layout: stores compact encoding tree flatbuffers (without stats) as a struct
//! array keyed by segment ID, alongside the data layout. Enables decode planning and
//! sub-segment random access without fetching data segments.

mod flat;
mod reader;
pub mod writer;

use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use vortex_array::EmptyMetadata;
use vortex_array::Executable;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::root;
use vortex_buffer::ByteBuffer;
use vortex_error::SharedVortexResult;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
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
use crate::layouts::array_tree::reader::ArrayTreeFlatReader;
use crate::layouts::array_tree::reader::ArrayTreeReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(ArrayTree);

/// Encoding marker for [`ArrayTreeLayout`].
#[derive(Debug)]
pub struct ArrayTreeLayoutEncoding;

/// Collects compact encoding tree flatbuffers from [`ArrayTreeFlatLayout`] descendants and
/// stores them as a struct array (`{segment_id, compact_tree}`) in an auxiliary child layout.
///
/// # Children
///
/// - Child 0 (`Transparent "data"`): The actual data layout tree (may contain any intermediate
///   layouts like `ChunkedLayout`, `DictLayout`, etc., with [`ArrayTreeFlatLayout`] at the leaves).
/// - Child 1 (`Auxiliary "array_trees"`): A struct array with two fields:
///   - `segment_id: u32` — the segment ID of the data leaf
///   - `compact_tree: bytes` — the compact encoding-tree flatbuffer for that leaf
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
    pub fn array_trees_dtype() -> DType {
        DType::Struct(
            StructFields::new(
                vec![
                    FieldName::from("segment_id"),
                    FieldName::from("compact_tree"),
                ]
                .into(),
                vec![
                    DType::Primitive(PType::U32, Nullability::NonNullable),
                    DType::Binary(Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        )
    }

    /// Build a [`LayoutReaderContext`] that overlays `ctx` with a source-injecting builder
    /// override for this layout's [`ArrayTreeFlat`] descendants.
    ///
    /// The returned context, when used to construct a reader on a descendant layout, will
    /// satisfy `ArrayTreeFlat`'s requirement for an injected [`ArrayTreesSource`]. Used by:
    /// - The normal [`crate::VTable::new_reader`] dispatch on `ArrayTreeLayout` (production path).
    /// - Tools that construct readers at arbitrary points in the layout tree (explorers,
    ///   debuggers) — they should walk from the root to the target node, calling this method
    ///   for each `ArrayTreeLayout` ancestor on the path so the accumulated ctx carries the
    ///   right override when the leaf is finally constructed.
    pub fn derive_reader_ctx(
        &self,
        name: &str,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderContext> {
        // Construct the array_trees auxiliary reader using the unmodified incoming context —
        // the array_trees subtree is a vanilla struct of (u32, bytes) and needs no overrides.
        let array_trees_child = self
            .children
            .child(1, &Self::array_trees_dtype())?;
        let trees_reader = array_trees_child.new_reader_in_ctx(
            Arc::from(format!("{name}/array_trees")),
            segment_source,
            session,
            ctx,
        )?;
        let source = Arc::new(ArrayTreesSource::new(trees_reader, session.clone()));

        Ok(ctx.with_override(
            ArrayTreeFlat::id(&ArrayTreeFlatLayoutEncoding),
            Arc::new(move |layout, name, segs, sess, _ctx| {
                let atf = layout
                    .as_opt::<ArrayTreeFlat>()
                    .vortex_expect("ArrayTreeFlat override applied to wrong layout encoding");
                Ok(Arc::new(ArrayTreeFlatReader::new(
                    atf.clone(),
                    name,
                    segs,
                    sess.clone(),
                    Arc::clone(&source),
                )))
            }),
        ))
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
        let data_reader = data_child.new_reader_in_ctx(
            Arc::clone(&name),
            segment_source,
            session,
            &derived_ctx,
        )?;
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

/// Shared source for compact array tree flatbuffers, keyed by [`SegmentId`].
///
/// Holds a reader for the array_trees child layout. On first lookup, materializes the full
/// struct array and builds a `HashMap<SegmentId, ByteBuffer>` for direct lookup. The map is
/// shared across all leaves of the parent [`ArrayTreeLayout`] via a `OnceLock`-cached future.
pub struct ArrayTreesSource {
    reader: LayoutReaderRef,
    /// Session used to construct execution contexts when canonicalizing the array_trees
    /// struct (its fields may be in compressed encodings depending on how the writer's
    /// `array_trees_strategy` is configured).
    session: VortexSession,
    /// Lazily initialized shared future for the segment-keyed lookup map.
    map: OnceLock<SharedSegmentMapFuture>,
}

type SharedSegmentMapFuture = futures::future::Shared<
    futures::future::BoxFuture<'static, SharedVortexResult<Arc<HashMap<SegmentId, ByteBuffer>>>>,
>;

/// Future returned by [`ArrayTreesSource::get_for_segment`].
pub type SharedSegmentBufferFuture =
    futures::future::Shared<futures::future::BoxFuture<'static, SharedVortexResult<ByteBuffer>>>;

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

    /// Returns a future that resolves to the compact-tree byte buffer for the given data-leaf
    /// segment ID.
    ///
    /// The first call triggers materialization of the entire struct array and the
    /// segment-id-keyed lookup map; subsequent calls reuse the cached map.
    pub fn get_for_segment(&self, segment_id: SegmentId) -> SharedSegmentBufferFuture {
        let map_fut = self.map_future();
        async move {
            let map = map_fut.await?;
            map.get(&segment_id).cloned().ok_or_else(|| {
                Arc::new(vortex_err!(
                    "no compact array tree found for segment id {}",
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

/// Decode the array_trees struct array into a `HashMap<SegmentId, ByteBuffer>`.
///
/// The struct array's columns may be in compressed encodings (bitpacked `segment_id`, dict
/// `compact_tree`, etc.) when read from a file whose array-trees strategy applies compression,
/// so we canonicalize each field via [`Executable::execute`] before downcasting to the
/// concrete typed array.
fn build_segment_map(
    array: vortex_array::ArrayRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<HashMap<SegmentId, ByteBuffer>> {
    let struct_array = StructArray::execute(array, ctx)?;

    let segment_ids_field = struct_array
        .unmasked_field_by_name_opt("segment_id")
        .ok_or_else(|| vortex_err!("array_trees missing 'segment_id' field"))?
        .clone();
    let trees_field = struct_array
        .unmasked_field_by_name_opt("compact_tree")
        .ok_or_else(|| vortex_err!("array_trees missing 'compact_tree' field"))?
        .clone();

    let segment_ids = PrimitiveArray::execute(segment_ids_field, ctx)?;
    let segment_ids = segment_ids.as_slice::<u32>();

    let trees = VarBinViewArray::execute(trees_field, ctx)?;

    let mut map = HashMap::with_capacity(segment_ids.len());
    for (idx, &seg) in segment_ids.iter().enumerate() {
        map.insert(SegmentId::from(seg), trees.bytes_at(idx));
    }
    Ok(map)
}
