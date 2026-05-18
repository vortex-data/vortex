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
use vortex_array::MaskFuture;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBinView;
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
        // Construct the array_trees auxiliary reader using the unmodified incoming context —
        // the array_trees subtree is a vanilla struct of (u32, bytes) and needs no overrides.
        let array_trees_child = Self::child(layout, 1)?;
        let trees_reader = array_trees_child.new_reader_in_ctx(
            Arc::from(format!("{name}/array_trees")),
            Arc::clone(&segment_source),
            session,
            ctx,
        )?;
        let source = Arc::new(ArrayTreesSource::new(trees_reader));

        // Derive a context that intercepts ArrayTreeFlat construction with our source-injecting
        // builder. The data subtree (and any nested layouts within it) sees this context, so
        // any ArrayTreeFlat descendant — no matter how deep — gets the source.
        let derived_ctx = ctx.with_override(
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
        );

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
    /// Creates a new source backed by the given array_trees reader.
    pub fn new(reader: LayoutReaderRef) -> Self {
        Self {
            reader,
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
                    build_segment_map(array).map(Arc::new).map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone()
    }
}

/// Decode the array_trees struct array into a `HashMap<SegmentId, ByteBuffer>`.
fn build_segment_map(
    array: vortex_array::ArrayRef,
) -> VortexResult<HashMap<SegmentId, ByteBuffer>> {
    let struct_array = array
        .try_downcast::<Struct>()
        .map_err(|_| vortex_err!("array_trees is not a Struct array"))?;

    let segment_ids_field = struct_array
        .unmasked_field_by_name_opt("segment_id")
        .ok_or_else(|| vortex_err!("array_trees missing 'segment_id' field"))?;
    let trees_field = struct_array
        .unmasked_field_by_name_opt("compact_tree")
        .ok_or_else(|| vortex_err!("array_trees missing 'compact_tree' field"))?;

    let segment_ids = segment_ids_field
        .clone()
        .try_downcast::<Primitive>()
        .map_err(|_| vortex_err!("array_trees 'segment_id' field is not Primitive"))?;
    let segment_ids = segment_ids.as_slice::<u32>();

    let trees = trees_field
        .clone()
        .try_downcast::<VarBinView>()
        .map_err(|_| vortex_err!("array_trees 'compact_tree' field is not a VarBinView"))?;

    let mut map = HashMap::with_capacity(segment_ids.len());
    for (idx, &seg) in segment_ids.iter().enumerate() {
        map.insert(SegmentId::from(seg), trees.bytes_at(idx));
    }
    Ok(map)
}
