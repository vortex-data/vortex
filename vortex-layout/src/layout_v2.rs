// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::env;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::LazyLock;

use flatbuffers::Follow;
use flatbuffers::VerifierOptions;
use flatbuffers::root_with_opts;
use once_cell::sync::OnceCell;
use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::layout;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::request::ScanRequest;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;
use vortex_session::registry::Registry;

use crate::LayoutChildType;
use crate::LayoutId;
use crate::layouts::zoned::LegacyStatsMetadata;
use crate::layouts::zoned::ZoneMapSchema;
use crate::layouts::zoned::ZonedMetadata;
use crate::layouts::zoned::aggregate_fns_from_specs;
use crate::layouts::zoned::aggregate_stats_table_dtype;
use crate::layouts::zoned::legacy_stats_table_dtype;
use crate::scan::v2::layouts::chunked as scan_chunked;
use crate::scan::v2::layouts::dict as scan_dict;
use crate::scan::v2::layouts::flat as scan_flat;
use crate::scan::v2::layouts::struct_ as scan_struct;
use crate::scan::v2::layouts::zoned as scan_zoned;
use crate::segments::SegmentId;

/// A reference-counted, type-erased v2 layout.
#[derive(Clone)]
pub struct LayoutRef(Arc<dyn DynLayout>);

/// Reference-counted v2 layout-vtable plugin.
pub type LayoutVTableRef = Arc<dyn LayoutVTablePlugin>;

/// Registry mapping layout IDs to v2 layout-vtable plugins.
pub type LayoutVTableRegistry = Registry<LayoutVTableRef>;

static LAYOUT_VERIFIER: LazyLock<VerifierOptions> = LazyLock::new(|| VerifierOptions {
    max_tables: env::var("VORTEX_MAX_LAYOUT_TABLES")
        .ok()
        .and_then(|lmt| lmt.parse::<usize>().ok())
        .unwrap_or(1000000),
    max_depth: env::var("VORTEX_MAX_LAYOUT_DEPTH")
        .ok()
        .and_then(|lmt| lmt.parse::<usize>().ok())
        .unwrap_or(64),
    max_apparent_size: 1 << 31,
    ignore_missing_null_terminator: false,
});

/// Layout-specific behavior for the v2 layout model.
///
/// Common layout fields live in [`Layout`] and are handled by the erased adapter. The vtable only
/// supplies layout-specific data interpretation, child typing, and runtime scan expansion.
pub trait VTable: 'static + Clone + Send + Sync + Debug {
    /// Layout-specific data. Common fields such as dtype, row count, children, and segments are
    /// stored by the adapter.
    type LayoutData: 'static + Send + Sync + Clone + Debug;

    /// Returns the ID of this layout encoding.
    fn id(&self) -> LayoutId;

    /// Deserialize layout-specific data from serialized metadata.
    ///
    /// Common fields are provided in `args`, but remain owned by [`LayoutParts`]. Implementations
    /// should only return layout-specific data.
    fn deserialize(&self, _args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        vortex_bail!(
            "layout v2 deserialization is not implemented for {}",
            self.id()
        )
    }

    /// Returns the expected dtype of child `idx`.
    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType>;

    /// Returns the relationship between child `idx` and its parent.
    fn child_type(layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType>;

    /// Expand this layout into a physical scan plan.
    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef>;
}

/// Object-safe plugin for deserializing v2 layouts by ID.
pub trait LayoutVTablePlugin: 'static + Send + Sync {
    /// Returns the ID of this layout encoding.
    fn id(&self) -> LayoutId;

    /// Deserialize a type-erased v2 layout.
    fn deserialize(&self, args: LayoutDeserializeArgs<'_>) -> VortexResult<LayoutRef>;
}

impl Debug for dyn LayoutVTablePlugin {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("LayoutVTablePlugin")
            .field(&self.id())
            .finish()
    }
}

impl<V: VTable> LayoutVTablePlugin for V {
    fn id(&self) -> LayoutId {
        VTable::id(self)
    }

    fn deserialize(&self, args: LayoutDeserializeArgs<'_>) -> VortexResult<LayoutRef> {
        Ok(LayoutParts::deserialize(self.clone(), args)?.into_layout())
    }
}

/// Common serialized layout fields made available while deserializing v2 layout data.
pub struct LayoutDeserializeArgs<'a> {
    /// The logical dtype of this layout.
    pub dtype: &'a DType,
    /// The row count of this layout.
    pub row_count: u64,
    /// The layout-specific metadata payload.
    pub metadata: &'a [u8],
    /// Segment IDs referenced directly by this layout.
    pub segment_ids: Vec<SegmentId>,
    /// Lazy child access for this layout.
    pub children: Arc<dyn LayoutChildren>,
    /// Array read context captured from the file footer.
    pub array_ctx: &'a ReadContext,
    /// Session used to deserialize session-registered layout metadata.
    pub session: &'a VortexSession,
}

/// Pieces used to construct a v2 layout.
pub struct LayoutParts<V: VTable> {
    vtable: V,
    dtype: DType,
    row_count: u64,
    segment_ids: Vec<SegmentId>,
    children: Arc<dyn LayoutChildren>,
    data: V::LayoutData,
}

impl<V: VTable> LayoutParts<V> {
    /// Create layout parts from common fields and vtable-specific data.
    pub fn new(
        vtable: V,
        dtype: DType,
        row_count: u64,
        segment_ids: Vec<SegmentId>,
        children: Arc<dyn LayoutChildren>,
        data: V::LayoutData,
    ) -> Self {
        Self {
            vtable,
            dtype,
            row_count,
            segment_ids,
            children,
            data,
        }
    }

    /// Deserialize layout-specific data and hoist common fields into layout parts.
    pub fn deserialize(vtable: V, args: LayoutDeserializeArgs<'_>) -> VortexResult<Self> {
        let data = vtable.deserialize(&args)?;
        Ok(Self {
            vtable,
            dtype: args.dtype.clone(),
            row_count: args.row_count,
            segment_ids: args.segment_ids,
            children: args.children,
            data,
        })
    }

    /// Convert these parts into a typed layout.
    pub fn into_typed(self) -> Layout<V> {
        Layout::from_parts(self)
    }

    /// Erase these parts into a layout reference.
    pub fn into_layout(self) -> LayoutRef {
        self.into_typed().into_layout()
    }
}

/// A typed v2 layout handle.
pub struct Layout<V: VTable> {
    inner: Arc<LayoutInner<V>>,
}

struct LayoutInner<V: VTable> {
    vtable: V,
    dtype: DType,
    row_count: u64,
    segment_ids: Vec<SegmentId>,
    children: Arc<dyn LayoutChildren>,
    data: V::LayoutData,
}

impl<V: VTable> Layout<V> {
    /// Create a typed layout from explicit construction parts.
    pub fn from_parts(parts: LayoutParts<V>) -> Self {
        Self {
            inner: Arc::new(LayoutInner {
                vtable: parts.vtable,
                dtype: parts.dtype,
                row_count: parts.row_count,
                segment_ids: parts.segment_ids,
                children: parts.children,
                data: parts.data,
            }),
        }
    }

    /// Returns this layout's vtable.
    pub fn vtable(&self) -> &V {
        &self.inner.vtable
    }

    /// Returns the layout-specific data.
    pub fn data(&self) -> &V::LayoutData {
        &self.inner.data
    }

    /// Returns this layout's dtype.
    pub fn dtype(&self) -> &DType {
        &self.inner.dtype
    }

    /// Returns this layout's row count.
    pub fn row_count(&self) -> u64 {
        self.inner.row_count
    }

    /// Returns this layout's segment IDs.
    pub fn segment_ids(&self) -> &[SegmentId] {
        &self.inner.segment_ids
    }

    /// Returns this layout's children adapter.
    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.inner.children
    }

    /// Returns the number of children.
    pub fn nchildren(&self) -> usize {
        self.inner.children.nchildren()
    }

    /// Returns child `idx`, materializing it lazily.
    pub fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        let dtype = V::child_dtype(self.clone(), idx)?;
        self.inner.children.child(idx, &dtype)
    }

    /// Returns the row count of child `idx`.
    pub fn child_row_count(&self, idx: usize) -> VortexResult<u64> {
        self.inner.children.child_row_count(idx)
    }

    /// Returns the relationship between child `idx` and this layout.
    pub fn child_type(&self, idx: usize) -> VortexResult<LayoutChildType> {
        V::child_type(self.clone(), idx)
    }

    /// Erase this typed layout into a layout reference.
    pub fn to_layout(&self) -> LayoutRef {
        self.clone().into_layout()
    }

    /// Erase this typed layout into a layout reference.
    pub fn into_layout(self) -> LayoutRef {
        LayoutRef(Arc::new(self))
    }
}

impl<V: VTable> Clone for Layout<V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<V: VTable> Debug for Layout<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layout")
            .field("encoding_id", &self.inner.vtable.id())
            .field("dtype", &self.inner.dtype)
            .field("row_count", &self.inner.row_count)
            .field("segment_ids", &self.inner.segment_ids)
            .field("data", &self.inner.data)
            .finish()
    }
}

impl<V: VTable> Deref for Layout<V> {
    type Target = V::LayoutData;

    fn deref(&self) -> &Self::Target {
        self.data()
    }
}

trait DynLayout: 'static + Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;

    fn dyn_encoding_id(&self) -> LayoutId;

    fn dyn_dtype(&self) -> &DType;

    fn dyn_row_count(&self) -> u64;

    fn dyn_segment_ids(&self) -> &[SegmentId];

    fn dyn_nchildren(&self) -> usize;

    fn dyn_child(&self, idx: usize) -> VortexResult<LayoutRef>;

    fn dyn_child_row_count(&self, idx: usize) -> VortexResult<u64>;

    fn dyn_child_type(&self, idx: usize) -> VortexResult<LayoutChildType>;

    fn dyn_new_scan_plan(
        &self,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef>;
}

impl LayoutRef {
    /// Downcast this layout to a typed v2 layout handle.
    pub fn as_opt<V: VTable>(&self) -> Option<Layout<V>> {
        self.0.as_any().downcast_ref::<Layout<V>>().cloned()
    }

    /// Returns a cloned layout reference.
    pub fn to_layout(&self) -> LayoutRef {
        self.clone()
    }

    /// Returns the layout encoding ID.
    pub fn encoding_id(&self) -> LayoutId {
        self.0.dyn_encoding_id()
    }

    /// Returns this layout's dtype.
    pub fn dtype(&self) -> &DType {
        self.0.dyn_dtype()
    }

    /// Returns this layout's row count.
    pub fn row_count(&self) -> u64 {
        self.0.dyn_row_count()
    }

    /// Returns this layout's segment IDs.
    pub fn segment_ids(&self) -> &[SegmentId] {
        self.0.dyn_segment_ids()
    }

    /// Returns the number of children.
    pub fn nchildren(&self) -> usize {
        self.0.dyn_nchildren()
    }

    /// Returns child `idx`, materializing it lazily.
    pub fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        self.0.dyn_child(idx)
    }

    /// Returns the row count of child `idx`.
    pub fn child_row_count(&self, idx: usize) -> VortexResult<u64> {
        self.0.dyn_child_row_count(idx)
    }

    /// Returns the relationship between child `idx` and this layout.
    pub fn child_type(&self, idx: usize) -> VortexResult<LayoutChildType> {
        self.0.dyn_child_type(idx)
    }

    /// Expand this layout into a physical scan plan.
    pub fn new_scan_plan(
        &self,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        self.0.dyn_new_scan_plan(req, session)
    }

    /// Returns an iterator over child row offsets.
    pub fn child_row_offsets(&self) -> impl Iterator<Item = VortexResult<Option<u64>>> + '_ {
        (0..self.nchildren()).map(|idx| Ok(self.child_type(idx)?.row_offset()))
    }
}

impl Debug for LayoutRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<V: VTable> DynLayout for Layout<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_encoding_id(&self) -> LayoutId {
        self.vtable().id()
    }

    fn dyn_dtype(&self) -> &DType {
        &self.inner.dtype
    }

    fn dyn_row_count(&self) -> u64 {
        self.inner.row_count
    }

    fn dyn_segment_ids(&self) -> &[SegmentId] {
        &self.inner.segment_ids
    }

    fn dyn_nchildren(&self) -> usize {
        self.inner.children.nchildren()
    }

    fn dyn_child(&self, idx: usize) -> VortexResult<LayoutRef> {
        Layout::child(self, idx)
    }

    fn dyn_child_row_count(&self, idx: usize) -> VortexResult<u64> {
        self.inner.children.child_row_count(idx)
    }

    fn dyn_child_type(&self, idx: usize) -> VortexResult<LayoutChildType> {
        V::child_type(self.clone(), idx)
    }

    fn dyn_new_scan_plan(
        &self,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        V::new_scan_plan(self.clone(), req, session)
    }
}

/// Lazily provides v2 layout children.
pub trait LayoutChildren: 'static + Send + Sync {
    /// Returns child `idx`, validating its dtype.
    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef>;

    /// Returns child `idx`'s row count.
    fn child_row_count(&self, idx: usize) -> VortexResult<u64>;

    /// Returns the number of children.
    fn nchildren(&self) -> usize;
}

impl Debug for dyn LayoutChildren {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutChildren")
            .field("nchildren", &self.nchildren())
            .finish()
    }
}

#[derive(Clone)]
struct ViewedLayoutChildren {
    flatbuffer: FlatBuffer,
    flatbuffer_loc: usize,
    array_ctx: ReadContext,
    layout_ctx: ReadContext,
    layouts: LayoutVTableRegistry,
    session: VortexSession,
    cache: Arc<[OnceCell<LayoutRef>]>,
}

impl ViewedLayoutChildren {
    unsafe fn new_unchecked(
        flatbuffer: FlatBuffer,
        flatbuffer_loc: usize,
        array_ctx: ReadContext,
        layout_ctx: ReadContext,
        layouts: LayoutVTableRegistry,
        session: VortexSession,
    ) -> Self {
        // SAFETY: guaranteed by caller.
        let nchildren = unsafe { layout::Layout::follow(flatbuffer.as_ref(), flatbuffer_loc) }
            .children()
            .unwrap_or_default()
            .len();
        let cache = vec![OnceCell::new(); nchildren].into_boxed_slice().into();
        Self {
            flatbuffer,
            flatbuffer_loc,
            array_ctx,
            layout_ctx,
            layouts,
            session,
            cache,
        }
    }

    fn flatbuffer(&self) -> layout::Layout<'_> {
        // SAFETY: flatbuffer_loc is produced from a verified flatbuffer table.
        unsafe { layout::Layout::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }
}

impl LayoutChildren for ViewedLayoutChildren {
    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {idx} of {}", self.cache.len());
        }
        let child = self.cache[idx].get_or_try_init(|| {
            let fb_child = self.flatbuffer().children().unwrap_or_default().get(idx);
            // SAFETY: same verified flatbuffer; fb_child._tab.loc() is a valid table location.
            let children = unsafe {
                ViewedLayoutChildren::new_unchecked(
                    self.flatbuffer.clone(),
                    fb_child._tab.loc(),
                    self.array_ctx.clone(),
                    self.layout_ctx.clone(),
                    self.layouts.clone(),
                    self.session.clone(),
                )
            };
            layout_from_fb_layout(
                fb_child,
                dtype,
                self.layout_ctx.clone(),
                self.array_ctx.clone(),
                self.layouts.clone(),
                &self.session,
                Arc::new(children),
            )
        })?;
        Ok(child.clone())
    }

    fn child_row_count(&self, idx: usize) -> VortexResult<u64> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {idx} of {}", self.cache.len());
        }
        Ok(self
            .flatbuffer()
            .children()
            .unwrap_or_default()
            .get(idx)
            .row_count())
    }

    fn nchildren(&self) -> usize {
        self.cache.len()
    }
}

/// Parse a v2 [`LayoutRef`] from a layout flatbuffer.
pub fn layout_from_flatbuffer(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    array_ctx: &ReadContext,
    layouts: &LayoutVTableRegistry,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    let fb_layout = root_with_opts::<layout::Layout>(&LAYOUT_VERIFIER, &flatbuffer)?;
    // SAFETY: the flatbuffer was verified by root_with_opts.
    let children = unsafe {
        ViewedLayoutChildren::new_unchecked(
            flatbuffer.clone(),
            fb_layout._tab.loc(),
            array_ctx.clone(),
            layout_ctx.clone(),
            layouts.clone(),
            session.clone(),
        )
    };
    layout_from_fb_layout(
        fb_layout,
        dtype,
        layout_ctx.clone(),
        array_ctx.clone(),
        layouts.clone(),
        session,
        Arc::new(children),
    )
}

fn layout_from_fb_layout(
    fb_layout: layout::Layout<'_>,
    dtype: &DType,
    layout_ctx: ReadContext,
    array_ctx: ReadContext,
    layouts: LayoutVTableRegistry,
    session: &VortexSession,
    children: Arc<dyn LayoutChildren>,
) -> VortexResult<LayoutRef> {
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding())
        .ok_or_else(|| vortex_err!("Invalid layout encoding ID: {}", fb_layout.encoding()))?;
    let vtable = layouts
        .find(&encoding_id)
        .ok_or_else(|| vortex_err!("Invalid v2 layout encoding ID: {encoding_id}"))?;
    vtable.deserialize(LayoutDeserializeArgs {
        dtype,
        row_count: fb_layout.row_count(),
        metadata: fb_layout
            .metadata()
            .map(|m| m.bytes())
            .unwrap_or_else(|| &[]),
        segment_ids: fb_layout
            .segments()
            .unwrap_or_default()
            .iter()
            .map(SegmentId::from)
            .collect(),
        children,
        array_ctx: &array_ctx,
        session,
    })
}

fn metadata_bool_field(metadata: &[u8], field_number: u64) -> VortexResult<Option<bool>> {
    Ok(metadata_varint_field(metadata, field_number)?.map(|value| value != 0))
}

fn metadata_varint_field(metadata: &[u8], field_number: u64) -> VortexResult<Option<u64>> {
    let mut offset = 0;
    while offset < metadata.len() {
        let key = read_varint(metadata, &mut offset)?;
        let field = key >> 3;
        let wire_type = key & 0x07;
        if field == field_number {
            if wire_type != 0 {
                vortex_bail!("metadata field {field_number} is not a varint");
            }
            return Ok(Some(read_varint(metadata, &mut offset)?));
        }
        skip_proto_field(metadata, &mut offset, wire_type)?;
    }
    Ok(None)
}

fn metadata_bytes_field(metadata: &[u8], field_number: u64) -> VortexResult<Option<Vec<u8>>> {
    let mut offset = 0;
    while offset < metadata.len() {
        let key = read_varint(metadata, &mut offset)?;
        let field = key >> 3;
        let wire_type = key & 0x07;
        if field == field_number {
            if wire_type != 2 {
                vortex_bail!("metadata field {field_number} is not length-delimited");
            }
            let len = usize::try_from(read_varint(metadata, &mut offset)?)?;
            let end = offset + len;
            if end > metadata.len() {
                vortex_bail!("metadata field extends past end of buffer");
            }
            return Ok(Some(metadata[offset..end].to_vec()));
        }
        skip_proto_field(metadata, &mut offset, wire_type)?;
    }
    Ok(None)
}

fn skip_proto_field(metadata: &[u8], offset: &mut usize, wire_type: u64) -> VortexResult<()> {
    match wire_type {
        0 => {
            read_varint(metadata, offset)?;
        }
        1 => {
            *offset += 8;
        }
        2 => {
            let len = usize::try_from(read_varint(metadata, offset)?)?;
            *offset += len;
        }
        5 => {
            *offset += 4;
        }
        _ => vortex_bail!("unsupported protobuf wire type {wire_type}"),
    }
    if *offset > metadata.len() {
        vortex_bail!("metadata field extends past end of buffer");
    }
    Ok(())
}

fn read_varint(metadata: &[u8], offset: &mut usize) -> VortexResult<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *metadata
            .get(*offset)
            .ok_or_else(|| vortex_err!("truncated protobuf varint"))?;
        *offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    vortex_bail!("protobuf varint exceeds 64 bits")
}

/// V2 flat layout vtable.
#[derive(Clone, Debug)]
pub struct Flat;

/// V2 flat layout data.
#[derive(Clone, Debug)]
pub struct FlatData {
    pub(crate) segment_id: SegmentId,
    pub(crate) array_ctx: ReadContext,
    pub(crate) array_tree: Option<ByteBuffer>,
}

impl FlatData {
    /// Returns the serialized array segment ID.
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    /// Returns the array read context.
    pub fn array_ctx(&self) -> &ReadContext {
        &self.array_ctx
    }

    /// Returns the optional inline array encoding tree.
    pub fn array_tree(&self) -> Option<&ByteBuffer> {
        self.array_tree.as_ref()
    }
}

impl VTable for Flat {
    type LayoutData = FlatData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.flat")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        vortex_ensure!(
            args.segment_ids.len() == 1,
            "Flat layout must have exactly one segment ID"
        );
        Ok(FlatData {
            segment_id: args.segment_ids[0],
            array_ctx: args.array_ctx.clone(),
            array_tree: metadata_bytes_field(args.metadata, 1)?.map(ByteBuffer::from),
        })
    }

    fn child_dtype(_layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        vortex_bail!("Flat layout has no child {idx}")
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        vortex_bail!("Flat layout has no child {idx}")
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_flat::new_scan_plan(layout, req, session)
    }
}

/// V2 chunked layout vtable.
#[derive(Clone, Debug)]
pub struct Chunked;

/// V2 chunked layout data.
#[derive(Clone, Debug)]
pub struct ChunkedData {
    pub(crate) chunk_offsets: Vec<u64>,
}

impl ChunkedData {
    /// Returns the cumulative chunk offsets.
    pub fn chunk_offsets(&self) -> &[u64] {
        &self.chunk_offsets
    }
}

impl VTable for Chunked {
    type LayoutData = ChunkedData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.chunked")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        EmptyMetadata::deserialize(args.metadata)?;
        let mut chunk_offsets = Vec::with_capacity(args.children.nchildren() + 1);
        chunk_offsets.push(0);
        for idx in 0..args.children.nchildren() {
            let next = chunk_offsets[idx] + args.children.child_row_count(idx)?;
            chunk_offsets.push(next);
        }
        vortex_ensure!(
            chunk_offsets.last().copied() == Some(args.row_count),
            "Chunked child row counts do not add up to parent row count"
        );
        Ok(ChunkedData { chunk_offsets })
    }

    fn child_dtype(layout: Layout<Self>, _idx: usize) -> VortexResult<DType> {
        Ok(layout.dtype().clone())
    }

    fn child_type(layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        let offset = *layout
            .data()
            .chunk_offsets
            .get(idx)
            .ok_or_else(|| vortex_err!("Chunked child index out of bounds: {idx}"))?;
        Ok(LayoutChildType::Chunk((idx, offset)))
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_chunked::new_scan_plan(layout, req, session)
    }
}

/// V2 struct layout vtable.
#[derive(Clone, Debug)]
pub struct Struct;

impl VTable for Struct {
    type LayoutData = ();

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.struct")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        EmptyMetadata::deserialize(args.metadata)?;
        Ok(())
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        let schema_index = if layout.dtype().is_nullable() {
            idx.saturating_sub(1)
        } else {
            idx
        };
        if idx == 0 && layout.dtype().is_nullable() {
            Ok(DType::Bool(Nullability::NonNullable))
        } else {
            layout
                .dtype()
                .as_struct_fields_opt()
                .and_then(|fields| fields.field_by_index(schema_index))
                .ok_or_else(|| vortex_err!("Missing struct field {schema_index}"))
        }
    }

    fn child_type(layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        let schema_index = if layout.dtype().is_nullable() {
            idx.saturating_sub(1)
        } else {
            idx
        };
        if idx == 0 && layout.dtype().is_nullable() {
            Ok(LayoutChildType::Auxiliary("validity".into()))
        } else {
            let name = layout
                .dtype()
                .as_struct_fields_opt()
                .and_then(|fields| fields.field_name(schema_index))
                .ok_or_else(|| vortex_err!("Missing struct field {schema_index}"))?;
            Ok(LayoutChildType::Field(name.clone()))
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_struct::new_scan_plan(layout, req, session)
    }
}

/// V2 dictionary layout vtable.
#[derive(Clone, Debug)]
pub struct Dict;

/// V2 dictionary layout data.
#[derive(Clone, Debug)]
pub struct DictData {
    pub(crate) codes_dtype: DType,
    pub(crate) all_values_referenced: bool,
}

impl DictData {
    /// Returns whether all dictionary values are definitely referenced.
    pub fn has_all_values_referenced(&self) -> bool {
        self.all_values_referenced
    }
}

impl VTable for Dict {
    type LayoutData = DictData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.dict")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        let codes_ptype = metadata_varint_field(args.metadata, 1)?
            .ok_or_else(|| vortex_err!("Dict metadata missing codes ptype"))?;
        let codes_ptype = PType::try_from(i32::try_from(codes_ptype)?)?;
        let codes_nullable = metadata_bool_field(args.metadata, 2)?
            .map(Nullability::from)
            .unwrap_or_else(|| args.dtype.nullability());
        Ok(DictData {
            codes_dtype: DType::Primitive(codes_ptype, codes_nullable),
            all_values_referenced: metadata_bool_field(args.metadata, 3)?.unwrap_or(false),
        })
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.data().codes_dtype.clone()),
            _ => vortex_bail!("Dict child index out of bounds: {idx}"),
        }
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        match idx {
            0 => Ok(LayoutChildType::Auxiliary("values".into())),
            1 => Ok(LayoutChildType::Transparent("codes".into())),
            _ => vortex_bail!("Dict child index out of bounds: {idx}"),
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_dict::new_scan_plan(layout, req, session)
    }
}

/// V2 zoned layout vtable.
#[derive(Clone, Debug)]
pub struct Zoned;

/// V2 legacy stats layout vtable.
#[derive(Clone, Debug)]
pub struct LegacyStats;

/// V2 zoned layout data.
#[derive(Clone, Debug)]
pub struct ZonedData {
    pub(crate) zone_len: usize,
    pub(crate) zone_map_schema: ZoneMapSchema,
    pub(crate) aggregate_fns: Arc<[AggregateFnRef]>,
}

impl ZonedData {
    /// Returns the configured zone length.
    pub fn zone_len(&self) -> usize {
        self.zone_len
    }

    /// Returns the aggregate functions stored in the zone table.
    pub fn aggregate_fns(&self) -> &Arc<[AggregateFnRef]> {
        &self.aggregate_fns
    }

    /// Returns the zone-map schema used by the zone table.
    pub(crate) fn zone_map_schema(&self) -> &ZoneMapSchema {
        &self.zone_map_schema
    }

    fn stats_table_dtype(&self, dtype: &DType) -> DType {
        match &self.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => legacy_stats_table_dtype(dtype, stats),
            ZoneMapSchema::AggregateFns(aggregate_fns) => {
                aggregate_stats_table_dtype(dtype, aggregate_fns)
            }
        }
    }
}

impl VTable for Zoned {
    type LayoutData = ZonedData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.zoned")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        let metadata = ZonedMetadata::deserialize(args.metadata)?;
        let aggregate_fns = aggregate_fns_from_specs(&metadata.aggregate_specs, args.session)?;
        Ok(ZonedData {
            zone_len: metadata.zone_len as usize,
            zone_map_schema: ZoneMapSchema::AggregateFns(Arc::clone(&aggregate_fns)),
            aggregate_fns,
        })
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.data().stats_table_dtype(layout.dtype())),
            _ => vortex_bail!("Zoned child index out of bounds: {idx}"),
        }
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        match idx {
            0 => Ok(LayoutChildType::Transparent("data".into())),
            1 => Ok(LayoutChildType::Auxiliary("zones".into())),
            _ => vortex_bail!("Zoned child index out of bounds: {idx}"),
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_zoned::new_scan_plan(layout, req, session)
    }
}

impl VTable for LegacyStats {
    type LayoutData = ZonedData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.stats")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        let metadata = LegacyStatsMetadata::deserialize(args.metadata)?;
        let aggregate_fns = match &metadata.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => stats
                .iter()
                .filter_map(|stat| stat.aggregate_fn())
                .collect::<Vec<_>>()
                .into(),
            ZoneMapSchema::AggregateFns(aggregate_fns) => Arc::clone(aggregate_fns),
        };
        Ok(ZonedData {
            zone_len: metadata.zone_len as usize,
            zone_map_schema: metadata.zone_map_schema,
            aggregate_fns,
        })
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.data().stats_table_dtype(layout.dtype())),
            _ => vortex_bail!("Legacy stats child index out of bounds: {idx}"),
        }
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        match idx {
            0 => Ok(LayoutChildType::Transparent("data".into())),
            1 => Ok(LayoutChildType::Auxiliary("zones".into())),
            _ => vortex_bail!("Legacy stats child index out of bounds: {idx}"),
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_zoned::new_scan_plan(layout, req, session)
    }
}
