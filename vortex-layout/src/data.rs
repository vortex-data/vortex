use std::collections::BTreeSet;
use std::ops::Deref;
use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::{
    FlatBufferBuilder, Follow, ForwardsUOffset, SIZE_UOFFSET, Verifier, VerifierOptions, WIPOffset,
};
use vortex_array::ArrayContext;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, layout};

use crate::LayoutId;
use crate::context::LayoutContext;
use crate::reader::LayoutReader;
use crate::segments::{SegmentId, SegmentSource};
use crate::vtable::LayoutVTableRef;

/// [`Layout`] is the lazy equivalent to [`vortex_array::ArrayRef`], providing a hierarchical
/// structure.
#[derive(Debug, Clone)]
pub struct Layout(Inner);

#[derive(Debug, Clone)]
enum Inner {
    Owned(OwnedLayout),
    Viewed(ViewedLayout),
}

/// A layout that is fully deserialized and heap-allocated.
#[derive(Debug, Clone)]
pub struct OwnedLayout {
    name: Arc<str>,
    vtable: LayoutVTableRef,
    dtype: DType,
    row_count: u64,
    segments: Vec<SegmentId>,
    children: Vec<Layout>,
    metadata: Option<Bytes>,
}

/// A layout that is lazily deserialized from a flatbuffer message.
#[derive(Debug, Clone)]
struct ViewedLayout {
    name: Arc<str>,
    vtable: LayoutVTableRef,
    dtype: DType,
    flatbuffer: FlatBuffer,
    flatbuffer_loc: usize,
    ctx: LayoutContext,
}

impl ViewedLayout {
    /// Creates a new viewed layout while validating the top-level flatbuffer only.
    #[allow(dead_code)]
    fn try_new(
        name: Arc<str>,
        dtype: DType,
        flatbuffer: FlatBuffer,
        flatbuffer_loc: usize,
        ctx: LayoutContext,
    ) -> VortexResult<Self> {
        // We perform manual flatbuffer verification of the top-level Layout message, deferring
        // any recursive validation until we access a specific child.
        let fb = Self::verify_root_layout_only(unsafe {
            layout::Layout::follow(flatbuffer.as_slice(), flatbuffer_loc)
        })?;

        let vtable = ctx
            .lookup_encoding(fb.encoding())
            .ok_or_else(|| vortex_err!("Child layout encoding {} not found", fb.encoding()))?
            .clone();

        Ok(Self {
            name,
            vtable,
            dtype,
            flatbuffer,
            flatbuffer_loc,
            ctx,
        })
    }

    /// Access the flatbuffer layout message.
    ///
    /// SAFETY: this flatbuffer has only been validated to the current layout. Child layouts
    ///  have not been validated and should be accessed via the `child` method.
    unsafe fn flatbuffer(&self) -> layout::Layout<'_> {
        unsafe { layout::Layout::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }

    /// Return the row count from the flatbuffer.
    fn metadata(&self) -> Option<&[u8]> {
        unsafe { self.flatbuffer() }.metadata().map(|m| m.bytes())
    }

    /// Return the row count from the flatbuffer.
    fn row_count(&self) -> u64 {
        unsafe { self.flatbuffer() }.row_count()
    }

    /// Return the number of children from the flatbuffer.
    fn nchildren(&self) -> usize {
        unsafe { self.flatbuffer() }
            .children()
            .map_or(0, |children| children.len())
    }

    /// Return the segment IDs used by this layout.
    fn segment_ids(&self) -> Option<flatbuffers::Vector<u32>> {
        unsafe { self.flatbuffer() }.segments()
    }

    /// Return the child row count.
    fn child_row_count(&self, i: usize) -> VortexResult<u64> {
        let unverified_fb = unsafe { self.flatbuffer() }
            .children()
            .and_then(|children| (children.len() > i).then(|| children.get(i)))
            .ok_or_else(|| vortex_err!("Child index out of bounds"))?;
        let fb = Self::verify_root_layout_only(unverified_fb)?;
        Ok(fb.row_count())
    }

    /// Return the i'th child layout.
    fn child(&self, i: usize, dtype: DType, name: &str) -> VortexResult<Self> {
        let unverified_fb = unsafe { self.flatbuffer() }
            .children()
            .and_then(|children| (children.len() > i).then(|| children.get(i)))
            .ok_or_else(|| vortex_err!("Child index out of bounds"))?;

        Self::try_new(
            format!("{}.{}", self.name, name).into(),
            dtype,
            self.flatbuffer.clone(),
            unverified_fb._tab.loc(),
            self.ctx.clone(),
        )
    }

    fn verify_root_layout_only(fb: layout::Layout<'_>) -> VortexResult<layout::Layout<'_>> {
        // We perform manual flatbuffer verification of the top-level Layout message, deferring
        // any recursive validation until we access a specific child.
        let opts = VerifierOptions::default();
        let mut v = Verifier::new(&opts, fb._tab.buf());

        let offset = v.get_uoffset(fb._tab.loc())? as usize;
        let next_pos = offset.saturating_add(fb._tab.loc());

        // The internals of [`layout::Layout::run_verifier`].
        let mut table_v = v
            .visit_table(next_pos)?
            .visit_field::<u16>("encoding", layout::Layout::VT_ENCODING, false)?
            .visit_field::<u64>("row_count", layout::Layout::VT_ROW_COUNT, false)?
            .visit_field::<ForwardsUOffset<flatbuffers::Vector<'_, u8>>>(
                "metadata",
                layout::Layout::VT_METADATA,
                false,
            )?
            // .visit_field::<ForwardsUOffset<flatbuffers::Vector<'_, ForwardsUOffset<layout::Layout>>>>("children", layout::Layout::VT_CHILDREN, false)?
            .visit_field::<ForwardsUOffset<flatbuffers::Vector<'_, u32>>>(
                "segments",
                layout::Layout::VT_SEGMENTS,
                false,
            )?;

        // Instead of recursively verifying the children, we manually verify that the vector is
        // within bounds, but do not verify the contents of each element.

        if let Some(field_pos) = table_v.deref(layout::Layout::VT_CHILDREN)? {
            let offset = table_v.verifier().get_uoffset(field_pos)? as usize;
            let next_pos = offset.saturating_add(field_pos);

            // Now we verify the vector itself, based on flatbuffers::verifier::verify_vector_range
            // which sadly isn't public.
            let len = v.get_uoffset(next_pos)? as usize;
            let start = next_pos.saturating_add(SIZE_UOFFSET);
            v.is_aligned::<ForwardsUOffset<layout::Layout>>(start)?;
            let size = len.saturating_mul(size_of::<ForwardsUOffset<layout::Layout>>());
            v.range_in_buffer(start, size)?;
        }

        Ok(fb)
    }
}

impl Layout {
    /// Create a new owned layout.
    pub fn new_owned(
        name: Arc<str>,
        vtable: LayoutVTableRef,
        dtype: DType,
        row_count: u64,
        segments: Vec<SegmentId>,
        children: Vec<Layout>,
        metadata: Option<Bytes>,
    ) -> Self {
        Self(Inner::Owned(OwnedLayout {
            name,
            vtable,
            dtype,
            row_count,
            segments,
            children,
            metadata,
        }))
    }

    /// Create a new viewed layout from a flatbuffer root message.
    pub fn try_new_viewed(
        name: Arc<str>,
        dtype: DType,
        flatbuffer: FlatBuffer,
        flatbuffer_loc: usize,
        ctx: LayoutContext,
    ) -> VortexResult<Self> {
        Ok(Self(Inner::Viewed(ViewedLayout::try_new(
            name,
            dtype,
            flatbuffer,
            flatbuffer_loc,
            ctx,
        )?)))
    }

    /// Returns the human-readable name of the layout.
    pub fn name(&self) -> &Arc<str> {
        match &self.0 {
            Inner::Owned(owned) => &owned.name,
            Inner::Viewed(viewed) => &viewed.name,
        }
    }

    /// Returns the [`crate::LayoutVTable`] for this layout.
    pub fn vtable(&self) -> &LayoutVTableRef {
        match &self.0 {
            Inner::Owned(owned) => &owned.vtable,
            Inner::Viewed(viewed) => &viewed.vtable,
        }
    }

    /// Returns the ID of the layout.
    pub fn id(&self) -> LayoutId {
        self.vtable().id()
    }

    /// Return the row-count of the layout.
    pub fn row_count(&self) -> u64 {
        match &self.0 {
            Inner::Owned(owned) => owned.row_count,
            Inner::Viewed(viewed) => viewed.row_count(),
        }
    }

    /// Return the data type of the layout.
    pub fn dtype(&self) -> &DType {
        match &self.0 {
            Inner::Owned(owned) => &owned.dtype,
            Inner::Viewed(viewed) => &viewed.dtype,
        }
    }

    /// Returns the number of children of the layout.
    pub fn nchildren(&self) -> usize {
        match &self.0 {
            Inner::Owned(owned) => owned.children.len(),
            Inner::Viewed(viewed) => viewed.nchildren(),
        }
    }

    /// Fetch the i'th child layout.
    ///
    /// ## Panics
    ///
    /// Panics if the child index is out of bounds.
    pub fn child(&self, i: usize, dtype: DType, name: impl AsRef<str>) -> VortexResult<Layout> {
        if i >= self.nchildren() {
            vortex_panic!("child index out of bounds");
        }
        match &self.0 {
            Inner::Owned(o) => {
                let child = o.children[i].clone();
                if child.dtype() != &dtype {
                    vortex_bail!(
                        "Child has dtype {}, but was requested with {}",
                        child.dtype(),
                        dtype
                    );
                }
                Ok(child)
            }
            Inner::Viewed(v) => Ok(Self(Inner::Viewed(v.child(i, dtype, name.as_ref())?))),
        }
    }

    /// Fetch the row count of the i'th child layout.
    ///
    /// ## Panics
    ///
    /// Panics if the child index is out of bounds.
    pub fn child_row_count(&self, i: usize) -> u64 {
        if i >= self.nchildren() {
            vortex_panic!("child index out of bounds");
        }
        match &self.0 {
            Inner::Owned(o) => o.children[i].row_count(),
            Inner::Viewed(v) => v
                .child_row_count(i)
                .vortex_expect("Failed to verify layout flatbuffer"),
        }
    }

    /// Returns the number of segments in the layout.
    pub fn nsegments(&self) -> usize {
        match &self.0 {
            Inner::Owned(owned) => owned.segments.len(),
            Inner::Viewed(viewed) => viewed.segment_ids().map_or(0, |segments| segments.len()),
        }
    }

    /// Fetch the i'th segment id of the layout.
    pub fn segment_id(&self, i: usize) -> Option<SegmentId> {
        match &self.0 {
            Inner::Owned(owned) => owned.segments.get(i).copied(),
            Inner::Viewed(viewed) => viewed
                .segment_ids()
                .and_then(|segments| (i < segments.len()).then(|| segments.get(i)))
                .map(SegmentId::from),
        }
    }

    /// Iterate the segment IDs of the layout.
    pub fn segments(&self) -> impl Iterator<Item = SegmentId> + '_ {
        (0..self.nsegments()).map(move |i| self.segment_id(i).vortex_expect("segment bounds"))
    }

    /// Returns the bytes of the metadata stored in the layout's flatbuffer.
    pub fn metadata(&self) -> Option<Bytes> {
        match &self.0 {
            Inner::Owned(owned) => owned.metadata.clone(),
            Inner::Viewed(viewed) => viewed.metadata().map(|m| {
                // Return the metadata bytes zero-copy by finding them in the flatbuffer.
                viewed.flatbuffer.as_ref().inner().slice_ref(m)
            }),
        }
    }

    /// Create a reader for this layout.
    pub fn reader(
        &self,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        self.vtable().reader(self.clone(), segment_source, ctx)
    }

    /// Register splits for this layout.
    pub fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.vtable()
            .register_splits(self, field_mask, row_offset, splits)
    }

    /// Serialize the layout into a [`FlatBufferBuilder`].
    pub fn flatbuffer_writer<'a>(
        &'a self,
        ctx: &'a LayoutContext,
    ) -> impl WriteFlatBuffer<Target<'a> = layout::Layout<'a>> + FlatBufferRoot + 'a {
        LayoutFlatBufferWriter { layout: self, ctx }
    }
}

/// An adapter struct for writing a layout to a FlatBuffer.
struct LayoutFlatBufferWriter<'a> {
    layout: &'a Layout,
    ctx: &'a LayoutContext,
}

impl FlatBufferRoot for LayoutFlatBufferWriter<'_> {}

impl WriteFlatBuffer for LayoutFlatBufferWriter<'_> {
    type Target<'t> = layout::Layout<'t>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        match &self.layout.0 {
            Inner::Owned(layout) => {
                let metadata = layout.metadata.as_ref().map(|b| fbb.create_vector(b));

                let children = (!layout.children.is_empty()).then(|| {
                    layout
                        .children
                        .iter()
                        .map(|c| {
                            LayoutFlatBufferWriter {
                                layout: c,
                                ctx: self.ctx,
                            }
                            .write_flatbuffer(fbb)
                        })
                        .collect::<Vec<_>>()
                });

                let children = children.map(|c| fbb.create_vector(&c));
                let segments = (!layout.segments.is_empty()).then(|| {
                    layout
                        .segments
                        .iter()
                        .map(|s| s.deref())
                        .copied()
                        .collect::<Vec<u32>>()
                });
                let segments = segments.map(|m| fbb.create_vector(&m));

                let encoding_idx = self.ctx.encoding_idx(&layout.vtable);

                layout::Layout::create(
                    fbb,
                    &layout::LayoutArgs {
                        encoding: encoding_idx,
                        row_count: layout.row_count,
                        metadata,
                        children,
                        segments,
                    },
                )
            }
            Inner::Viewed(layout) => {
                // FIXME(ngates): we should verify this?
                LayoutFlatBuffer(unsafe { layout.flatbuffer() }).write_flatbuffer(fbb)
            }
        }
    }
}

struct LayoutFlatBuffer<'l>(layout::Layout<'l>);

impl WriteFlatBuffer for LayoutFlatBuffer<'_> {
    type Target<'a> = layout::Layout<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let metadata = self.0.metadata().map(|m| fbb.create_vector(m.bytes()));
        let children = self.0.children().map(|c| {
            c.iter()
                .map(|child| LayoutFlatBuffer(child).write_flatbuffer(fbb))
                .collect::<Vec<_>>()
        });
        let children = children.map(|c| fbb.create_vector(&c));
        let segments = self
            .0
            .segments()
            .map(|m| fbb.create_vector_from_iter(m.iter()));

        layout::Layout::create(
            fbb,
            &layout::LayoutArgs {
                encoding: self.0.encoding(),
                row_count: self.0.row_count(),
                metadata,
                children,
                segments,
            },
        )
    }
}
