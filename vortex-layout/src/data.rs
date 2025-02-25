use std::collections::BTreeSet;
use std::ops::Deref;
use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::{FlatBufferBuilder, Follow, Verifiable, Verifier, VerifierOptions, WIPOffset};
use vortex_array::ContextRef;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, layout as fb, layout};

use crate::LayoutId;
use crate::context::LayoutContextRef;
use crate::reader::LayoutReader;
use crate::segments::{AsyncSegmentReader, SegmentId};
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
    ctx: LayoutContextRef,
}

impl ViewedLayout {
    /// Return the flatbuffer layout message.
    fn flatbuffer(&self) -> layout::Layout<'_> {
        unsafe { layout::Layout::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
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
        vtable: LayoutVTableRef,
        dtype: DType,
        flatbuffer: FlatBuffer,
        flatbuffer_loc: usize,
        ctx: LayoutContextRef,
    ) -> VortexResult<Self> {
        // Validate the buffer contains a layout message at the given location.
        let opts = VerifierOptions::default();
        let mut v = Verifier::new(&opts, flatbuffer.as_ref());
        fb::Layout::run_verifier(&mut v, flatbuffer_loc)?;

        // SAFETY: we just verified the buffer contains a valid layout message.
        let fb_layout = unsafe { fb::Layout::follow(flatbuffer.as_ref(), flatbuffer_loc) };
        if fb_layout.encoding() != vtable.id().0 {
            vortex_bail!(
                "Mismatched encoding, flatbuffer contains {}, given {}",
                fb_layout.encoding(),
                vtable.id(),
            );
        }

        Ok(Self(Inner::Viewed(ViewedLayout {
            name,
            vtable,
            dtype,
            flatbuffer,
            flatbuffer_loc,
            ctx,
        })))
    }

    /// Create a new viewed layout from a flatbuffer root message.
    ///
    /// # SAFETY
    ///
    /// Assumes that flatbuffer has been previously validated and has same encoding id as the passed encoding
    pub unsafe fn new_viewed_unchecked(
        name: Arc<str>,
        encoding: LayoutVTableRef,
        dtype: DType,
        flatbuffer: FlatBuffer,
        flatbuffer_loc: usize,
        ctx: LayoutContextRef,
    ) -> Self {
        Self(Inner::Viewed(ViewedLayout {
            name,
            vtable: encoding,
            dtype,
            flatbuffer,
            flatbuffer_loc,
            ctx,
        }))
    }

    /// Returns the human-readable name of the layout.
    pub fn name(&self) -> &str {
        match &self.0 {
            Inner::Owned(owned) => owned.name.as_ref(),
            Inner::Viewed(viewed) => viewed.name.as_ref(),
        }
    }

    /// Returns the [`crate::LayoutVTable`] for this layout.
    pub fn encoding(&self) -> &LayoutVTableRef {
        match &self.0 {
            Inner::Owned(owned) => &owned.vtable,
            Inner::Viewed(viewed) => &viewed.vtable,
        }
    }

    /// Returns the ID of the layout.
    pub fn id(&self) -> LayoutId {
        match &self.0 {
            Inner::Owned(owned) => owned.vtable.id(),
            Inner::Viewed(viewed) => LayoutId(viewed.flatbuffer().encoding()),
        }
    }

    /// Return the row-count of the layout.
    pub fn row_count(&self) -> u64 {
        match &self.0 {
            Inner::Owned(owned) => owned.row_count,
            Inner::Viewed(viewed) => viewed.flatbuffer().row_count(),
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
            Inner::Viewed(viewed) => viewed
                .flatbuffer()
                .children()
                .map_or(0, |children| children.len()),
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
            Inner::Viewed(v) => {
                let fb = v
                    .flatbuffer()
                    .children()
                    .vortex_expect("child bounds already checked")
                    .get(i);
                let encoding = v
                    .ctx
                    .lookup_layout(LayoutId(fb.encoding()))
                    .ok_or_else(|| {
                        vortex_err!("Child layout encoding {} not found", fb.encoding())
                    })?;
                Ok(Self(Inner::Viewed(ViewedLayout {
                    name: format!("{}.{}", v.name, name.as_ref()).into(),
                    vtable: encoding,
                    dtype,
                    flatbuffer: v.flatbuffer.clone(),
                    flatbuffer_loc: fb._tab.loc(),
                    ctx: v.ctx.clone(),
                })))
            }
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
                .flatbuffer()
                .children()
                .vortex_expect("child bounds already checked")
                .get(i)
                .row_count(),
        }
    }

    /// Returns the number of segments in the layout.
    pub fn nsegments(&self) -> usize {
        match &self.0 {
            Inner::Owned(owned) => owned.segments.len(),
            Inner::Viewed(viewed) => viewed
                .flatbuffer()
                .segments()
                .map_or(0, |segments| segments.len()),
        }
    }

    /// Fetch the i'th segment id of the layout.
    pub fn segment_id(&self, i: usize) -> Option<SegmentId> {
        match &self.0 {
            Inner::Owned(owned) => owned.segments.get(i).copied(),
            Inner::Viewed(viewed) => viewed
                .flatbuffer()
                .segments()
                .and_then(|segments| (i < segments.len()).then(|| segments.get(i)))
                .map(SegmentId::from),
        }
    }

    /// Iterate the segment IDs of the layout.
    pub fn segments(&self) -> impl Iterator<Item = SegmentId> + '_ {
        (0..self.nsegments()).map(move |i| self.segment_id(i).vortex_expect("segment bounds"))
    }

    /// Returns the layout metadata
    pub fn metadata(&self) -> Option<Bytes> {
        match &self.0 {
            Inner::Owned(owned) => owned.metadata.clone(),
            Inner::Viewed(viewed) => viewed.flatbuffer().metadata().map(|m| {
                // Return the metadata bytes zero-copy by finding them in the flatbuffer.
                viewed.flatbuffer.as_ref().inner().slice_ref(m.bytes())
            }),
        }
    }

    /// Create a reader for this layout.
    pub fn reader(
        &self,
        segment_reader: Arc<dyn AsyncSegmentReader>,
        ctx: ContextRef,
    ) -> VortexResult<Arc<dyn LayoutReader + 'static>> {
        self.encoding().reader(self.clone(), ctx, segment_reader)
    }

    /// Register splits for this layout.
    pub fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.encoding()
            .register_splits(self, field_mask, row_offset, splits)
    }
}

impl FlatBufferRoot for Layout {}

impl WriteFlatBuffer for Layout {
    type Target<'a> = layout::Layout<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        match &self.0 {
            Inner::Owned(layout) => {
                let metadata = layout.metadata.as_ref().map(|b| fbb.create_vector(b));
                let children = (!layout.children.is_empty()).then(|| {
                    layout
                        .children
                        .iter()
                        .map(|c| c.write_flatbuffer(fbb))
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

                layout::Layout::create(
                    fbb,
                    &layout::LayoutArgs {
                        encoding: layout.vtable.id().0,
                        row_count: layout.row_count,
                        metadata,
                        children,
                        segments,
                    },
                )
            }
            Inner::Viewed(layout) => LayoutFlatBuffer(layout.flatbuffer()).write_flatbuffer(fbb),
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
