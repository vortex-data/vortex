use std::collections::BTreeSet;
use std::ops::Deref;
use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::{FlatBufferBuilder, Follow, Verifiable, Verifier, VerifierOptions, WIPOffset};
use vortex_array::ContextRef;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_flatbuffers::{layout as fb, layout, FlatBufferRoot, WriteFlatBuffer};

use crate::context::LayoutContextRef;
use crate::encoding::{LayoutEncodingRef, LayoutId};
use crate::reader::LayoutReader;
use crate::segments::{AsyncSegmentReader, SegmentId};

/// [`LayoutData`] is the lazy equivalent to [`vortex_array::ArrayData`], providing a hierarchical
/// structure.
#[derive(Debug, Clone)]
pub struct LayoutData(Inner);

#[derive(Debug, Clone)]
enum Inner {
    Owned(OwnedLayoutData),
    Viewed(ViewedLayoutData),
}

/// A layout that is fully deserialized and heap-allocated.
#[derive(Debug, Clone)]
pub struct OwnedLayoutData {
    encoding: LayoutEncodingRef,
    dtype: DType,
    row_count: u64,
    segments: Option<Vec<SegmentId>>,
    children: Option<Vec<LayoutData>>,
    metadata: Option<Bytes>,
}

/// A layout that is lazily deserialized from a flatbuffer message.
#[derive(Debug, Clone)]
struct ViewedLayoutData {
    encoding: LayoutEncodingRef,
    dtype: DType,
    flatbuffer: ByteBuffer,
    flatbuffer_loc: usize,
    ctx: LayoutContextRef,
}

impl ViewedLayoutData {
    /// Return the flatbuffer layout message.
    fn flatbuffer(&self) -> layout::Layout<'_> {
        unsafe { layout::Layout::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }
}

impl LayoutData {
    /// Create a new owned layout.
    pub fn new_owned(
        encoding: LayoutEncodingRef,
        dtype: DType,
        row_count: u64,
        segments: Option<Vec<SegmentId>>,
        children: Option<Vec<LayoutData>>,
        metadata: Option<Bytes>,
    ) -> Self {
        Self(Inner::Owned(OwnedLayoutData {
            encoding,
            dtype,
            row_count,
            segments,
            children,
            metadata,
        }))
    }

    /// Create a new viewed layout from a flatbuffer root message.
    pub fn try_new_viewed(
        encoding: LayoutEncodingRef,
        dtype: DType,
        flatbuffer: ByteBuffer,
        flatbuffer_loc: usize,
        ctx: LayoutContextRef,
    ) -> VortexResult<Self> {
        // Validate the buffer contains a layout message at the given location.
        let opts = VerifierOptions::default();
        let mut v = Verifier::new(&opts, flatbuffer.as_ref());
        fb::Layout::run_verifier(&mut v, flatbuffer_loc)?;

        // SAFETY: we just verified the buffer contains a valid layout message.
        let fb_layout = unsafe { fb::Layout::follow(flatbuffer.as_ref(), flatbuffer_loc) };
        if fb_layout.encoding() != encoding.id().0 {
            vortex_bail!(
                "Mismatched encoding, flatbuffer contains {}, given {}",
                fb_layout.encoding(),
                encoding.id(),
            );
        }

        Ok(Self(Inner::Viewed(ViewedLayoutData {
            encoding,
            dtype,
            flatbuffer,
            flatbuffer_loc,
            ctx,
        })))
    }

    /// Returns the [`crate::LayoutEncoding`] for this layout.
    pub fn encoding(&self) -> LayoutEncodingRef {
        match &self.0 {
            Inner::Owned(owned) => owned.encoding,
            Inner::Viewed(viewed) => viewed.encoding,
        }
    }

    /// Returns the ID of the layout.
    pub fn id(&self) -> LayoutId {
        match &self.0 {
            Inner::Owned(owned) => owned.encoding.id(),
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
            Inner::Owned(owned) => owned.children.as_ref().map_or(0, |children| children.len()),
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
    pub fn child(&self, i: usize, dtype: DType) -> VortexResult<LayoutData> {
        if i >= self.nchildren() {
            vortex_panic!("child index out of bounds");
        }
        match &self.0 {
            Inner::Owned(o) => {
                let child = o
                    .children
                    .as_ref()
                    .vortex_expect("child bounds already checked")[i]
                    .clone();
                if child.dtype() != &dtype {
                    vortex_bail!("child dtype mismatch");
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
                Ok(Self(Inner::Viewed(ViewedLayoutData {
                    encoding,
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
            Inner::Owned(o) => o
                .children
                .as_ref()
                .vortex_expect("child bounds already checked")[i]
                .row_count(),
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
            Inner::Owned(owned) => owned.segments.as_ref().map_or(0, |segments| segments.len()),
            Inner::Viewed(viewed) => viewed
                .flatbuffer()
                .segments()
                .map_or(0, |segments| segments.len()),
        }
    }

    /// Fetch the i'th segment id of the layout.
    pub fn segment_id(&self, i: usize) -> Option<SegmentId> {
        match &self.0 {
            Inner::Owned(owned) => owned
                .segments
                .as_ref()
                .and_then(|msgs| msgs.get(i).copied()),
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
                let bytes = viewed.flatbuffer.clone().into_inner();
                bytes.slice_ref(m.bytes())
            }),
        }
    }

    /// Create a reader for this layout.
    pub fn reader(
        &self,
        segments: Arc<dyn AsyncSegmentReader>,
        ctx: ContextRef,
    ) -> VortexResult<Arc<dyn LayoutReader + 'static>> {
        self.encoding().reader(self.clone(), ctx, segments)
    }

    /// Register splits for this layout.
    pub fn register_splits(&self, row_offset: u64, splits: &mut BTreeSet<u64>) -> VortexResult<()> {
        self.encoding().register_splits(self, row_offset, splits)
    }
}

impl FlatBufferRoot for LayoutData {}

impl WriteFlatBuffer for LayoutData {
    type Target<'a> = layout::Layout<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        match &self.0 {
            Inner::Owned(layout) => {
                let metadata = layout.metadata.as_ref().map(|b| fbb.create_vector(b));
                let children = layout.children.as_ref().map(|children| {
                    children
                        .iter()
                        .map(|c| c.write_flatbuffer(fbb))
                        .collect::<Vec<_>>()
                });
                let children = children.map(|c| fbb.create_vector(&c));
                let segments = layout
                    .segments
                    .as_ref()
                    .map(|m| m.iter().map(|s| s.deref()).copied().collect::<Vec<u32>>());
                let segments = segments.map(|m| fbb.create_vector(&m));

                layout::Layout::create(
                    fbb,
                    &layout::LayoutArgs {
                        encoding: layout.encoding.id().0,
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
