use std::marker::PhantomData;
use std::sync::Arc;

use flatbuffers::root;
use futures_util::stream::FuturesUnordered;
use futures_util::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use moka::future::CacheBuilder;
use vortex_array::ContextRef;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::{dtype as fbd, footer as fb, FlatBuffer, ReadFlatBuffer};
use vortex_io::VortexReadAt;
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::SegmentId;
use vortex_layout::{Layout, LayoutContextRef, LayoutId};
use vortex_sampling_compressor::ALL_ENCODINGS_CONTEXT;

use crate::footer::{FileLayout, Postscript, Segment};
use crate::segments::{InMemorySegmentCache, NoOpSegmentCache, SegmentCache};
use crate::{
    GenericVortexFile, InMemoryVortexFile, VortexFile, EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE,
    VERSION,
};

pub trait FileType: Sized {
    type Options: Clone;
    type Read: VortexReadAt;
    type ScanDriver: ScanDriver;

    fn scan_driver(
        read: Self::Read,
        options: Self::Options,
        file_layout: FileLayout,
        segment_cache: Arc<dyn SegmentCache>,
    ) -> Self::ScanDriver;
}

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions<F: FileType> {
    /// The underlying file reader.
    read: F::Read,
    /// File-specific options
    pub(crate) options: F::Options,
    /// The Vortex Array encoding context.
    ctx: ContextRef,
    /// The Vortex Layout encoding context.
    layout_ctx: LayoutContextRef,
    /// An optional, externally provided, file size.
    file_size: Option<u64>,
    /// An optional, externally provided, file layout.
    // TODO(ngates): add an optional DType so we only read the layout segment.
    file_layout: Option<FileLayout>,
    segment_cache: Arc<dyn SegmentCache>,
    initial_read_size: u64,
}

impl<F: FileType> VortexOpenOptions<F> {
    /// Configure a Vortex Array context.
    pub fn with_ctx(mut self, ctx: ContextRef) -> Self {
        self.ctx = ctx;
        self
    }

    /// Configure a layout context.
    pub fn with_layouts(mut self, layout_ctx: LayoutContextRef) -> Self {
        self.layout_ctx = layout_ctx;
        self
    }

    /// Configure a known file layout.
    ///
    /// If this is provided, then the Vortex file can be opened without performing any I/O.
    /// Once open, the [`FileLayout`] can be accessed via [`VortexFile::file_layout`].
    pub fn with_file_layout(mut self, file_layout: FileLayout) -> Self {
        self.file_layout = Some(file_layout);
        self
    }

    /// Configure a known file size.
    ///
    /// This helps to prevent an I/O request to discover the size of the file.
    /// Of course, all bets are off if you pass an incorrect value.
    pub fn with_file_size(mut self, file_size: u64) -> Self {
        self.file_size = Some(file_size);
        self
    }

    /// Configure the initial read size for the Vortex file.
    pub fn with_initial_read_size(mut self, initial_read_size: u64) -> VortexResult<Self> {
        if self.initial_read_size < u16::MAX as u64 {
            vortex_bail!("initial_read_size must be at least u16::MAX");
        }
        self.initial_read_size = initial_read_size;
        Ok(self)
    }

    /// Configure a custom [`SegmentCache`].
    pub fn with_segment_cache(mut self, segment_cache: Arc<dyn SegmentCache>) -> Self {
        self.segment_cache = segment_cache;
        self
    }

    /// Disable segment caching entirely.
    pub fn without_segment_cache(self) -> Self {
        self.with_segment_cache(Arc::new(NoOpSegmentCache))
    }
}

impl VortexOpenOptions<InMemoryVortexFile> {
    /// Open an in-memory file contained in the provided buffer.
    pub fn in_memory<B: Into<ByteBuffer>>(buffer: B) -> Self {
        Self {
            read: buffer.into(),
            options: (),
            ctx: ALL_ENCODINGS_CONTEXT.clone(),
            layout_ctx: Arc::new(Default::default()),
            file_size: None,
            file_layout: None,
            segment_cache: Arc::new(NoOpSegmentCache),
            initial_read_size: 0,
        }
    }
}

impl<R: VortexReadAt> VortexOpenOptions<GenericVortexFile<R>> {
    const INITIAL_READ_SIZE: u64 = 1 << 20; // 1 MB

    pub fn file(read: R) -> Self {
        Self {
            read,
            // TODO(ngates): move this context into the vortex-file crate
            options: Default::default(),
            ctx: ALL_ENCODINGS_CONTEXT.clone(),
            layout_ctx: LayoutContextRef::default(),
            file_size: None,
            file_layout: None,
            segment_cache: Arc::new(InMemorySegmentCache::new(
                // For now, use a fixed 1GB overhead.
                CacheBuilder::new(1 << 30),
            )),
            initial_read_size: Self::INITIAL_READ_SIZE,
        }
    }
}

impl<F: FileType> VortexOpenOptions<F> {
    /// Open the Vortex file using asynchronous IO.
    pub async fn open(mut self) -> VortexResult<VortexFile<F>> {
        // If we need to read the file layout, then do so.
        let file_layout = match self.file_layout.take() {
            None => self.read_file_layout().await?,
            Some(file_layout) => file_layout,
        };

        Ok(VortexFile {
            read: self.read,
            options: self.options,
            ctx: self.ctx.clone(),
            file_layout,
            segment_cache: self.segment_cache,
            _marker: PhantomData,
        })
    }

    /// Read the [`FileLayout`] from the file.
    async fn read_file_layout(&self) -> VortexResult<FileLayout> {
        // Fetch the file size and perform the initial read.
        let file_size = match self.file_size {
            None => self.read.size().await?,
            Some(file_size) => file_size,
        };
        let initial_read_size = self
            .initial_read_size
            // Make sure we read enough to cover the postscript
            .max(MAX_FOOTER_SIZE as u64 + EOF_SIZE as u64)
            .min(file_size);
        let initial_offset = file_size - initial_read_size;
        let initial_read: ByteBuffer = self
            .read
            .read_byte_range(initial_offset, initial_read_size)
            .await?
            .into();

        // We know the initial read _must_ contain at least the Postscript.
        let postscript = self.parse_postscript(&initial_read)?;

        // Check if we need to read more bytes.
        let (initial_offset, initial_read) = if (postscript.dtype.offset < initial_offset)
            || (postscript.file_layout.offset < initial_offset)
        {
            // NOTE(ngates): for now, we assume the dtype and layout segments are adjacent.
            let offset = postscript.dtype.offset.min(postscript.file_layout.offset);
            let mut new_initial_read =
                ByteBufferMut::with_capacity(usize::try_from(file_size - offset)?);
            new_initial_read.extend_from_slice(
                &self
                    .read
                    .read_byte_range(offset, initial_offset - offset)
                    .await?,
            );
            new_initial_read.extend_from_slice(&initial_read);
            (offset, new_initial_read.freeze())
        } else {
            (initial_offset, initial_read)
        };

        // Now we try to read the DType and Layout segments.
        let dtype = self
            .parse_dtype(initial_offset, &initial_read, postscript.dtype)
            .vortex_expect("Failed to parse dtype");
        let file_layout = self.file_layout.clone().unwrap_or_else(|| {
            self.parse_file_layout(
                initial_offset,
                &initial_read,
                postscript.file_layout,
                dtype.clone(),
            )
            .vortex_expect("Failed to parse file layout")
        });

        // If the initial read happened to cover any segments, then we can populate the
        // segment cache
        self.populate_segments(initial_offset, &initial_read, &file_layout)
            .await?;

        Ok(file_layout)
    }

    /// Parse the postscript from the initial read.
    fn parse_postscript(&self, initial_read: &[u8]) -> VortexResult<Postscript> {
        let eof_loc = initial_read.len() - EOF_SIZE;
        let magic_bytes_loc = eof_loc + (EOF_SIZE - MAGIC_BYTES.len());

        let magic_number = &initial_read[magic_bytes_loc..];
        if magic_number != MAGIC_BYTES {
            vortex_bail!("Malformed file, invalid magic bytes, got {magic_number:?}")
        }

        let version = u16::from_le_bytes(
            initial_read[eof_loc..eof_loc + 2]
                .try_into()
                .map_err(|e| vortex_err!("Version was not a u16 {e}"))?,
        );
        if version != VERSION {
            vortex_bail!("Malformed file, unsupported version {version}")
        }

        let ps_size = u16::from_le_bytes(
            initial_read[eof_loc + 2..eof_loc + 4]
                .try_into()
                .map_err(|e| vortex_err!("Postscript size was not a u16 {e}"))?,
        ) as usize;

        Postscript::read_flatbuffer_bytes(&initial_read[eof_loc - ps_size..eof_loc])
    }

    /// Parse the DType from the initial read.
    fn parse_dtype(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        dtype: Segment,
    ) -> VortexResult<DType> {
        let offset = usize::try_from(dtype.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::align_from(initial_read.slice(offset..offset + (dtype.length as usize)));
        let fbd_dtype = root::<fbd::DType>(&sliced_buffer)?;

        DType::try_from_view(fbd_dtype, sliced_buffer.clone())
    }

    /// Parse the FileLayout from the initial read.
    fn parse_file_layout(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        segment: Segment,
        dtype: DType,
    ) -> VortexResult<FileLayout> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let bytes = initial_read.slice(offset..offset + (segment.length as usize));

        let fb = root::<fb::FileLayout>(&bytes)?;
        let fb_root_layout = fb
            .root_layout()
            .ok_or_else(|| vortex_err!("FileLayout missing root layout"))?;

        let root_encoding = self
            .layout_ctx
            .lookup_layout(LayoutId(fb_root_layout.encoding()))
            .ok_or_else(|| {
                vortex_err!(
                    "FileLayout root layout encoding {} not found",
                    fb_root_layout.encoding()
                )
            })?;

        // SAFETY: We have validated the fb_root_layout at the beginning of this function
        let root_layout = unsafe {
            Layout::new_viewed_unchecked(
                root_encoding,
                dtype,
                bytes.clone(),
                fb_root_layout._tab.loc(),
                self.layout_ctx.clone(),
            )
        };

        let fb_segments = fb
            .segments()
            .ok_or_else(|| vortex_err!("FileLayout missing segments"))?;
        let segments = fb_segments.iter().map(Segment::try_from).try_collect()?;

        Ok(FileLayout::new(root_layout, segments))
    }

    /// Populate segments in the cache that were covered by the initial read.
    async fn populate_segments(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        file_layout: &FileLayout,
    ) -> VortexResult<()> {
        stream::iter(
            file_layout
                .segment_map()
                .iter()
                .enumerate()
                .filter(|(_, segment)| segment.offset > initial_offset)
                .map(|(idx, segment)| async move {
                    let segment_id = SegmentId::from(u32::try_from(idx)?);
                    let offset = usize::try_from(segment.offset - initial_offset)?;
                    let buffer = initial_read
                        .slice(offset..offset + (segment.length as usize))
                        .aligned(segment.alignment);

                    self.segment_cache.put(segment_id, buffer).await
                }),
        )
        .collect::<FuturesUnordered<_>>()
        .await
        .try_collect::<()>()
        .await
    }
}
