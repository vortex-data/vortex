mod exec;
mod split_by;

use std::sync::Arc;

pub use exec::*;
use flatbuffers::root;
use itertools::Itertools;
pub use split_by::*;
use vortex_array::ContextRef;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::{dtype as fbd, footer2 as fb, FlatBuffer, ReadFlatBuffer};
use vortex_io::VortexReadAt;
use vortex_layout::segments::SegmentId;
use vortex_layout::{LayoutContextRef, LayoutData, LayoutId};

use crate::v2::footer::{FileLayout, Postscript, Segment};
use crate::v2::io::file::FileIoDriver;
use crate::v2::segments::{InMemorySegmentCache, NoOpSegmentCache, SegmentCache};
use crate::v2::VortexFile;
use crate::{EOF_SIZE, MAGIC_BYTES, VERSION};

const INITIAL_READ_SIZE: u64 = 1 << 20; // 1 MB

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions {
    /// The Vortex Array encoding context.
    ctx: ContextRef,
    /// The Vortex Layout encoding context.
    layout_ctx: LayoutContextRef,
    /// An optional, externally provided, file size.
    file_size: Option<u64>,
    /// An optional, externally provided, file layout.
    file_layout: Option<FileLayout>,
    // TODO(ngates): also support a messages_middleware that can wrap a message cache to provide
    //  additional caching, metrics, or other intercepts, etc. It should support synchronous
    //  read + write of Map<MessageId, ByteBuffer> or similar.
    initial_read_size: u64,
    split_by: SplitBy,
    segment_cache: Option<Arc<dyn SegmentCache>>,
    execution_mode: Option<ExecutionMode>,
    // TODO(ngates): allow fully configurable I/O driver.
    io_concurrency: usize,
}

impl VortexOpenOptions {
    pub fn new(ctx: ContextRef) -> Self {
        Self {
            ctx,
            layout_ctx: LayoutContextRef::default(),
            file_size: None,
            file_layout: None,
            initial_read_size: INITIAL_READ_SIZE,
            split_by: SplitBy::Layout,
            segment_cache: None,
            execution_mode: None,
            // TODO(ngates): pick some numbers...
            io_concurrency: 16,
        }
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

    /// Configure how to split the file into batches for reading.
    ///
    /// Defaults to [`SplitBy::Layout`].
    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// Configure a custom [`SegmentCache`].
    pub fn with_segment_cache(mut self, segment_cache: Arc<dyn SegmentCache>) -> Self {
        self.segment_cache = Some(segment_cache);
        self
    }

    /// Disable segment caching entirely.
    pub fn without_segment_cache(self) -> Self {
        self.with_segment_cache(Arc::new(NoOpSegmentCache))
    }

    /// Configure the execution mode
    pub fn with_execution_mode(mut self, execution_mode: ExecutionMode) -> Self {
        self.execution_mode = Some(execution_mode);
        self
    }
}

impl VortexOpenOptions {
    /// Open the Vortex file using asynchronous IO.
    pub async fn open<R: VortexReadAt>(
        mut self,
        read: R,
    ) -> VortexResult<VortexFile<FileIoDriver<R>>> {
        // Set up our segment cache.
        let segment_cache = self
            .segment_cache
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Arc::new(InMemorySegmentCache::default()));

        // If we need to read the file layout, then do so.
        let file_layout = match self.file_layout.take() {
            None => self.read_file_layout(&read, segment_cache.as_ref()).await?,
            Some(file_layout) => file_layout,
        };

        // Set up the I/O driver.
        let io_driver = FileIoDriver {
            read,
            file_layout: file_layout.clone(),
            concurrency: self.io_concurrency,
            segment_cache,
        };

        // Set up the execution driver.
        let exec_driver = self
            .execution_mode
            .unwrap_or(ExecutionMode::Inline)
            .into_driver();

        // Compute the splits of the file.
        let splits = self.split_by.splits(file_layout.root_layout())?.into();

        // Finally, create the VortexFile.
        Ok(VortexFile {
            ctx: self.ctx.clone(),
            file_layout,
            io_driver,
            exec_driver,
            splits,
        })
    }

    /// Read the [`FileLayout`] from the file.
    async fn read_file_layout<R: VortexReadAt>(
        &self,
        read: &R,
        segment_cache: &dyn SegmentCache,
    ) -> VortexResult<FileLayout> {
        // Fetch the file size and perform the initial read.
        let file_size = match self.file_size {
            None => read.size().await?,
            Some(file_size) => file_size,
        };
        let initial_read_size = self.initial_read_size.min(file_size);
        let initial_offset = file_size - initial_read_size;
        let initial_read: ByteBuffer = read
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
                &read
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
        self.populate_segments(initial_offset, &initial_read, &file_layout, segment_cache)
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
        let _fb_encoding_id = fb_root_layout.encoding();
        let root_layout = LayoutData::try_new_viewed(
            root_encoding,
            dtype,
            bytes.clone(),
            fb_root_layout._tab.loc(),
            self.layout_ctx.clone(),
        )?;

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
        segments: &dyn SegmentCache,
    ) -> VortexResult<()> {
        for (idx, segment) in file_layout.segment_map().iter().enumerate() {
            if segment.offset < initial_offset {
                // Skip segments that aren't in the initial read.
                continue;
            }
            let segment_id = SegmentId::from(u32::try_from(idx)?);
            let offset = usize::try_from(segment.offset - initial_offset)?;
            let buffer = initial_read.slice(offset..offset + (segment.length as usize));

            segments.put(segment_id, buffer).await?;
        }
        Ok(())
    }
}
