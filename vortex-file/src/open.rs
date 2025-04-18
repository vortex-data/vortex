use std::sync::{Arc, RwLock};

use flatbuffers::root;
use vortex_array::ArrayRegistry;
use vortex_array::aliases::hash_map::HashMap;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, ReadFlatBuffer, dtype as fbd};
use vortex_io::VortexReadAt;
use vortex_layout::segments::SegmentId;
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

use crate::footer::{FileStatistics, Footer, Postscript, PostscriptSegment};
use crate::segments::{NoOpSegmentCache, SegmentCache};
use crate::{DEFAULT_REGISTRY, EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION};

pub trait FileType: Sized {
    type Options;
}

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions<F: FileType> {
    /// File-specific options
    pub(crate) options: F::Options,
    /// The registry of array encodings.
    pub(crate) registry: Arc<ArrayRegistry>,
    /// The registry of layouts.
    pub(crate) layout_registry: Arc<LayoutRegistry>,
    /// An optional, externally provided, file size.
    pub(crate) file_size: Option<u64>,
    /// An optional, externally provided, DType.
    pub(crate) dtype: Option<DType>,
    /// An optional, externally provided, file layout.
    // TODO(ngates): add an optional DType so we only read the layout segment.
    footer: Option<Footer>,
    pub(crate) segment_cache: Arc<dyn SegmentCache>,
    pub(crate) initial_read_size: u64,
    pub(crate) initial_read_segments: RwLock<HashMap<SegmentId, ByteBuffer>>,
    pub(crate) metrics: VortexMetrics,
}

impl<F: FileType> VortexOpenOptions<F> {
    pub(crate) fn new(options: F::Options) -> Self {
        Self {
            options,
            registry: DEFAULT_REGISTRY.clone(),
            layout_registry: Arc::new(LayoutRegistry::default()),
            file_size: None,
            dtype: None,
            footer: None,
            segment_cache: Arc::new(NoOpSegmentCache),
            initial_read_size: 0,
            initial_read_segments: Default::default(),
            metrics: VortexMetrics::default(),
        }
    }

    /// Configure a Vortex array registry.
    pub fn with_array_registry(mut self, registry: Arc<ArrayRegistry>) -> Self {
        self.registry = registry;
        self
    }

    /// Configure a Vortex array registry.
    pub fn with_layout_registry(mut self, registry: Arc<LayoutRegistry>) -> Self {
        self.layout_registry = registry;
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

    /// Configure a known DType.
    ///
    /// If this is provided, then the Vortex file may be opened with fewer I/O requests.
    ///
    /// For Vortex files that do not contain a `DType`, this is required.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Configure a known file layout.
    ///
    /// If this is provided, then the Vortex file can be opened without performing any I/O.
    /// Once open, the [`Footer`] can be accessed via [`crate::VortexFile::footer`].
    pub fn with_footer(mut self, footer: Footer) -> Self {
        self.dtype = Some(footer.layout().dtype().clone());
        self.footer = Some(footer);
        self
    }

    /// Configure the initial read size for the Vortex file.
    pub fn with_initial_read_size(mut self, initial_read_size: u64) -> Self {
        self.initial_read_size = initial_read_size;
        self
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

    /// Configure a custom [`VortexMetrics`].
    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }
}

impl<F: FileType> VortexOpenOptions<F> {
    /// Read the [`Footer`] from the file.
    pub(crate) async fn read_footer<R: VortexReadAt>(&self, read: &R) -> VortexResult<Footer> {
        if let Some(footer) = self.footer.as_ref() {
            return Ok(footer.clone());
        }

        // Fetch the file size and perform the initial read.
        let file_size = match self.file_size {
            None => read.size().await?,
            Some(file_size) => file_size,
        };
        let initial_read_size = self
            .initial_read_size
            // Make sure we read enough to cover the postscript
            .max(MAX_FOOTER_SIZE as u64 + EOF_SIZE as u64)
            .min(file_size);
        let mut initial_offset = file_size - initial_read_size;
        let mut initial_read: ByteBuffer = read
            .read_byte_range(initial_offset..file_size, Alignment::none())
            .await?;

        // We know the initial read _must_ contain at least the Postscript.
        let postscript = self.parse_postscript(&initial_read)?;

        // If we haven't been provided a DType, we must read one from the file.
        let dtype_segment = self.dtype.is_none().then(|| postscript.dtype.ok_or_else(|| vortex_err!("Vortex file doesn't embed a DType and one has not been provided to VortexOpenOptions"))).transpose()?;

        // The other postscript segments are required, so now we figure out our the offset that
        // contains all the required segments.
        let mut read_more_offset = initial_offset;
        if let Some(dtype_segment) = &dtype_segment {
            read_more_offset = read_more_offset.min(dtype_segment.offset);
        }
        if let Some(stats_segment) = &postscript.statistics {
            read_more_offset = read_more_offset.min(stats_segment.offset);
        }
        read_more_offset = read_more_offset.min(postscript.layout.offset);

        // Read more bytes if necessary.
        if read_more_offset < initial_offset {
            log::info!(
                "Initial read from {} did not cover all footer segments, reading from {}",
                initial_offset,
                read_more_offset
            );

            let mut new_initial_read =
                ByteBufferMut::with_capacity(usize::try_from(file_size - read_more_offset)?);
            new_initial_read.extend_from_slice(
                &read
                    .read_byte_range(read_more_offset..initial_offset, Alignment::none())
                    .await?,
            );
            new_initial_read.extend_from_slice(&initial_read);

            initial_offset = read_more_offset;
            initial_read = new_initial_read.freeze();
        }

        // Now we read our initial segments.
        let dtype = dtype_segment
            .map(|segment| self.parse_dtype(initial_offset, &initial_read, &segment))
            .transpose()?
            .unwrap_or_else(|| self.dtype.clone().vortex_expect("DType was provided"));
        let file_stats = postscript
            .statistics
            .map(|segment| {
                self.parse_flatbuffer::<FileStatistics>(initial_offset, &initial_read, &segment)
            })
            .transpose()?;
        let footer = self.parse_file_layout(
            initial_offset,
            &initial_read,
            &postscript.layout,
            dtype,
            file_stats,
        )?;

        // If the initial read happened to cover any segments, then we can populate the
        // segment cache
        self.populate_initial_segments(initial_offset, &initial_read, &footer);

        Ok(footer)
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
        segment: &PostscriptSegment,
    ) -> VortexResult<DType> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::align_from(initial_read.slice(offset..offset + (segment.length as usize)));
        let fbd_dtype = root::<fbd::DType>(&sliced_buffer)?;

        DType::try_from_view(fbd_dtype, sliced_buffer.clone())
    }

    /// Parse a [`ReadFlatBuffer`] from the initial read buffer.
    fn parse_flatbuffer<T: ReadFlatBuffer<Error = VortexError>>(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        segment: &PostscriptSegment,
    ) -> VortexResult<T> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::align_from(initial_read.slice(offset..offset + (segment.length as usize)));
        T::read_flatbuffer_bytes(&sliced_buffer)
    }

    /// Parse the rest of the footer from the initial read.
    fn parse_file_layout(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        layout_segment: &PostscriptSegment,
        dtype: DType,
        file_stats: Option<FileStatistics>,
    ) -> VortexResult<Footer> {
        let offset = usize::try_from(layout_segment.offset - initial_offset)?;
        let bytes = FlatBuffer::align_from(
            initial_read.slice(offset..offset + (layout_segment.length as usize)),
        );
        Footer::from_flatbuffer(
            bytes,
            dtype,
            file_stats,
            &self.registry,
            &self.layout_registry,
        )
    }

    /// Populate segments in the cache that were covered by the initial read.
    fn populate_initial_segments(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        footer: &Footer,
    ) {
        let first_idx = footer
            .segment_map()
            .partition_point(|segment| segment.offset < initial_offset);

        let mut initial_segments = self
            .initial_read_segments
            .write()
            .vortex_expect("poisoned lock");

        for idx in first_idx..footer.segment_map().len() {
            let segment = &footer.segment_map()[idx];
            let segment_id =
                SegmentId::from(u32::try_from(idx).vortex_expect("Invalid segment ID"));
            let offset =
                usize::try_from(segment.offset - initial_offset).vortex_expect("Invalid offset");
            let buffer = initial_read
                .slice(offset..offset + (segment.length as usize))
                .aligned(segment.alignment);
            initial_segments.insert(segment_id, buffer);
        }
    }
}
