// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::sync::Arc;

use dashmap::DashMap;
use flatbuffers::root;
use futures::executor::block_on;
use vortex_array::ArrayRegistry;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, ReadFlatBuffer, dtype as fbd};
use vortex_io::{PerformanceHint, ReadAt, VortexIO};
use vortex_layout::segments::SegmentId;
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

use crate::driver::{CoalescedDriver, DirectDriver, FileDriver};
use crate::footer::{FileStatistics, Footer, Postscript, PostscriptSegment};
use crate::segments::{
    InitialReadSegmentCache, MokaSegmentCache, SegmentCache, SegmentCacheMetrics,
};
use crate::{DEFAULT_REGISTRY, EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, VortexFile};

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions {
    /// We store the underlying I/O object in the open options and defer the error until
    /// the user finally calls `open()`.
    read: Option<VortexResult<Arc<dyn ReadAt>>>,

    /// The registry of array encodings.
    registry: Arc<ArrayRegistry>,
    /// The registry of layouts.
    layout_registry: Arc<LayoutRegistry>,
    /// An optional, externally provided, file size.
    file_size: Option<u64>,
    /// An optional, externally provided, DType.
    dtype: Option<DType>,
    /// An optional, externally provided, file layout.
    // TODO(ngates): add an optional DType so we only read the layout segment.
    footer: Option<Footer>,
    /// A metrics registry for the file.
    metrics: VortexMetrics,

    /// The driver used to create the file's segment source.
    driver: Box<dyn FileDriver>,
    /// The segment cache to use when reading from this file.
    segment_cache: Option<Arc<dyn SegmentCache>>,
    /// How many bytes to read initially when opening the file.
    /// Zero means only the necessary bytes will be read.
    initial_read_size: usize,
    /// An internal store for segments that were read during the initial read.
    initial_read_segments: DashMap<SegmentId, ByteBuffer>,
}

impl VortexOpenOptions {
    /// Create a new [`VortexOpenOptions`] with the expected options for the file source.
    pub fn new<R: VortexIO>(read: R) -> Self {
        let hint = read.performance_hint();
        Self {
            read: Some(read.into_read_at()),
            registry: DEFAULT_REGISTRY.clone(),
            // TODO(ngates): constructing this has overhead since we're about to replace it...
            //  We should make it mandatory in the new function, and encourage users to do
            //  VortexSession::open() instead? Possibly?
            layout_registry: Arc::new(LayoutRegistry::default()),
            driver: Box::new(DirectDriver),
            segment_cache: None,
            initial_read_size: hint.coalescing_window(),
            file_size: None,
            dtype: None,
            footer: None,
            metrics: VortexMetrics::default(),
            initial_read_segments: Default::default(),
        }
        .with_performance_hint(hint)
    }

    /// Create a new [`VortexOpenOptions`] with the expected options for the file source.
    pub fn new_file(path: impl AsRef<Path>) -> Self {
        Self::new(path.as_ref())
    }

    /// Open the file with the default options.
    pub fn open_file(path: impl AsRef<Path>) -> VortexResult<VortexFile> {
        Self::new(path.as_ref()).open()
    }

    #[cfg(feature = "object_store")]
    pub fn new_object_store(
        object_store: Arc<dyn object_store::ObjectStore>,
        path: impl Into<object_store::path::Path>,
    ) -> Self {
        Self::new(vortex_io::ObjectStoreIo::new(
            object_store,
            path.into(),
            None,
        ))
    }

    /// Configure the scan with the given performance hint.
    ///
    /// This will set up coalesced reads if [`PerformanceHint::coalescing_window`] is greater than
    /// zero, as well as set the initial read size to the coalescing window.
    pub fn with_performance_hint(mut self, hint: PerformanceHint) -> Self {
        self.initial_read_size = hint.coalescing_window();
        if hint.coalescing_window() > 0 {
            self.driver = Box::new(CoalescedDriver::new(hint));
            // Start with an initial in-memory cache of 256MB.
            // TODO(ngates): would it be better to default to a home directory disk cache?
            // TODO(ngates): we may actually just not want this at all...
            self.segment_cache = Some(Arc::new(MokaSegmentCache::new(256 << 20)))
        } else {
            self.driver = Box::new(DirectDriver);
            self.segment_cache = None;
        }
        self
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

    /// Configure a segment cache.
    pub fn with_segment_cache(mut self, segment_cache: Arc<dyn SegmentCache>) -> Self {
        self.segment_cache = Some(segment_cache);
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

    /// Configure a custom [`VortexMetrics`].
    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Open the Vortex file with the configured options.
    pub fn open(self) -> VortexResult<VortexFile> {
        block_on(self.open_internal())
    }

    /// Open the Vortex file asynchronously.
    pub fn open_async(self) -> impl Future<Output = VortexResult<VortexFile>> {
        self.open_internal()
    }

    async fn open_internal(mut self) -> VortexResult<VortexFile> {
        // We deferred the result to here just to avoid the user having to deal with too many
        // errors at construction time.
        let read = self
            .read
            .take()
            .vortex_expect("Builder missing read object")?;

        let footer = if let Some(footer) = self.footer {
            footer
        } else {
            self.read_footer(&read).await?
        };

        // Wrap up the segments that we covered in the initial read.
        let segment_cache = (!self.initial_read_segments.is_empty()).then(|| {
            Arc::new(SegmentCacheMetrics::new(
                InitialReadSegmentCache {
                    initial: self.initial_read_segments,
                    fallback: self.segment_cache,
                },
                self.metrics.clone(),
            )) as _
        });

        // Construct the segment source based on the configured driver.
        let segment_source = self.driver.create_segment_source(
            read,
            footer.segment_map().clone(),
            segment_cache,
            &self.metrics,
        )?;

        Ok(VortexFile {
            footer,
            segment_source,
            metrics: self.metrics,
        })
    }

    async fn read_footer(&self, read: &Arc<dyn ReadAt>) -> VortexResult<Footer> {
        // Fetch the file size and perform the initial read.
        let file_size = match self.file_size {
            None => read.size().await?,
            Some(file_size) => file_size,
        };
        let initial_read_size = self
            .initial_read_size
            // Make sure we read enough to cover the postscript
            .max(MAX_FOOTER_SIZE as usize + EOF_SIZE)
            .min(usize::try_from(file_size).unwrap_or(usize::MAX));
        let mut initial_offset = file_size - (initial_read_size as u64);
        let mut initial_read: ByteBuffer = read
            .read_range(initial_offset, initial_read_size, Alignment::none())
            .await?;

        let postscript = self.parse_postscript(&initial_read)?;

        // If we haven't been provided a DType, we must read one from the file.
        let dtype_segment = self
            .dtype
            .is_none()
            .then(|| {
                postscript.dtype.ok_or_else(|| {
                    vortex_err!(
                        "Vortex file doesn't embed a DType and none provided to VortexOpenOptions"
                    )
                })
            })
            .transpose()?;

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
        read_more_offset = read_more_offset.min(postscript.footer.offset);

        // Read more bytes if necessary.
        if read_more_offset < initial_offset {
            log::info!(
                "Initial read from {initial_offset} did not cover all footer segments, reading from {read_more_offset}"
            );

            let mut new_initial_read = ByteBufferMut::with_capacity_aligned(
                usize::try_from(file_size - read_more_offset)?,
                // We know our offset now points to a FlatBuffer, so align the new buffer.
                FlatBuffer::alignment(),
            );
            new_initial_read.extend_from_slice(
                &read
                    .read_range(
                        read_more_offset,
                        usize::try_from(initial_offset - read_more_offset)?,
                        Alignment::none(),
                    )
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
            .map(|segment| self.parse_file_statistics(initial_offset, &initial_read, &segment))
            .transpose()?;
        let footer = self.parse_footer(
            initial_offset,
            &initial_read,
            &postscript.footer,
            &postscript.layout,
            dtype,
            file_stats,
        )?;

        // If the initial read happened to cover any segments, then we can populate the
        // segment cache
        self.populate_initial_segments(initial_offset, &initial_read, &footer);

        Ok(footer)
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

        for idx in first_idx..footer.segment_map().len() {
            let segment = &footer.segment_map()[idx];
            let segment_id =
                SegmentId::from(u32::try_from(idx).vortex_expect("Invalid segment ID"));
            let offset =
                usize::try_from(segment.offset - initial_offset).vortex_expect("Invalid offset");
            let buffer = initial_read
                .slice(offset..offset + (segment.length as usize))
                .aligned(segment.alignment);
            self.initial_read_segments.insert(segment_id, buffer);
        }
    }

    /// Parse the postscript from the initial read.
    fn parse_postscript(&self, initial_read: &[u8]) -> VortexResult<Postscript> {
        if initial_read.len() < EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least EOF_SIZE ({}) bytes",
                EOF_SIZE
            );
        }
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

        if initial_read.len() < ps_size + EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least {} bytes to include the Postscript",
                ps_size + EOF_SIZE
            );
        }

        Postscript::read_flatbuffer_bytes(&initial_read[eof_loc - ps_size..eof_loc])
    }

    /// Parse the DType from the initial read.
    fn parse_dtype(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        segment: &PostscriptSegment,
    ) -> VortexResult<DType> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::copy_from(&initial_read[offset..offset + (segment.length as usize)]);
        let fbd_dtype = root::<fbd::DType>(&sliced_buffer)?;

        DType::try_from_view(fbd_dtype, sliced_buffer.clone())
    }

    /// Parse the [`FileStatistics`] from the initial read buffer.
    fn parse_file_statistics(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        segment: &PostscriptSegment,
    ) -> VortexResult<FileStatistics> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::copy_from(&initial_read[offset..offset + (segment.length as usize)]);
        FileStatistics::read_flatbuffer_bytes(&sliced_buffer)
    }

    /// Parse the rest of the footer from the initial read.
    fn parse_footer(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        footer_segment: &PostscriptSegment,
        layout_segment: &PostscriptSegment,
        dtype: DType,
        file_stats: Option<FileStatistics>,
    ) -> VortexResult<Footer> {
        let footer_offset = usize::try_from(footer_segment.offset - initial_offset)?;
        let footer_bytes = FlatBuffer::copy_from(
            &initial_read[footer_offset..footer_offset + (footer_segment.length as usize)],
        );

        let layout_offset = usize::try_from(layout_segment.offset - initial_offset)?;
        let layout_bytes = FlatBuffer::copy_from(
            &initial_read[layout_offset..layout_offset + (layout_segment.length as usize)],
        );

        Footer::from_flatbuffer(
            footer_bytes,
            layout_bytes,
            dtype,
            file_stats,
            &self.registry,
            &self.layout_registry,
        )
    }
}
