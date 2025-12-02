// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::executor::block_on;
use parking_lot::RwLock;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
#[cfg(not(target_arch = "wasm32"))]
use vortex_io::InstrumentedReadAt;
use vortex_io::VortexReadAt;
use vortex_io::file::IntoReadSource;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::segments::NoOpSegmentCache;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentCacheMetrics;
use vortex_layout::segments::SegmentCacheSourceAdapter;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SharedSegmentSource;
use vortex_layout::session::LayoutSessionExt;
use vortex_metrics::MetricsSessionExt;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::DeserializeStep;
use crate::EOF_SIZE;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VortexFile;
use crate::footer::Footer;
use crate::segments::FileSegmentSource;
use crate::segments::InitialReadSegmentCache;

const INITIAL_READ_SIZE: usize = 1 << 20; // 1 MB

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions {
    /// The session to use for opening the file.
    session: VortexSession,
    /// Cache to use for file segments.
    segment_cache: Arc<dyn SegmentCache>,
    /// The number of bytes to read when parsing the footer.
    initial_read_size: usize,
    /// An optional, externally provided, file size.
    file_size: Option<u64>,
    /// An optional, externally provided, DType.
    dtype: Option<DType>,
    /// An optional, externally provided, file layout.
    footer: Option<Footer>,
    /// The segments read during the initial read.
    initial_read_segments: RwLock<HashMap<SegmentId, ByteBuffer>>,
    /// A metrics registry for the file.
    metrics: VortexMetrics,
}

pub trait OpenOptionsSessionExt:
    ArraySessionExt + LayoutSessionExt + MetricsSessionExt + RuntimeSessionExt
{
    /// Create a new [`VortexOpenOptions`] using the provided session to open a file.
    fn open_options(&self) -> VortexOpenOptions {
        VortexOpenOptions {
            session: self.session(),
            segment_cache: Arc::new(NoOpSegmentCache),
            initial_read_size: INITIAL_READ_SIZE,
            file_size: None,
            dtype: None,
            footer: None,
            initial_read_segments: Default::default(),
            metrics: self.metrics(),
        }
    }
}
impl<S: ArraySessionExt + LayoutSessionExt + MetricsSessionExt + RuntimeSessionExt>
    OpenOptionsSessionExt for S
{
}

impl VortexOpenOptions {
    /// Configure the initial read size for the Vortex file.
    pub fn with_initial_read_size(mut self, initial_read_size: usize) -> Self {
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

    /// Configure a custom [`VortexMetrics`].
    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Open a Vortex file using the provided I/O source.
    ///
    /// This is the most common way to open a [`VortexFile`] and tends to provide the best
    /// out-of-the-box performance. The underlying I/O system will continue to be optimised for
    /// different file systems and object stores so we encourage users to use this method
    /// whenever possible and file issues if they encounter problems.
    pub async fn open<S: IntoReadSource>(self, source: S) -> VortexResult<VortexFile> {
        let handle = self.session.handle();
        let metrics = self.metrics.clone();
        self.open_read_at(handle.open_read(source, metrics)?).await
    }

    /// Open a Vortex file from an in-memory buffer.
    pub fn open_buffer<B: Into<ByteBuffer>>(self, buffer: B) -> VortexResult<VortexFile> {
        // We know this is in memory, so we can open it synchronously.
        block_on(
            self.with_initial_read_size(0)
                .without_segment_cache()
                .open_read_at(buffer.into()),
        )
    }

    /// An API for opening a [`VortexFile`] using any [`VortexReadAt`] implementation.
    ///
    /// This is a low-level API and we strongly recommend using [`VortexOpenOptions::open`].
    pub async fn open_read_at<R: VortexReadAt>(self, read: R) -> VortexResult<VortexFile> {
        // On WASM, skip instrumentation because it uses std::time which is not available.
        #[cfg(target_arch = "wasm32")]
        let read: Arc<dyn VortexReadAt> = Arc::new(read);
        #[cfg(not(target_arch = "wasm32"))]
        let read: Arc<dyn VortexReadAt> =
            Arc::new(InstrumentedReadAt::new(Arc::new(read), &self.metrics));

        let footer = if let Some(footer) = self.footer {
            footer
        } else {
            self.read_footer(read.clone()).await?
        };

        let segment_cache = Arc::new(SegmentCacheMetrics::new(
            InitialReadSegmentCache {
                initial: self.initial_read_segments,
                fallback: self.segment_cache,
            },
            self.metrics.clone(),
        ));

        // Create a segment source backed by the VortexReadAt implementation.
        let segment_source = Arc::new(SharedSegmentSource::new(FileSegmentSource::new(
            footer.segment_map().clone(),
            read,
        )));

        // Wrap up the segment source to first resolve segments from the initial read cache.
        let segment_source = Arc::new(SegmentCacheSourceAdapter::new(
            segment_cache,
            segment_source,
        ));

        Ok(VortexFile {
            footer,
            segment_source,
            metrics: self.metrics,
            session: self.session.clone(),
        })
    }

    async fn read_footer(&self, read: Arc<dyn VortexReadAt>) -> VortexResult<Footer> {
        // Fetch the file size and perform the initial read.
        let file_size = match self.file_size {
            None => read.size().await?,
            Some(file_size) => file_size,
        };
        let mut initial_read_size = self
            .initial_read_size
            // Make sure we read enough to cover the postscript
            .max(MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE);
        if let Ok(file_size) = usize::try_from(file_size) {
            initial_read_size = initial_read_size.min(file_size);
        }

        let initial_offset = file_size - initial_read_size as u64;
        let initial_read: ByteBuffer = read
            .clone()
            .read_at(initial_offset, initial_read_size, Alignment::none())
            .await?;

        let mut deserializer = Footer::deserializer(initial_read, self.session.clone())
            .with_size(file_size)
            .with_some_dtype(self.dtype.clone());

        let footer = loop {
            match deserializer.deserialize()? {
                DeserializeStep::NeedMoreData { offset, len } => {
                    let more_data = read.clone().read_at(offset, len, Alignment::none()).await?;
                    deserializer.prefix_data(more_data);
                }
                DeserializeStep::NeedFileSize => unreachable!("We passed file_size above"),
                DeserializeStep::Done(footer) => break Ok::<_, VortexError>(footer),
            }
        }?;

        // If the initial read happened to cover any segments, then we can populate the
        // segment cache
        let initial_offset = file_size - (deserializer.buffer().len() as u64);
        self.populate_initial_segments(initial_offset, deserializer.buffer(), &footer);

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

        let mut initial_read_segments = self.initial_read_segments.write();

        for idx in first_idx..footer.segment_map().len() {
            let segment = &footer.segment_map()[idx];
            let segment_id =
                SegmentId::from(u32::try_from(idx).vortex_expect("Invalid segment ID"));
            let offset =
                usize::try_from(segment.offset - initial_offset).vortex_expect("Invalid offset");
            let buffer = initial_read
                .slice(offset..offset + (segment.length as usize))
                .aligned(segment.alignment);
            initial_read_segments.insert(segment_id, buffer);
        }
    }
}

#[cfg(feature = "object_store")]
impl VortexOpenOptions {
    pub async fn open_object_store(
        self,
        object_store: &Arc<dyn object_store::ObjectStore>,
        path: &str,
    ) -> VortexResult<VortexFile> {
        use vortex_io::file::object_store::ObjectStoreReadSource;

        self.open(ObjectStoreReadSource::new(
            object_store.clone(),
            path.into(),
        ))
        .await
    }
}
