// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::{StreamExt, pin_mut};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_io::{Dispatch, InstrumentedReadAt, IoDispatcher, VortexReadAt};
use vortex_layout::segments::{SegmentEvents, SegmentId};
use vortex_utils::aliases::dash_map::DashMap;

use crate::driver::CoalescedDriver;
use crate::footer::DeserializeStep;
use crate::segments::{
    InitialReadSegmentCache, MokaSegmentCache, NoOpSegmentCache, SegmentCache, SegmentCacheMetrics,
    SegmentCacheSourceAdapter,
};
use crate::{EOF_SIZE, FileType, Footer, MAX_POSTSCRIPT_SIZE, VortexFile, VortexOpenOptions};

#[cfg(feature = "tokio")]
static TOKIO_DISPATCHER: std::sync::LazyLock<IoDispatcher> =
    std::sync::LazyLock::new(|| IoDispatcher::new_tokio(1));

/// A type of Vortex file that supports any [`VortexReadAt`] implementation.
///
/// This is a reasonable choice for files backed by a network since it performs I/O coalescing.
// TODO(ngates): rename to TokioVortexFile
pub struct GenericVortexFile;

impl FileType for GenericVortexFile {
    type Options = GenericFileOptions;
}

impl VortexOpenOptions<GenericVortexFile> {
    const INITIAL_READ_SIZE: u64 = 1 << 20; // 1 MB

    /// Open a file using the provided [`VortexReadAt`] implementation.
    pub fn file() -> Self {
        Self::new(Default::default())
            // Start with an initial in-memory cache of 256MB.
            // TODO(ngates): would it be better to default to a home directory disk cache?
            .with_segment_cache(Arc::new(MokaSegmentCache::new(256 << 20)))
            .with_initial_read_size(Self::INITIAL_READ_SIZE)
    }

    /// Configure the initial read size for the Vortex file.
    pub fn with_initial_read_size(mut self, initial_read_size: u64) -> Self {
        self.options.initial_read_size = initial_read_size;
        self
    }

    /// Configure a custom [`SegmentCache`].
    pub fn with_segment_cache(mut self, segment_cache: Arc<dyn SegmentCache>) -> Self {
        self.options.segment_cache = segment_cache;
        self
    }

    /// Disable segment caching entirely.
    pub fn without_segment_cache(self) -> Self {
        self.with_segment_cache(Arc::new(NoOpSegmentCache))
    }

    pub fn with_io_concurrency(mut self, io_concurrency: usize) -> Self {
        self.options.io_concurrency = io_concurrency;
        self
    }

    /// Blocking call to open a Vortex file using the provided [`std::path::Path`].
    #[cfg(feature = "tokio")]
    pub fn open_blocking(self, read: impl AsRef<std::path::Path>) -> VortexResult<VortexFile> {
        // Since we dispatch all I/O to a dedicated Tokio dispatcher thread, we can just
        // block-on the async call to open.
        futures::executor::block_on(self.open(read))
    }

    /// Open a Vortex file using the provided [`std::path::Path`].
    #[cfg(feature = "tokio")]
    pub async fn open(mut self, read: impl AsRef<std::path::Path>) -> VortexResult<VortexFile> {
        self.options.io_dispatcher = TOKIO_DISPATCHER.clone();
        self.open_read_at(vortex_io::TokioFile::open(read)?).await
    }

    /// Low-level API for opening any [`VortexReadAt`]. Note that the user is responsible for
    /// ensuring the `VortexReadAt` implementation is compatible with the chosen I/O dispatcher.
    pub async fn open_read_at<R: VortexReadAt + Send + Sync>(
        self,
        read: R,
    ) -> VortexResult<VortexFile> {
        let read = Arc::new(read);

        let footer = if let Some(footer) = self.footer {
            footer
        } else {
            self.read_footer(read.clone()).await?
        };

        let segment_cache = Arc::new(SegmentCacheMetrics::new(
            InitialReadSegmentCache {
                initial: self.options.initial_read_segments,
                fallback: self.options.segment_cache,
            },
            self.metrics.clone(),
        ));

        // We use segment events for driving I/O.
        let (events_source, events) = SegmentEvents::create();

        // Wrap the events source to first resolve segments from the initial read cache.
        let segment_source = Arc::new(SegmentCacheSourceAdapter::new(segment_cache, events_source));

        let read = InstrumentedReadAt::new(read.clone(), &self.metrics);

        let driver = CoalescedDriver::new(
            read.performance_hint(),
            footer.segment_map().clone(),
            events,
            self.metrics.clone(),
        );

        // Spawn an I/O driver onto the dispatcher.
        let io_concurrency = self.options.io_concurrency;
        let io_dispatcher = self.options.io_dispatcher.clone();
        self.options
            .io_dispatcher
            .dispatch(move || {
                async move {
                    // Drive the segment event stream.
                    let stream = driver
                        .map(|coalesced_req| {
                            let read = read.clone();
                            io_dispatcher
                                .dispatch(move || coalesced_req.launch(read))
                                .vortex_expect("Failed to dispatch I/O request")
                        })
                        .buffer_unordered(io_concurrency)
                        .map(|result| result.vortex_expect("infallible"));
                    pin_mut!(stream);

                    // Drive the stream to completion.
                    stream.collect::<()>().await
                }
            })
            .vortex_expect("Failed to spawn I/O driver");

        Ok(VortexFile {
            footer,
            segment_source,
            metrics: self.metrics,
        })
    }

    async fn read_footer<R: VortexReadAt + Send + Sync>(
        &self,
        read: Arc<R>,
    ) -> VortexResult<Footer> {
        // Fetch the file size and perform the initial read.
        let file_size = match self.file_size {
            None => self.dispatched_size(read.clone()).await?,
            Some(file_size) => file_size,
        };
        let initial_read_size = self
            .options
            .initial_read_size
            // Make sure we read enough to cover the postscript
            .max(MAX_POSTSCRIPT_SIZE as u64 + EOF_SIZE as u64)
            .min(file_size);
        let initial_offset = file_size - initial_read_size;
        let initial_read: ByteBuffer = self
            .dispatched_read(read.clone(), initial_offset..file_size)
            .await?;

        let mut deserializer = Footer::deserializer(initial_read)
            .with_size(file_size)
            .with_some_dtype(self.dtype.clone())
            .with_array_registry(self.registry.clone())
            .with_layout_registry(self.layout_registry.clone());

        let footer = loop {
            match deserializer.deserialize()? {
                DeserializeStep::NeedMoreData { offset, len } => {
                    let more_data = self
                        .dispatched_read(read.clone(), offset..offset + (len as u64))
                        .await?;
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

    /// Dispatch a [`VortexReadAt::size`] request onto the configured I/O dispatcher.
    async fn dispatched_size<R: VortexReadAt + Send + Sync>(
        &self,
        read: Arc<R>,
    ) -> VortexResult<u64> {
        Ok(self
            .options
            .io_dispatcher
            .dispatch(move || async move { read.size().await })?
            .await??)
    }

    /// Dispatch a read onto the configured I/O dispatcher.
    async fn dispatched_read<R: VortexReadAt + Send + Sync>(
        &self,
        read: Arc<R>,
        range: Range<u64>,
    ) -> VortexResult<ByteBuffer> {
        Ok(self
            .options
            .io_dispatcher
            .dispatch(move || async move { read.read_byte_range(range, Alignment::none()).await })?
            .await??)
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
            self.options
                .initial_read_segments
                .insert(segment_id, buffer);
        }
    }
}

#[cfg(feature = "object_store")]
impl VortexOpenOptions<GenericVortexFile> {
    pub async fn open_object_store(
        mut self,
        object_store: &Arc<dyn object_store::ObjectStore>,
        path: &str,
    ) -> VortexResult<VortexFile> {
        use std::path::Path;

        use vortex_io::ObjectStoreReadAt;

        // Object store _must_ use tokio for I/O.
        self.options.io_dispatcher = TOKIO_DISPATCHER.clone();

        // If the file is local, we much prefer to use TokioFile since object store re-opens the
        // file on every read. This check is a little naive... but we hope that ObjectStore will
        // soon expose the scheme in a way that we can check more thoroughly.
        // See: https://github.com/apache/arrow-rs-object-store/issues/259
        let local_path = Path::new("/").join(path);
        if local_path.exists() {
            // Local disk is too fast to justify prefetching.
            self.open(local_path).await
        } else {
            self.open_read_at(ObjectStoreReadAt::new(
                object_store.clone(),
                path.into(),
                None,
            ))
            .await
        }
    }
}

pub struct GenericFileOptions {
    segment_cache: Arc<dyn SegmentCache>,
    initial_read_size: u64,
    initial_read_segments: DashMap<SegmentId, ByteBuffer>,
    /// The number of concurrent I/O requests to spawn.
    /// This should be smaller than execution concurrency for coalescing to occur.
    io_concurrency: usize,
    /// The dispatcher to use for I/O requests.
    io_dispatcher: IoDispatcher,
}

impl Default for GenericFileOptions {
    fn default() -> Self {
        Self {
            segment_cache: Arc::new(NoOpSegmentCache),
            initial_read_size: 0,
            initial_read_segments: Default::default(),
            io_concurrency: 8,
            io_dispatcher: IoDispatcher::shared(),
        }
    }
}
