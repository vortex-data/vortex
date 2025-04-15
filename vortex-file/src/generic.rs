use std::ops::Range;
use std::sync::Arc;

use futures::{StreamExt, pin_mut};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::{Dispatch, InstrumentedReadAt, IoDispatcher, VortexReadAt};
use vortex_layout::segments::{SegmentEvents, SegmentSource};
use vortex_metrics::VortexMetrics;

use crate::driver::CoalescedDriver;
use crate::open::ParseResult;
use crate::segments::{
    InitialReadSegmentCache, MokaSegmentCache, SegmentCache, SegmentCacheMetrics,
    SegmentCacheSourceAdapter,
};
use crate::{
    EOF_SIZE, FileType, Footer, MAX_FOOTER_SIZE, SegmentSourceFactory, SegmentSpec, VortexFile,
    VortexOpenOptions,
};

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

    pub fn with_io_concurrency(mut self, io_concurrency: usize) -> Self {
        self.options.io_concurrency = io_concurrency;
        self
    }

    pub async fn open<R: VortexReadAt + Send + Sync>(self, read: R) -> VortexResult<VortexFile> {
        let read = Arc::new(read);

        let footer = if let Some(footer) = self.footer {
            footer
        } else {
            self.read_footer(&read).await?
        };

        let segment_cache = Arc::new(SegmentCacheMetrics::new(
            InitialReadSegmentCache {
                initial: self.initial_read_segments,
                fallback: self.segment_cache,
            },
            self.metrics.clone(),
        ));

        let segment_source_factory = Arc::new(GenericVortexFileIo {
            read,
            segment_map: footer.segment_map().clone(),
            segment_cache,
            options: self.options,
        });

        Ok(VortexFile {
            footer,
            segment_source_factory,
            metrics: self.metrics,
        })
    }

    async fn read_footer<R: VortexReadAt + Send + Sync>(&self, read: &R) -> VortexResult<Footer> {
        // Fetch the file size and perform the initial read.
        let file_size = match self.file_size {
            None => self.dispatched_size(read).await?,
            Some(file_size) => file_size,
        };
        let initial_read_size = self
            .initial_read_size
            // Make sure we read enough to cover the postscript
            .max(MAX_FOOTER_SIZE as u64 + EOF_SIZE as u64)
            .min(file_size);
        let mut initial_offset = file_size - initial_read_size;
        let mut initial_read: ByteBuffer = self
            .dispatched_read(read, initial_offset..file_size)
            .await?;

        let postscript = self.parse_postscript(&initial_read)?;

        loop {
            match self.parse_footer(&initial_read).await? {
                ParseResult::Ok(footer) => return Ok(footer),
                ParseResult::NeedFrom(need_more_offset) => {
                    // Do another range read.
                    assert!(
                        initial_offset > need_more_offset,
                        "Requested fewer bytes than we passed?"
                    );
                    let mut need_more_buffer =
                        ByteBufferMut::with_capacity((file_size - need_more_offset).try_into()?);
                    need_more_buffer.extend_from_slice(
                        &self
                            .dispatched_read(read, need_more_offset..initial_offset)
                            .await?,
                    );
                    need_more_buffer.extend_from_slice(&initial_read);

                    initial_read = need_more_buffer.into();
                    initial_offset = need_more_offset;
                }
            }
        }
    }

    /// Dispatch a [`VortexReadAt::size`] request onto the configured I/O dispatcher.
    async fn dispatched_size<R: VortexReadAt + Send>(&self, read: &R) -> VortexResult<u64> {
        Ok(self
            .options
            .io_dispatcher
            .dispatch(|| async move { read.size().await })?
            .await??)
    }

    /// Dispatch a read onto the configured I/O dispatcher.
    async fn dispatched_read<R: VortexReadAt + Send>(
        &self,
        read: &R,
        range: Range<u64>,
    ) -> VortexResult<ByteBuffer> {
        Ok(self
            .options
            .io_dispatcher
            .dispatch(|| async move { read.read_byte_range(range, Alignment::none()).await })?
            .await??)
    }
}

struct GenericVortexFileIo<R> {
    read: Arc<R>,
    segment_map: Arc<[SegmentSpec]>,
    segment_cache: Arc<dyn SegmentCache>,
    options: GenericFileOptions,
}

impl<R: VortexReadAt + Send + Sync> SegmentSourceFactory for GenericVortexFileIo<R> {
    fn segment_source(&self, metrics: VortexMetrics) -> Arc<dyn SegmentSource> {
        // We use segment events for driving I/O.
        let (segment_source, events) = SegmentEvents::create();

        // Wrap the source to resolve segments from the initial read cache.
        let segment_source = Arc::new(SegmentCacheSourceAdapter::new(
            self.segment_cache.clone(),
            segment_source,
        ));

        let read = InstrumentedReadAt::new(self.read.clone(), &metrics);

        let driver = CoalescedDriver::new(
            read.performance_hint(),
            self.segment_map.clone(),
            events,
            metrics,
        );

        // Spawn an I/O driver onto the dispatcher.
        let io_concurrency = self.options.io_concurrency;
        self.options
            .io_dispatcher
            .dispatch(move || {
                async move {
                    // Drive the segment event stream.
                    let stream = driver
                        .map(|coalesced_req| coalesced_req.launch(&read))
                        .buffer_unordered(io_concurrency);
                    pin_mut!(stream);

                    // Drive the stream to completion.
                    stream.collect::<()>().await
                }
            })
            .vortex_expect("Failed to spawn I/O driver");

        segment_source
    }
}

#[cfg(feature = "object_store")]
impl VortexOpenOptions<GenericVortexFile> {
    pub async fn open_object_store(
        self,
        object_store: &Arc<dyn object_store::ObjectStore>,
        path: &str,
    ) -> VortexResult<VortexFile> {
        use std::path::Path;

        use vortex_io::{ObjectStoreReadAt, TokioFile};

        // If the file is local, we much prefer to use TokioFile since object store re-opens the
        // file on every read. This check is a little naive... but we hope that ObjectStore will
        // soon expose the scheme in a way that we can check more thoroughly.
        // See: https://github.com/apache/arrow-rs-object-store/issues/259
        let local_path = Path::new("/").join(path);
        if local_path.exists() {
            // Local disk is too fast to justify prefetching.
            self.open(TokioFile::open(local_path)?).await
        } else {
            self.open(ObjectStoreReadAt::new(
                object_store.clone(),
                path.into(),
                None,
            ))
            .await
        }
    }
}

#[derive(Clone)]
pub struct GenericFileOptions {
    /// The number of concurrent I/O requests to spawn.
    /// This should be smaller than execution concurrency for coalescing to occur.
    io_concurrency: usize,
    /// The dispatcher to use for I/O requests.
    io_dispatcher: IoDispatcher,
}

impl Default for GenericFileOptions {
    fn default() -> Self {
        Self {
            io_concurrency: 8,
            io_dispatcher: IoDispatcher::shared(),
        }
    }
}
