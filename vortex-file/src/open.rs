// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::executor::block_on;
use parking_lot::RwLock;
use vortex_array::dtype::DType;
use vortex_array::memory::MemorySessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::segments::InstrumentedSegmentCache;
use vortex_layout::segments::NoOpSegmentCache;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentCacheSourceAdapter;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::SharedSegmentSource;
use vortex_layout::session::LayoutSessionExt;
use vortex_metrics::DefaultMetricsRegistry;
use vortex_metrics::Label;
use vortex_metrics::MetricsRegistry;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::DeserializeStep;
use crate::EOF_SIZE;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VortexFile;
use crate::footer::Footer;
use crate::segments::BufferSegmentSource;
use crate::segments::FileSegmentSource;
use crate::segments::InitialReadSegmentCache;
use crate::segments::RequestMetrics;

const INITIAL_READ_SIZE: usize = MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE;

/// Open options for a Vortex file reader.
///
/// Options are session-bound because opening a file may need array, layout, dtype, runtime, and
/// memory registries. Construct with [`OpenOptionsSessionExt::open_options`] for the common path.
///
/// The opener first resolves a [`Footer`], then creates a segment source for later scans. Known
/// file metadata can be supplied up front to avoid IO during footer discovery.
pub struct VortexOpenOptions {
    /// The session to use for opening the file.
    session: VortexSession,
    /// Cache to use for file segments.
    segment_cache: Option<Arc<dyn SegmentCache>>,
    /// The number of bytes to read when parsing the footer.
    initial_read_size: usize,
    /// An optional, externally provided, file size.
    file_size: Option<u64>,
    /// An optional, externally provided, DType.
    dtype: Option<DType>,
    /// An optional, externally provided, file layout.
    footer: Option<Footer>,
    /// Whether to include user-defined metadata segments when opening the file.
    include_metadata: bool,
    /// The segments read during the initial read.
    initial_read_segments: RwLock<HashMap<SegmentId, ByteBuffer>>,
    /// A metrics registry for the file.
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
    /// Default labels applied to all the file's metrics
    labels: Vec<Label>,
    /// Whether to cache file's LayoutReader between scans
    cache_layout_reader: bool,
}

/// Extension trait for constructing [`VortexOpenOptions`] from a session.
pub trait OpenOptionsSessionExt:
    ArraySessionExt + LayoutSessionExt + RuntimeSessionExt + MemorySessionExt
{
    /// Create a new [`VortexOpenOptions`] using the provided session to open a file.
    fn open_options(&self) -> VortexOpenOptions {
        VortexOpenOptions {
            session: self.session(),
            segment_cache: None,
            initial_read_size: INITIAL_READ_SIZE,
            file_size: None,
            dtype: None,
            footer: None,
            include_metadata: false,
            initial_read_segments: Default::default(),
            metrics_registry: None,
            labels: Vec::default(),
            cache_layout_reader: false,
        }
    }
}
impl<S: ArraySessionExt + LayoutSessionExt + RuntimeSessionExt + MemorySessionExt>
    OpenOptionsSessionExt for S
{
}

impl VortexOpenOptions {
    /// Configure how many bytes to read from the end of the file before parsing the footer.
    ///
    /// The actual read is at least large enough to contain the maximum postscript and EOF marker,
    /// and no larger than the file. Increase this when you expect footer segments to be near the
    /// end and want to avoid a second footer read.
    pub fn with_initial_read_size(mut self, initial_read_size: usize) -> Self {
        self.initial_read_size = initial_read_size;
        self
    }

    /// Cache the file's [`LayoutReader`](vortex_layout::LayoutReader) between scans.
    ///
    /// This avoids rebuilding the reader tree for repeated scans of the same [`VortexFile`], at the
    /// cost of keeping reader state alive for the lifetime of the file handle.
    pub fn with_layout_reader_cache(mut self) -> Self {
        self.cache_layout_reader = true;
        self
    }

    /// Configure a custom [`SegmentCache`].
    ///
    /// The cache is checked before the underlying file segment source. Segments covered by the
    /// initial footer read are also inserted into an internal first-read cache.
    pub fn with_segment_cache(mut self, segment_cache: Arc<dyn SegmentCache>) -> Self {
        self.segment_cache = Some(segment_cache);
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

    /// Configure a known file size.
    ///
    /// This helps to prevent an I/O request to discover the size of the file.
    /// Of course, all bets are off if you pass an incorrect value.
    pub fn with_some_file_size(mut self, file_size: Option<u64>) -> Self {
        self.file_size = file_size;
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

    /// Configure a known file footer.
    ///
    /// If this is provided, then the Vortex file can be opened without performing any I/O.
    /// Once open, the [`Footer`] can be accessed via [`crate::VortexFile::footer`].
    pub fn with_footer(mut self, footer: Footer) -> Self {
        self.dtype = Some(footer.layout().dtype().clone());
        self.footer = Some(footer);
        self
    }

    /// Include user-defined metadata segments when opening the file.
    ///
    /// By default, opening a file reads only the metadata required to interpret the layout.
    /// Enabling this option loads all metadata segments named by the postscript, which may require
    /// additional reads before the footer is returned.
    pub fn include_metadata(mut self) -> Self {
        self.include_metadata = true;
        self
    }

    /// Configure whether user-defined metadata segments are included when opening the file.
    ///
    /// Enabling this option loads all metadata segments named by the postscript, which may require
    /// additional reads before the footer is returned.
    pub fn with_include_metadata(mut self, include_metadata: bool) -> Self {
        self.include_metadata = include_metadata;
        self
    }

    /// Configure a custom [`MetricsRegistry`] implementation.
    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics);
        self
    }

    /// Adds labels to all the file's metrics.
    pub fn with_labels(mut self, labels: Vec<Label>) -> Self {
        self.labels.extend(labels);
        self
    }

    /// Open a Vortex file using the provided I/O source.
    ///
    /// This is the most common way to open a [`VortexFile`] and tends to provide the best
    /// out-of-the-box performance. The underlying I/O system will continue to be optimised for
    /// different file systems and object stores so we encourage users to use this method
    /// whenever possible and file issues if they encounter problems.
    pub async fn open(self, source: Arc<dyn VortexReadAt>) -> VortexResult<VortexFile> {
        self.open_read(source).await
    }

    /// Open a Vortex file from a filesystem path.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn open_path(self, path: impl AsRef<std::path::Path>) -> VortexResult<VortexFile> {
        use vortex_io::std_file::FileReadAt;
        let handle = self.session.handle();
        let allocator = self.session.allocator();
        let source = Arc::new(FileReadAt::open_with_allocator(path, handle, allocator)?);
        self.open(source).await
    }

    /// Open a Vortex file from an in-memory buffer.
    ///
    /// This uses a `BufferSegmentSource` that resolves segments synchronously
    /// by slicing the buffer directly, bypassing the async I/O pipeline.
    ///
    /// Segment cache and metrics registry settings are ignored for this path.
    pub fn open_buffer<B: Into<ByteBuffer>>(self, buffer: B) -> VortexResult<VortexFile> {
        let buffer: ByteBuffer = buffer.into();

        if self.segment_cache.is_some() {
            tracing::warn!("segment cache is ignored for in-memory `open_buffer`");
        }
        if self.metrics_registry.is_some() {
            tracing::warn!("metrics registry is ignored for in-memory `open_buffer`");
        }

        let cache_layout_reader = self.cache_layout_reader;
        let include_metadata = self.include_metadata;
        let mut opts = self.with_initial_read_size(0);

        let footer = match opts.footer.take() {
            Some(footer) => footer,
            None => block_on(opts.read_footer(&buffer))?,
        };

        let segment_source: Arc<dyn SegmentSource> = Arc::new(BufferSegmentSource::new(
            buffer,
            footer.segment_specs_with_metadata(),
        ));
        let metadata = if include_metadata {
            block_on(resolve_metadata(&footer, Arc::clone(&segment_source)))?
        } else {
            Arc::new(HashMap::new())
        };
        let file = VortexFile::new(footer, segment_source, opts.session).with_metadata(metadata);
        Ok(if cache_layout_reader {
            file.with_caching()
        } else {
            file
        })
    }

    /// Open a [`VortexFile`] using any [`VortexReadAt`] implementation.
    ///
    /// This is the common path for files, object stores, and custom random-access sources.
    pub async fn open_read<R: VortexReadAt + Clone>(self, reader: R) -> VortexResult<VortexFile> {
        let segment_cache = self
            .segment_cache
            .clone()
            .unwrap_or_else(|| Arc::new(NoOpSegmentCache));

        let metrics_registry = self
            .metrics_registry
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultMetricsRegistry::default()));

        let footer = if let Some(footer) = self.footer {
            footer
        } else {
            self.read_footer(&reader).await?
        };

        let segment_cache = Arc::new(InstrumentedSegmentCache::new(
            InitialReadSegmentCache {
                initial: self.initial_read_segments,
                fallback: segment_cache,
            },
            metrics_registry.as_ref(),
            self.labels.clone(),
        ));

        let metrics = RequestMetrics::new(metrics_registry.as_ref(), self.labels);

        // Create a segment source backed by the VortexRead implementation.
        let segment_source = Arc::new(SharedSegmentSource::new(FileSegmentSource::open(
            footer.segment_specs_with_metadata(),
            reader,
            self.session.handle(),
            metrics,
        )));

        // Wrap up the segment source to first resolve segments from the initial read cache.
        let segment_source: Arc<dyn SegmentSource> = Arc::new(SegmentCacheSourceAdapter::new(
            segment_cache,
            segment_source,
        ));

        let metadata = if self.include_metadata {
            resolve_metadata(&footer, Arc::clone(&segment_source)).await?
        } else {
            Arc::new(HashMap::new())
        };
        let file =
            VortexFile::new(footer, segment_source, self.session.clone()).with_metadata(metadata);
        Ok(if self.cache_layout_reader {
            file.with_caching()
        } else {
            file
        })
    }

    async fn read_footer(&self, read: &dyn VortexReadAt) -> VortexResult<Footer> {
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
            .read_at(initial_offset, initial_read_size, Alignment::none())
            .await?
            .try_into_host()?
            .await?;

        let mut deserializer = Footer::deserializer(initial_read, self.session.clone())
            .with_size(file_size)
            .with_some_dtype(self.dtype.clone());

        let footer = loop {
            match deserializer.deserialize()? {
                DeserializeStep::NeedMoreData { offset, len } => {
                    let more_data = read
                        .read_at(offset, len, Alignment::none())
                        .await?
                        .try_into_host()?
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

    /// Populate segments in the cache that were covered by the initial read.
    fn populate_initial_segments(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        footer: &Footer,
    ) {
        let mut initial_read_segments = self.initial_read_segments.write();

        for (idx, segment) in footer.segment_specs_with_metadata().iter().enumerate() {
            if segment.offset < initial_offset {
                continue;
            }
            let segment_id =
                SegmentId::from(u32::try_from(idx).vortex_expect("Invalid segment ID"));
            let offset =
                usize::try_from(segment.offset - initial_offset).vortex_expect("Invalid offset");
            let end = offset + segment.length as usize;
            if end > initial_read.len() {
                continue;
            }
            let buffer = initial_read.slice(offset..end).aligned(segment.alignment);
            initial_read_segments.insert(segment_id, buffer);
        }
    }
}

async fn resolve_metadata(
    footer: &Footer,
    segment_source: Arc<dyn SegmentSource>,
) -> VortexResult<Arc<HashMap<String, ByteBuffer>>> {
    let first_metadata_id = footer.segment_map().len();
    let requests = footer
        .metadata_segments()
        .enumerate()
        .map(|(index, (key, locator))| {
            let id = u32::try_from(first_metadata_id + index).map(SegmentId::from);
            let key = key.to_string();
            let alignment = locator.alignment;
            let segment_source = Arc::clone(&segment_source);
            async move {
                let handle = segment_source.request(id?).await?;
                let buffer = handle.try_into_host()?.await?;
                Ok::<_, VortexError>((
                    key,
                    ByteBuffer::copy_from_aligned(buffer.as_slice(), alignment),
                ))
            }
        });
    let metadata = futures::future::try_join_all(requests)
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();

    Ok(Arc::new(metadata))
}

#[cfg(feature = "object_store")]
impl VortexOpenOptions {
    /// Open a Vortex file from an `object_store` backend and path.
    pub async fn open_object_store(
        self,
        object_store: &Arc<dyn object_store::ObjectStore>,
        path: &str,
    ) -> VortexResult<VortexFile> {
        use vortex_io::object_store::ObjectStoreReadAt;

        let handle = self.session.handle();
        let allocator = self.session.allocator();
        let source = Arc::new(ObjectStoreReadAt::new_with_allocator(
            Arc::clone(object_store),
            path.into(),
            handle,
            allocator,
        ));
        self.open(source).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::future::BoxFuture;
    use vortex_array::IntoArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::memory::DefaultHostAllocator;
    use vortex_array::memory::HostAllocator;
    use vortex_array::memory::MemorySessionExt;
    use vortex_array::memory::WritableHostBuffer;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;

    use super::*;
    use crate::WriteOptionsSessionExt;

    #[derive(Clone)]
    // Define CountingRead struct
    struct CountingRead<R> {
        inner: R,
        total_read: Arc<AtomicUsize>,
        first_read_len: Arc<AtomicUsize>,
        reads: Arc<Mutex<Vec<(u64, usize)>>>,
    }

    impl<R: VortexReadAt + Clone> VortexReadAt for CountingRead<R> {
        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            self.inner.size()
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.total_read.fetch_add(length, Ordering::Relaxed);
            self.reads.lock().unwrap().push((offset, length));
            let _ = self.first_read_len.compare_exchange(
                0,
                length,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            self.inner.read_at(offset, length, alignment)
        }

        fn concurrency(&self) -> usize {
            self.inner.concurrency()
        }
    }

    #[derive(Debug)]
    struct CountingAllocator {
        allocations: Arc<AtomicUsize>,
    }

    impl HostAllocator for CountingAllocator {
        fn allocate(&self, len: usize, alignment: Alignment) -> VortexResult<WritableHostBuffer> {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            DefaultHostAllocator.allocate(len, alignment)
        }
    }

    #[tokio::test]
    async fn test_initial_read_size() {
        let session = vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();

        crate::register_default_encodings(&session);

        // Create a large file (> 1MB)
        let mut buf = ByteBufferMut::empty();

        // 1.5M integers -> ~6MB. We use high-entropy (pseudo-random) values so the data does not
        // compress well under any encoding (Sequence, RunEnd, Delta, ...), keeping the written
        // file comfortably above 1MB.
        let mut state = 0x9E37_79B9u32;
        let array = Buffer::from(
            (0i32..1_500_000)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    state as i32
                })
                .collect::<Vec<i32>>(),
        )
        .into_array();

        session
            .write_options()
            .write(&mut buf, array.to_array_stream())
            .await
            .unwrap();

        let buffer = ByteBuffer::from(buf);
        assert!(
            buffer.len() > 1024 * 1024,
            "Buffer length is only {} bytes",
            buffer.len()
        );

        let total_read = Arc::new(AtomicUsize::new(0));
        let first_read_len = Arc::new(AtomicUsize::new(0));
        let reader = CountingRead {
            inner: buffer,
            total_read: Arc::clone(&total_read),
            first_read_len: Arc::clone(&first_read_len),
            reads: Arc::new(Mutex::new(Vec::new())),
        };

        // Open the file
        let _file = session.open_options().open_read(reader).await.unwrap();

        // Assert that we read approximately the postscript size, not 1MB
        let first = first_read_len.load(Ordering::Relaxed);
        assert_eq!(
            first,
            MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE,
            "Read exactly the postscript size"
        );
        let read = total_read.load(Ordering::Relaxed);
        assert!(read < 1024 * 1024, "Read {} bytes, expected < 1MB", read);
    }

    #[tokio::test]
    async fn test_metadata_outside_initial_read_uses_targeted_read() -> VortexResult<()> {
        let session = vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);

        let metadata = ByteBuffer::copy_from(vec![0x5a; INITIAL_READ_SIZE * 2]);
        let mut output = ByteBufferMut::empty();
        let summary = session
            .write_options()
            .with_metadata_segment("outside", metadata.clone())
            .write(
                &mut output,
                Buffer::from(vec![1u32]).into_array().to_array_stream(),
            )
            .await?;
        let locator = *summary
            .footer()
            .metadata_segment("outside")
            .vortex_expect("metadata locator");
        let bytes = ByteBuffer::from(output);
        assert!(locator.offset < bytes.len() as u64 - INITIAL_READ_SIZE as u64);

        let default_total = Arc::new(AtomicUsize::new(0));
        let default_reads = Arc::new(Mutex::new(Vec::new()));
        let default_reader = CountingRead {
            inner: bytes.clone(),
            total_read: Arc::clone(&default_total),
            first_read_len: Arc::new(AtomicUsize::new(0)),
            reads: Arc::clone(&default_reads),
        };
        let default_file = session.open_options().open_read(default_reader).await?;
        assert!(default_file.metadata_segment("outside").is_none());
        assert!(
            !default_reads
                .lock()
                .unwrap()
                .contains(&(locator.offset, locator.length as usize))
        );
        // A default open must not amplify into the metadata: total bytes read stay
        // below the metadata segment's size (itself 2x the initial read).
        assert!(
            default_total.load(Ordering::Relaxed) < metadata.len(),
            "default open read {} bytes; metadata segment is {}",
            default_total.load(Ordering::Relaxed),
            metadata.len()
        );

        let metadata_reads = Arc::new(Mutex::new(Vec::new()));
        let metadata_reader = CountingRead {
            inner: bytes,
            total_read: Arc::new(AtomicUsize::new(0)),
            first_read_len: Arc::new(AtomicUsize::new(0)),
            reads: Arc::clone(&metadata_reads),
        };
        let file = session
            .open_options()
            .include_metadata()
            .open_read(metadata_reader)
            .await?;
        assert_eq!(
            file.metadata_segment("outside").map(ByteBuffer::as_slice),
            Some(metadata.as_slice())
        );
        assert!(
            metadata_reads
                .lock()
                .unwrap()
                .contains(&(locator.offset, locator.length as usize))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_with_footer_include_metadata_open_buffer_resolves() -> VortexResult<()> {
        let session = vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);

        let value = ByteBuffer::copy_from(b"supplied-footer metadata".to_vec());
        let mut output = ByteBufferMut::empty();
        let summary = session
            .write_options()
            .with_metadata_segment("key", value.clone())
            .write(
                &mut output,
                Buffer::from(vec![1u32]).into_array().to_array_stream(),
            )
            .await?;
        let footer = summary.footer().clone();
        let bytes = ByteBuffer::from(output);

        let file = session
            .open_options()
            .with_footer(footer.clone())
            .include_metadata()
            .open_buffer(bytes.clone())?;
        assert_eq!(
            file.metadata_segment("key").map(ByteBuffer::as_slice),
            Some(value.as_slice())
        );

        let default = session
            .open_options()
            .with_footer(footer)
            .open_buffer(bytes)?;
        assert!(default.metadata_segment("key").is_none());

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_open_path_uses_memory_session_allocator() {
        let session = vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();

        crate::register_default_encodings(&session);

        let mut buf = ByteBufferMut::empty();
        let array = Buffer::from((0i32..16_384).collect::<Vec<i32>>()).into_array();
        session
            .write_options()
            .write(&mut buf, array.to_array_stream())
            .await
            .unwrap();

        let file_path = std::env::temp_dir().join(format!(
            "vortex-open-memory-session-{}.vx",
            std::process::id()
        ));
        std::fs::write(&file_path, ByteBuffer::from(buf).as_slice()).unwrap();

        let allocations = Arc::new(AtomicUsize::new(0));
        let session = session.with_allocator(Arc::new(CountingAllocator {
            allocations: Arc::clone(&allocations),
        }));

        let _file = session.open_options().open_path(&file_path).await.unwrap();
        std::fs::remove_file(&file_path).unwrap();

        assert!(
            allocations.load(Ordering::Relaxed) > 0,
            "expected at least one host allocation from MemorySession"
        );
    }
}
