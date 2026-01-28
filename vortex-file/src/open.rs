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
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::segments::NoOpSegmentCache;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentCacheMetrics;
use vortex_layout::segments::SegmentCacheSourceAdapter;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SharedSegmentSource;
use vortex_layout::session::LayoutSessionExt;
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

const INITIAL_READ_SIZE: usize = MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE;

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
    metrics: Option<VortexMetrics>,
}

pub trait OpenOptionsSessionExt: ArraySessionExt + LayoutSessionExt + RuntimeSessionExt {
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
            metrics: None,
        }
    }
}
impl<S: ArraySessionExt + LayoutSessionExt + RuntimeSessionExt> OpenOptionsSessionExt for S {}

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
        self.metrics = Some(metrics);
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
        use vortex_io::file::std_file::FileReadAdapter;
        let handle = self.session.handle();
        let source = Arc::new(FileReadAdapter::open(path, handle)?);
        self.open(source).await
    }

    /// Open a Vortex file from an in-memory buffer.
    pub fn open_buffer<B: Into<ByteBuffer>>(self, buffer: B) -> VortexResult<VortexFile> {
        // We know this is in memory, so we can open it synchronously.
        block_on(
            self.with_initial_read_size(0)
                .without_segment_cache()
                .open_read(buffer.into()),
        )
    }

    /// An API for opening a [`VortexFile`] using any [`VortexReadAt`] implementation.
    pub async fn open_read<R: VortexReadAt + Clone>(self, reader: R) -> VortexResult<VortexFile> {
        let metrics = self.metrics.clone().unwrap_or_default();
        let footer = if let Some(footer) = self.footer {
            footer
        } else {
            self.read_footer(&reader).await?
        };

        let segment_cache = Arc::new(SegmentCacheMetrics::new(
            InitialReadSegmentCache {
                initial: self.initial_read_segments,
                fallback: self.segment_cache,
            },
            metrics.clone(),
        ));

        // Create a segment source backed by the VortexRead implementation.
        let segment_source = Arc::new(SharedSegmentSource::new(FileSegmentSource::open(
            footer.segment_map().clone(),
            reader,
            self.session.handle(),
            metrics.clone(),
        )));

        // Wrap up the segment source to first resolve segments from the initial read cache.
        let segment_source = Arc::new(SegmentCacheSourceAdapter::new(
            segment_cache,
            segment_source,
        ));

        Ok(VortexFile {
            footer,
            segment_source,
            session: self.session.clone(),
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
            .await?;

        let mut deserializer = Footer::deserializer(initial_read, self.session.clone())
            .with_size(file_size)
            .with_some_dtype(self.dtype.clone());

        let footer = loop {
            match deserializer.deserialize()? {
                DeserializeStep::NeedMoreData { offset, len } => {
                    let more_data = read.read_at(offset, len, Alignment::none()).await?;
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
        use vortex_io::file::object_store::ObjectStoreSource;

        let handle = self.session.handle();
        let source = Arc::new(ObjectStoreSource::new(
            object_store.clone(),
            path.into(),
            handle,
        ));
        self.open(source).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::future::BoxFuture;
    use vortex_array::IntoArray;
    use vortex_array::expr::session::ExprSession;
    use vortex_array::session::ArraySession;
    use vortex_buffer::Buffer;
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
        ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
            self.total_read.fetch_add(length, Ordering::Relaxed);
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

    #[tokio::test]
    async fn test_initial_read_size() {
        // Create a large file (> 1MB)
        let mut buf = ByteBufferMut::empty();
        let mut session = VortexSession::empty()
            .with::<VortexMetrics>()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ExprSession>()
            .with::<RuntimeSession>();

        crate::register_default_encodings(&mut session);

        // 1.5M integers -> ~6MB. We use a pattern to avoid Sequence encoding.
        let array = Buffer::from(
            (0i32..1_500_000)
                .map(|i| if i % 2 == 0 { i } else { -i })
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
            total_read: total_read.clone(),
            first_read_len: first_read_len.clone(),
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
}
