// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::executor::block_on;
use parking_lot::RwLock;
use vortex_array::ArrayRegistry;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail};
use vortex_io::file::IntoReadSource;
use vortex_io::runtime::Handle;
use vortex_io::{InstrumentedReadAt, VortexReadAt};
use vortex_layout::segments::{
    NoOpSegmentCache, SegmentCache, SegmentCacheMetrics, SegmentCacheSourceAdapter, SegmentId,
    SharedSegmentSource,
};
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;
use vortex_utils::aliases::hash_map::HashMap;

use crate::footer::Footer;
use crate::segments::{FileSegmentSource, InitialReadSegmentCache};
use crate::{DEFAULT_REGISTRY, DeserializeStep, EOF_SIZE, MAX_POSTSCRIPT_SIZE, VortexFile};

const INITIAL_READ_SIZE: usize = 1 << 20; // 1 MB

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions {
    /// The handle used by the open file.
    handle: Option<Handle>,
    /// Cache to use for file segments.
    segment_cache: Arc<dyn SegmentCache>,
    /// The number of bytes to read when parsing the footer.
    initial_read_size: usize,
    /// The registry of array encodings.
    registry: Arc<ArrayRegistry>,
    /// The registry of layouts.
    layout_registry: Arc<LayoutRegistry>,
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

impl Default for VortexOpenOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl VortexOpenOptions {
    /// Create a new [`VortexOpenOptions`] with the expected options for the file source.
    ///
    /// This should not be used directly, instead public API clients are expected to
    /// access either `VortexOpenOptions::new()` or `VortexOpenOptions::memory()`
    pub fn new() -> Self {
        Self {
            handle: Handle::find(),
            segment_cache: Arc::new(NoOpSegmentCache),
            initial_read_size: INITIAL_READ_SIZE,
            registry: DEFAULT_REGISTRY.clone(),
            layout_registry: Arc::new(LayoutRegistry::default()),
            file_size: None,
            dtype: None,
            footer: None,
            initial_read_segments: Default::default(),
            metrics: VortexMetrics::default(),
        }
    }

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

    /// Configure a [`Handle`] to use for opening the file.
    ///
    /// **Warning**: it is important that the runtime associated with the handle remains alive
    /// while the file is being used. If the runtime is dropped, any I/O operations on the
    /// file will fail.
    ///
    /// We tried to enforce this with Rust lifetimes, but sadly Rust async cannot express scoped
    /// futures in a safe way, so we need static lifetimes for now. If you're interested in the
    /// details, see [this post](https://without.boats/blog/the-scoped-task-trilemma/).
    pub fn with_handle(mut self, handle: Handle) -> Self {
        self.handle = Some(handle);
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
        let Some(handle) = self.handle.clone() else {
            vortex_bail!("VortexOpenOptions::handle must be set, or else be running inside Tokio");
        };
        self.open_read_at(handle.open_read(source)?).await
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
        let read = Arc::new(InstrumentedReadAt::new(Arc::new(read), &self.metrics));

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

        let mut deserializer = Footer::deserializer(initial_read)
            .with_size(file_size)
            .with_some_dtype(self.dtype.clone())
            .with_array_registry(self.registry.clone())
            .with_layout_registry(self.layout_registry.clone());

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
        use std::path::Path;

        use vortex_io::file::object_store::ObjectStoreReadSource;

        // If the file is local, we much prefer to use TokioFile since object store re-opens the
        // file on every read. This check is a little naive... but we hope that ObjectStore will
        // soon expose the scheme in a way that we can check more thoroughly.
        // See: https://github.com/apache/arrow-rs-object-store/issues/259
        let local_path = Path::new("/").join(path);
        if local_path.exists() {
            // Local disk is too fast to justify prefetching.
            self.open(local_path).await
        } else {
            self.open(ObjectStoreReadSource::new(
                object_store.clone(),
                path.into(),
            ))
            .await
        }
    }
}
