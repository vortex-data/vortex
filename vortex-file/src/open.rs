use std::marker::PhantomData;
use std::sync::Arc;

use flatbuffers::root;
use futures::stream::FuturesUnordered;
use futures::{StreamExt, TryStreamExt, stream};
use vortex_array::ArrayRegistry;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, ReadFlatBuffer, dtype as fbd};
use vortex_io::VortexReadAt;
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::SegmentId;
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

use crate::footer::{Footer, Postscript, SegmentSpec};
use crate::segments::{NoOpSegmentCache, SegmentCache};
use crate::{DEFAULT_REGISTRY, EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, VortexFile};

pub trait FileType: Sized {
    type Options: Clone;
    type Read: VortexReadAt;
    type ScanDriver: ScanDriver;

    fn scan_driver(
        read: Self::Read,
        options: Self::Options,
        footer: Footer,
        segment_cache: Arc<dyn SegmentCache>,
        metrics: VortexMetrics,
    ) -> Self::ScanDriver;
}

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions<F: FileType> {
    /// The underlying file reader.
    read: F::Read,
    /// File-specific options
    pub(crate) options: F::Options,
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
    segment_cache: Arc<dyn SegmentCache>,
    initial_read_size: u64,
    metrics: VortexMetrics,
}

impl<F: FileType> VortexOpenOptions<F> {
    pub(crate) fn new(read: F::Read, options: F::Options) -> Self {
        Self {
            read,
            options,
            registry: DEFAULT_REGISTRY.clone(),
            layout_registry: Arc::new(LayoutRegistry::default()),
            file_size: None,
            dtype: None,
            footer: None,
            segment_cache: Arc::new(NoOpSegmentCache),
            initial_read_size: 0,
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
    /// Once open, the [`Footer`] can be accessed via [`VortexFile::footer`].
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
    /// Open the Vortex file using asynchronous IO.
    pub async fn open(mut self) -> VortexResult<VortexFile<F>> {
        // If we need to read the file layout, then do so.
        let footer = match self.footer.take() {
            None => self.read_footer().await?,
            Some(footer) => footer,
        };

        // TODO(ngates): construct layout and array context from the footer + registry.

        Ok(VortexFile {
            read: self.read,
            options: self.options,
            footer,
            segment_cache: self.segment_cache,
            metrics: self.metrics,
            _marker: PhantomData,
        })
    }

    /// Read the [`Footer`] from the file.
    async fn read_footer(&self) -> VortexResult<Footer> {
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
        let mut initial_offset = file_size - initial_read_size;
        let mut initial_read: ByteBuffer = self
            .read
            .read_byte_range(initial_offset..file_size, Alignment::none())
            .await?;

        // We know the initial read _must_ contain at least the Postscript.
        let postscript = self.parse_postscript(&initial_read)?;

        // If we haven't been provided a DType, we must read one from the file.
        let dtype_segment = self.dtype.is_none().then(|| postscript.dtype.ok_or_else(|| vortex_err!("Vortex file doesn't embed a DType and one has not been provided to VortexOpenOptions"))).transpose()?;

        // Check if we need to read more bytes for the DType or Footer.
        let mut read_more_offset = initial_offset;

        // We always need to read the footer.
        if postscript.footer.offset < read_more_offset {
            read_more_offset = postscript.footer.offset;
        }
        // We sometimes need to read the DType.
        if let Some(dtype) = &dtype_segment {
            if dtype.offset < read_more_offset {
                read_more_offset = dtype.offset;
            }
        }

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
                &self
                    .read
                    .read_byte_range(read_more_offset..initial_offset, Alignment::none())
                    .await?,
            );
            new_initial_read.extend_from_slice(&initial_read);

            initial_offset = read_more_offset;
            initial_read = new_initial_read.freeze();
        }

        // Now we try to read the DType and Footer segments.
        let dtype = dtype_segment
            .map(|segment| self.parse_dtype(initial_offset, &initial_read, segment))
            .transpose()?
            .unwrap_or_else(|| self.dtype.clone().vortex_expect("DType was provided"));
        let footer = self.footer.clone().map(Ok).unwrap_or_else(|| {
            self.parse_footer(
                initial_offset,
                &initial_read,
                postscript.footer,
                dtype.clone(),
            )
        })?;

        // If the initial read happened to cover any segments, then we can populate the
        // segment cache
        self.populate_segments(initial_offset, &initial_read, &footer)
            .await?;

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
        dtype: SegmentSpec,
    ) -> VortexResult<DType> {
        let offset = usize::try_from(dtype.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::align_from(initial_read.slice(offset..offset + (dtype.length as usize)));
        let fbd_dtype = root::<fbd::DType>(&sliced_buffer)?;

        DType::try_from_view(fbd_dtype, sliced_buffer.clone())
    }

    /// Parse the Footer from the initial read.
    fn parse_footer(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        segment: SegmentSpec,
        dtype: DType,
    ) -> VortexResult<Footer> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let bytes =
            FlatBuffer::align_from(initial_read.slice(offset..offset + (segment.length as usize)));
        Footer::read_flatbuffer(bytes, dtype, &self.registry, &self.layout_registry)
    }

    /// Populate segments in the cache that were covered by the initial read.
    async fn populate_segments(
        &self,
        initial_offset: u64,
        initial_read: &ByteBuffer,
        footer: &Footer,
    ) -> VortexResult<()> {
        stream::iter(
            footer
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
