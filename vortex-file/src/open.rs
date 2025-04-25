use std::sync::Arc;

use flatbuffers::root;
use vortex_array::ArrayRegistry;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, ReadFlatBuffer, dtype as fbd};
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

use crate::footer::{FileStatistics, Footer, Postscript, PostscriptSegment};
use crate::{DEFAULT_REGISTRY, EOF_SIZE, MAGIC_BYTES, VERSION};

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
    pub(crate) footer: Option<Footer>,
    /// A metrics registry for the file.
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

    /// Configure a custom [`VortexMetrics`].
    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }
}

impl<F: FileType> VortexOpenOptions<F> {
    /// Parse the postscript from the initial read.
    pub(crate) fn parse_postscript(&self, initial_read: &[u8]) -> VortexResult<Postscript> {
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
    pub(crate) fn parse_dtype(
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
    pub(crate) fn parse_file_statistics(
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
    pub(crate) fn parse_footer(
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
