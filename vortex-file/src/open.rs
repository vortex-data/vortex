// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRegistry;
use vortex_dtype::DType;
use vortex_io::runtime::Handle;
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;

use crate::footer::Footer;
use crate::DEFAULT_REGISTRY;

pub trait FileType: Sized {
    type Options;
}

/// Open options for a Vortex file reader.
pub struct VortexOpenOptions<F: FileType> {
    /// The handle used by the open file.
    pub(crate) handle: Option<Handle>,
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
    /// Create a new [`VortexOpenOptions`] with the expected options for the file source.
    ///
    /// This should not be used directly, instead public API clients are expected to
    /// access either `VortexOpenOptions::file()` or `VortexOpenOptions::memory()`
    pub(crate) fn new(options: F::Options) -> Self {
        Self {
            handle: Handle::find(),
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
