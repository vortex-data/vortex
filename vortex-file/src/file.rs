//! This module defines the [`VortexFile`] struct, which represents a Vortex file on disk or in memory.
//!
//! The `VortexFile` provides methods for accessing file metadata, creating segment sources for reading
//! data from the file, and initiating scans to read the file's contents into memory as Vortex arrays.
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::scan::ScanBuilder;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::VortexMetrics;

use crate::footer::Footer;

/// Represents a Vortex file, providing access to its metadata and content.
///
/// A `VortexFile` is created by opening a Vortex file using [`VortexOpenOptions`](crate::VortexOpenOptions).
/// It provides methods for accessing file metadata (such as row count, data type, and statistics)
/// and for initiating scans to read the file's contents.
#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file, containing metadata and layout information.
    pub(crate) footer: Footer,
    /// A factory for creating segment sources that read data from the file.
    pub(crate) segment_source_factory: Arc<dyn SegmentSourceFactory>,
    /// Metrics tied to the file.
    pub(crate) metrics: VortexMetrics,
}

impl VortexFile {
    /// Returns a reference to the file's footer, which contains metadata and layout information.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }

    /// Returns the data type of the file's contents.
    pub fn dtype(&self) -> &DType {
        self.footer.dtype()
    }

    /// Returns the file's statistics, if available.
    ///
    /// Statistics can be used for query optimization and data exploration.
    pub fn file_stats(&self) -> Option<&Arc<[StatsSet]>> {
        self.footer.statistics()
    }

    /// Returns a reference to the file's metrics.
    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }

    /// Create a new segment source for reading from the file.
    ///
    /// This may spawn a background I/O driver that will exit when the returned segment source
    /// is dropped.
    pub fn segment_source(&self) -> Arc<dyn SegmentSource> {
        self.segment_source_factory
            .segment_source(self.metrics.clone())
    }

    /// Create a new layout reader for the file.
    pub fn layout_reader(&self) -> VortexResult<Arc<dyn LayoutReader>> {
        let segment_source = self.segment_source();
        self.footer
            .layout()
            // TODO(ngates): we may want to allow the user pass in a name here?
            .new_reader(&"".into(), &segment_source, self.footer().ctx())
    }

    /// Initiate a scan of the file, returning a builder for configuring the scan.
    pub fn scan(&self) -> VortexResult<ScanBuilder<ArrayRef>> {
        Ok(ScanBuilder::new(self.layout_reader()?).with_metrics(self.metrics.clone()))
    }
}

/// A factory for creating segment sources that read data from a Vortex file.
///
/// This trait abstracts over different implementations of segment sources, allowing
/// for different I/O strategies (e.g., synchronous, asynchronous, memory-mapped)
/// to be used with the same file interface.
pub trait SegmentSourceFactory: 'static + Send + Sync {
    /// Create a segment source for reading segments from the file.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Metrics for monitoring the performance of the segment source.
    ///
    /// # Returns
    ///
    /// A new segment source that can be used to read data from the file.
    fn segment_source(&self, metrics: VortexMetrics) -> Arc<dyn SegmentSource>;
}
