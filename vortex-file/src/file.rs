use std::sync::Arc;

use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::scan::ScanBuilder;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::VortexMetrics;

use crate::footer::Footer;

#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file.
    pub(crate) footer: Footer,
    /// A for reading segments from the file.
    pub(crate) segment_source_factory: Arc<dyn SegmentSourceFactory>,
    /// Metrics tied to the file.
    pub(crate) metrics: VortexMetrics,
}

impl VortexFile {
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }

    pub fn dtype(&self) -> &DType {
        self.footer.dtype()
    }

    pub fn file_stats(&self) -> Option<&Arc<[StatsSet]>> {
        self.footer.statistics()
    }

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
            .reader(&segment_source, self.footer().ctx())
    }

    /// Initiate a scan of the file, returning a builder for configuring the scan.
    pub fn scan(&self) -> VortexResult<ScanBuilder> {
        Ok(ScanBuilder::new(self.layout_reader()?).with_metrics(self.metrics.clone()))
    }
}

pub trait SegmentSourceFactory: 'static + Send + Sync {
    /// Create a segment source for reading segments from the file.
    fn segment_source(&self, metrics: VortexMetrics) -> Arc<dyn SegmentSource>;
}
