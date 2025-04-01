use std::sync::Arc;

use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::VortexMetrics;

use crate::ScanBuilder;
use crate::footer::Footer;

#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file.
    pub(crate) footer: Footer,
    /// A source for reading segments from the file.
    pub(crate) segment_source: Arc<dyn SegmentSource>,
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

    pub fn segment_source(&self) -> &Arc<dyn SegmentSource> {
        &self.segment_source
    }

    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }

    /// Create a new layout reader for the file.
    pub fn layout_reader(&self) -> VortexResult<Arc<dyn LayoutReader>> {
        self.footer
            .layout()
            .reader(self.segment_source(), self.footer().ctx())
    }

    pub fn scan(&self) -> ScanBuilder {
        ScanBuilder::new(self.clone())
    }
}
