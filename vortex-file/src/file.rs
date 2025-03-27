use std::sync::Arc;

use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_layout::scan::ScanBuilder;
use vortex_layout::segments::AsyncSegmentReader;
use vortex_metrics::VortexMetrics;

use crate::footer::Footer;

#[derive(Clone)]
pub struct VortexFile {
    pub(crate) footer: Footer,
    pub(crate) segment_reader: Arc<dyn AsyncSegmentReader>,
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

    pub fn segment_reader(&self) -> &Arc<dyn AsyncSegmentReader> {
        &self.segment_reader
    }

    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }

    pub fn scan(&self) -> ScanBuilder {
        ScanBuilder::new(
            self.footer()
                .layout()
                .reader(self.segment_reader().clone(), self.footer.ctx().clone())
                // FIXME(ngates): why can this fail?
                .vortex_expect("failed to create layout reader"),
        )
    }
}
