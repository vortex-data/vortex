use std::marker::PhantomData;
use std::sync::Arc;

use vortex_array::stats::StatsSet;
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_layout::scan::ScanBuilder;
use vortex_metrics::VortexMetrics;

use crate::footer::Footer;
use crate::open::FileType;
use crate::segments::SegmentCache;

pub struct VortexFile<F: FileType> {
    pub(crate) read: F::Read,
    pub(crate) options: F::Options,
    pub(crate) ctx: ContextRef,
    pub(crate) footer: Footer,
    pub(crate) segment_cache: Arc<dyn SegmentCache>,
    pub(crate) metrics: VortexMetrics,
    pub(crate) _marker: PhantomData<F>,
}

impl<F: FileType> VortexFile<F> {
    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }

    pub fn dtype(&self) -> &DType {
        self.footer.dtype()
    }

    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    pub fn file_stats(&self) -> Option<&Arc<[StatsSet]>> {
        self.footer.statistics()
    }

    pub fn scan(&self) -> ScanBuilder<F::ScanDriver> {
        let driver = F::scan_driver(
            self.read.clone(),
            self.options.clone(),
            self.footer.clone(),
            self.segment_cache.clone(),
            self.metrics.clone(),
        );
        ScanBuilder::new(driver, self.footer.layout().clone(), self.ctx.clone())
    }
}
