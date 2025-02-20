use std::marker::PhantomData;
use std::sync::Arc;

use vortex_array::stats::StatsSet;
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_layout::scan::ScanBuilder;
use vortex_metrics::VortexMetrics;

use crate::footer::FileLayout;
use crate::open::FileType;
use crate::segments::SegmentCache;

pub struct VortexFile<F: FileType> {
    pub(crate) read: F::Read,
    pub(crate) options: F::Options,
    pub(crate) ctx: ContextRef,
    pub(crate) file_layout: FileLayout,
    pub(crate) segment_cache: Arc<dyn SegmentCache>,
    pub(crate) metrics: VortexMetrics,
    pub(crate) _marker: PhantomData<F>,
}

impl<F: FileType> VortexFile<F> {
    pub fn row_count(&self) -> u64 {
        self.file_layout.row_count()
    }

    pub fn dtype(&self) -> &DType {
        self.file_layout.dtype()
    }

    pub fn file_layout(&self) -> &FileLayout {
        &self.file_layout
    }

    pub fn file_stats(&self) -> &[StatsSet] {
        self.file_layout.statistics()
    }

    pub fn scan(&self) -> ScanBuilder<F::ScanDriver> {
        let driver = F::scan_driver(
            self.read.clone(),
            self.options.clone(),
            self.file_layout.clone(),
            self.segment_cache.clone(),
            self.metrics.clone(),
        );
        ScanBuilder::new(
            driver,
            self.file_layout.root_layout().clone(),
            self.ctx.clone(),
        )
    }
}
