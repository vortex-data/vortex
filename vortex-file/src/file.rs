use std::marker::PhantomData;
use std::sync::Arc;

use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_layout::scan::ScanBuilder;
use vortex_layout::segments::AsyncSegmentReader;
use vortex_metrics::VortexMetrics;

use crate::footer::Footer;
use crate::open::FileType;
use crate::segments::SegmentCache;

pub struct VortexFile<F: FileType> {
    pub(crate) read: F::Read,
    pub(crate) options: F::Options,
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
}

pub trait VortexFileDyn {
    fn footer(&self) -> &Footer;

    fn row_count(&self) -> u64 {
        self.footer().row_count()
    }

    fn dtype(&self) -> &DType {
        self.footer().dtype()
    }

    fn file_stats(&self) -> Option<&Arc<[StatsSet]>> {
        self.footer().statistics()
    }

    fn metrics(&self) -> &VortexMetrics;

    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader>;

    fn scan(&self) -> ScanBuilder {
        let layout_reader = self
            .footer()
            .layout()
            .reader(self.segment_reader(), self.footer().ctx().clone())
            // FIXME(ngates): why can this fail?
            .vortex_expect("failed to create layout reader");
        ScanBuilder::new(layout_reader).with_metrics(self.metrics().clone())
    }
}

pub type VortexFileRef = Arc<dyn VortexFileDyn>;
